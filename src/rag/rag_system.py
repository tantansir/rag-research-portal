"""
Research-Grade RAG System with Resolvable Citations

Core guarantees for the PRP project:
- Retrieval over an ingested corpus with metadata (data/data_manifest.csv)
- Answers constrained to retrieved evidence
- Inline citations in the exact format (source_id, chunk_id)
- Logs that make every run reproducible and auditable
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import google.generativeai as genai
import uuid
import hashlib
import re
import time
import random


class ResearchRAG:
    """Research-grade RAG system with hybrid retrieval + strict, resolvable citations."""

    def __init__(
        self,
        chunks_path: str = "data/processed/all_chunks.json",
        manifest_path: str = "data/data_manifest.csv",
        embeddings_dir: str = "data/embeddings",
        logs_dir: str = "logs",
    ):
        self.chunks_path = chunks_path
        self.manifest_path = manifest_path
        self.embeddings_dir = Path(embeddings_dir)
        self.logs_dir = Path(logs_dir)

        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Version tags to make runs traceable
        self.prompt_version = os.getenv("PROMPT_VERSION", "phase3_strict_citations_v3")

        # Citation patterns
        self.citation_pattern = re.compile(r"\(([^,\)]+),\s*([^,\)]+)\)")
        self.citation_tail_pattern = re.compile(r"(?:\s*\(([^,\)]+),\s*([^,\)]+)\)\s*)+$")

        # Load corpus
        self.chunks = self._load_chunks()
        self.chunks_by_key = {
            (c.get("source_id"), c.get("chunk_id")): c for c in self.chunks
        }

        # Load manifest with encoding fallback
        encodings = ["utf-8-sig", "utf-8", "gbk", "cp936", "cp1252", "latin-1"]
        last_err = None
        for enc in encodings:
            try:
                self.manifest = pd.read_csv(manifest_path, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_err = e
        else:
            raise last_err

        # Normalize manifest paths to be OS-agnostic
        for col in ["raw_path", "processed_path"]:
            if col in self.manifest.columns:
                self.manifest[col] = (
                    self.manifest[col]
                    .fillna("")
                    .astype(str)
                    .str.replace("\\", "/", regex=False)
                )

        # Embedding model
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Retrievers
        self.vector_index = None
        self.bm25 = None
        self._setup_retrievers()

        # Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.model = genai.GenerativeModel(self.model_name)

        # Counters for UI
        self.llm_call_count = 0

    # -- LLM helper ------------------------------------------------------------
    def _gen_text(self, prompt: str, generation_config=None, retries: int = 2) -> str:
        last_err = None
        for attempt in range(retries + 1):
            try:
                self.llm_call_count += 1
                if generation_config is not None:
                    resp = self.model.generate_content(prompt, generation_config=generation_config)
                else:
                    resp = self.model.generate_content(prompt)
                return (getattr(resp, "text", "") or "").strip()
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                transient = ("504" in msg) or ("deadline" in msg) or ("timeout" in msg) or ("unavailable" in msg)
                if transient and attempt < retries:
                    time.sleep(0.8 * (attempt + 1) + random.random() * 0.2)
                    continue
                raise
        raise last_err

    # -- Loading ---------------------------------------------------------------
    def _load_chunks(self) -> List[Dict]:
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
        return chunks

    # -- Retriever caching -----------------------------------------------------
    def _setup_retrievers(self):
        """Setup both vector and BM25 retrievers, rebuild if cache is stale."""
        index_path = self.embeddings_dir / "faiss_index.bin"
        embeddings_path = self.embeddings_dir / "embeddings.pkl"
        bm25_path = self.embeddings_dir / "bm25.pkl"
        meta_path = self.embeddings_dir / "retriever_meta.json"

        # Fingerprint current chunks file so cache mismatches are caught
        try:
            chunks_sha1 = hashlib.sha1(Path(self.chunks_path).read_bytes()).hexdigest()
        except Exception:
            chunks_sha1 = None
        n_chunks = len(self.chunks)

        def rebuild():
            print("Rebuilding retrievers (cache missing or stale)...")
            self._build_retrievers()
            meta = {
                "chunks_path": str(self.chunks_path),
                "num_chunks": n_chunks,
                "chunks_sha1": chunks_sha1,
                "created_at": datetime.now().isoformat(),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if index_path.exists() and embeddings_path.exists() and bm25_path.exists() and meta_path.exists():
            print("Loading cached retrievers...")
            try:
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
            except Exception:
                meta = {}

            try:
                self.vector_index = faiss.read_index(str(index_path))
                with open(embeddings_path, "rb") as f:
                    self.embeddings = pickle.load(f)
                with open(bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
            except Exception as e:
                print(f"Cache load failed: {e}")
                rebuild()
                return

            ok = True
            if meta.get("num_chunks") != n_chunks:
                ok = False
            if chunks_sha1 is not None and meta.get("chunks_sha1") != chunks_sha1:
                ok = False

            try:
                if int(getattr(self.embeddings, "shape", [0])[0]) != n_chunks:
                    ok = False
            except Exception:
                ok = False

            try:
                if int(getattr(self.vector_index, "ntotal", 0)) != n_chunks:
                    ok = False
            except Exception:
                ok = False

            try:
                if len(getattr(self.bm25, "doc_freqs", [])) != n_chunks:
                    ok = False
            except Exception:
                ok = False

            if ok:
                print("Retrievers loaded from cache")
                return

            print("Cached retrievers do not match current chunks, rebuilding...")
            rebuild()
            return

        rebuild()

    def _build_retrievers(self):
        chunk_texts = [chunk.get("text", "") for chunk in self.chunks]

        print("Generating embeddings...")
        self.embeddings = self.embed_model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(self.embeddings.astype("float32"))

        print("Building BM25 index...")
        tokenized_chunks = [(c.get("text") or "").lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        print("Caching retrievers...")
        faiss.write_index(self.vector_index, str(self.embeddings_dir / "faiss_index.bin"))
        with open(self.embeddings_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)
        with open(self.embeddings_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        print("Retrievers built and cached")

    # -- Retrieval -------------------------------------------------------------
    def retrieve_vector(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        distances, indices = self.vector_index.search(query_embedding.astype("float32"), k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[int(idx)]
            similarity = 1 / (1 + float(distance))
            results.append((chunk, similarity))
        return results

    def retrieve_bm25(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]
        results = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            results.append((chunk, float(scores[int(idx)])))
        return results

    def hybrid_retrieve(
        self,
        query: str,
        k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> List[Tuple[Dict, float]]:
        """
        Hybrid retrieval via Reciprocal Rank Fusion (RRF).

        score(d) = w_vec / (rrf_k + rank_vec(d)) + w_bm25 / (rrf_k + rank_bm25(d))
        """
        # Pull more candidates from each retriever for better fusion
        cand_n = max(int(k) * 4, 40)

        vector_results = self.retrieve_vector(query, k=cand_n)
        bm25_results = self.retrieve_bm25(query, k=cand_n)

        # Build rank maps (1-based)
        vec_rank = {}
        for r, (chunk, _s) in enumerate(vector_results, 1):
            vec_rank[(chunk.get("source_id"), chunk.get("chunk_id"))] = r

        bm_rank = {}
        for r, (chunk, _s) in enumerate(bm25_results, 1):
            bm_rank[(chunk.get("source_id"), chunk.get("chunk_id"))] = r

        all_keys = set(vec_rank.keys()) | set(bm_rank.keys())
        fused = []
        for key in all_keys:
            score = 0.0
            if key in vec_rank:
                score += float(vector_weight) / (float(rrf_k) + float(vec_rank[key]))
            if key in bm_rank:
                score += float(bm25_weight) / (float(rrf_k) + float(bm_rank[key]))
            chunk = self.chunks_by_key.get(key)
            if chunk:
                fused.append((chunk, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[: int(k)]

    def rerank_with_llm(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Dict, float]],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        if not retrieved_chunks:
            return []

        top_k = max(1, min(int(top_k), len(retrieved_chunks)))

        def _build_prompt(preview_len: int) -> str:
            chunks_text = ""
            for i, (chunk, _score) in enumerate(retrieved_chunks, 1):
                preview = (chunk.get("text") or "")[:preview_len].replace("\n", " ")
                chunks_text += f"\n{i}. [{chunk.get('source_id')}, {chunk.get('chunk_id')}]\n{preview}\n"
            return f"""Rank the following chunks by relevance to the query.
Return ONLY a comma-separated list of the top {top_k} chunk numbers (1-{len(retrieved_chunks)}) in order of relevance.

Query: {query}

Chunks:
{chunks_text}

Ranking (numbers only):"""

        preview_lens = [160, 130, 100, 80]
        for preview_len in preview_lens:
            try:
                ranking_text = self._gen_text(_build_prompt(preview_len), retries=2)

                rankings = []
                for x in (ranking_text or "").split(","):
                    x = x.strip()
                    if x.isdigit():
                        idx = int(x) - 1
                        if 0 <= idx < len(retrieved_chunks) and idx not in rankings:
                            rankings.append(idx)

                reranked = [retrieved_chunks[i] for i in rankings]
                if len(reranked) < top_k:
                    for i, c in enumerate(retrieved_chunks):
                        if i not in rankings:
                            reranked.append(c)
                            if len(reranked) >= top_k:
                                break
                return reranked[:top_k]
            except Exception as e:
                print(f"Reranking failed (preview_len={preview_len}): {e}")

        return retrieved_chunks[:top_k]

    # -- Evidence block building ------------------------------------------------
    def _build_evidence_blocks(self, retrieved_chunks: List[Tuple[Dict, float]]) -> str:
        blocks: List[str] = []
        for chunk, _score in retrieved_chunks:
            citation = f"({chunk['source_id']}, {chunk['chunk_id']})"
            blocks.append(f"{citation}\n{chunk.get('text', '')}")
        return "\n\n".join(blocks)

    def _build_evidence_blocks_limited(self, retrieved_chunks: List[Tuple[Dict, float]]) -> str:
        """Build evidence blocks with truncation to reduce prompt size and timeouts."""
        max_chunks = int(os.getenv("GEN_MAX_EVIDENCE_CHUNKS", "8"))
        max_chunk_chars = int(os.getenv("GEN_MAX_CHUNK_CHARS", "1200"))
        max_total_chars = int(os.getenv("GEN_MAX_EVIDENCE_CHARS", "16000"))

        blocks: List[str] = []
        total = 0
        for chunk, _score in (retrieved_chunks or [])[:max_chunks]:
            citation = f"({chunk['source_id']}, {chunk['chunk_id']})"
            txt = (chunk.get("text") or "").strip()
            if len(txt) > max_chunk_chars:
                txt = txt[:max_chunk_chars] + "..."
            block = f"{citation}\n{txt}"
            if total + len(block) > max_total_chars:
                remaining = max_total_chars - total
                if remaining <= 0:
                    break
                block = block[:remaining]
            blocks.append(block)
            total += len(block) + 2
            if total >= max_total_chars:
                break

        return "\n\n".join(blocks)

    # -- Citation validation ---------------------------------------------------
    def _extract_citations(self, text: str) -> List[Tuple[str, str]]:
        return [(a.strip(), b.strip()) for a, b in self.citation_pattern.findall(text or "")]

    def _sentence_candidates(self, answer: str) -> List[str]:
        """
        Produce a conservative list of sentence-like segments that should carry citations.
        This is used only for validation, not for generation.
        """
        if not answer:
            return []

        # Normalize newlines a bit but keep bullets as separate lines
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        segments: List[str] = []

        for ln in lines:
            # Skip code blocks and separators
            if ln.startswith("```") or ln.strip() in ("---", "***"):
                continue

            # Headings
            if ln.startswith("#"):
                continue

            # Bullets: keep as one segment
            if re.match(r"^(\-|\*|•|\d+\.)\s+", ln):
                segments.append(ln)
                continue

            # Split by sentence end punctuation
            parts = re.split(r"(?<=[\.\!\?])\s+", ln)
            for p in parts:
                p = p.strip()
                if p:
                    segments.append(p)

        return segments

    def _needs_citation(self, segment: str) -> bool:
        """
        Decide if a segment should have citations.
        We keep it simple: any segment containing letters/digits is treated as claim-like,
        unless it is clearly a references label.
        """
        s = (segment or "").strip()
        if not s:
            return False

        lower = s.lower()
        if lower in ("references", "reference", "bibliography"):
            return False

        # If it's purely citations or punctuation, ignore
        if re.fullmatch(r"[\(\)\s,;:\.\[\]\---]+", s):
            return False
        if re.fullmatch(r"(?:\s*\([^,\)]+,\s*[^,\)]+\)\s*)+", s):
            return False

        return bool(re.search(r"[A-Za-z0-9]", s))

    def _missing_sentence_citations(self, answer: str) -> List[str]:
        """
        Return segments that look like claims but do not end with a citation tail.
        """
        missing: List[str] = []
        for seg in self._sentence_candidates(answer):
            if not self._needs_citation(seg):
                continue
            if not self.citation_tail_pattern.search(seg.strip()):
                missing.append(seg)
        return missing

    def _validate_citations(self, answer: str, allowed: set) -> Dict:
        citations = self._extract_citations(answer)
        invalid = [c for c in citations if c not in allowed]
        missing_sentence_cits = self._missing_sentence_citations(answer) if allowed else []
        return {
            "citations": citations,
            "invalid": invalid,
            "missing_sentence_citations": missing_sentence_cits,
        }

    def _repair_answer_with_allowed_citations(
        self,
        query: str,
        draft_answer: str,
        retrieved_chunks: List[Tuple[Dict, float]],
        allowed_citations: List[str],
    ) -> str:
        evidence = self._build_evidence_blocks_limited(retrieved_chunks)
        allowed_list = "\n".join(f"- {c}" for c in allowed_citations)

        prompt = f"""Rewrite the draft answer so that every claim is supported by the provided evidence.

Rules:
1) Use ONLY citations from the allowed list.
2) Each citation must be its own parentheses in the exact format (source_id, chunk_id).
3) Every claim sentence must end with at least one citation.
4) If evidence is missing or conflicting, say so and suggest what to search for next.

Allowed citations:
{allowed_list}

Question:
{query}

Evidence:
{evidence}

Draft answer:
{draft_answer}

Return only the revised answer text."""
        return self._gen_text(prompt, retries=2)

    # -- Answer generation -----------------------------------------------------
    def _no_evidence_answer(self, query: str) -> str:
        return (
            "The current corpus does not contain enough direct evidence to answer this question with citations.\n\n"
            "Suggested next steps:\n"
            "- Try rephrasing the query with key terms and synonyms.\n"
            "- Narrow the question to one sub-claim, then search again.\n"
            "- Add 2-3 targeted sources to the corpus (e.g., a survey, a standard, or an empirical study) and re-run ingestion."
        )

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Dict, float]],
        use_strict_citations: bool = True,
    ) -> Dict:
        # If retrieval returns nothing, refuse to guess
        if not retrieved_chunks:
            answer = self._no_evidence_answer(query)
            return {
                "run_id": str(uuid.uuid4()),
                "query": query,
                "answer": answer,
                "retrieved_chunks": [],
                "model": self.model_name,
                "prompt_version": self.prompt_version,
                "prompt_hash": "",
                "citations_used": [],
                "invalid_citations": [],
                "timestamp": datetime.now().isoformat(),
            }

        evidence = self._build_evidence_blocks_limited(retrieved_chunks)
        allowed_citations = [f"({chunk['source_id']}, {chunk['chunk_id']})" for chunk, _ in retrieved_chunks]
        allowed_set = {(chunk["source_id"], chunk["chunk_id"]) for chunk, _ in retrieved_chunks}

        if use_strict_citations:
            allowed_list = "\n".join(f"- {c}" for c in allowed_citations)
            system_prompt = f"""You are a research assistant that produces citation-backed answers.

Rules:
1) Use ONLY information supported by the evidence.
2) Every claim sentence must end with at least one citation in the format (source_id, chunk_id).
3) Each citation must be its own parentheses.
4) Use ONLY citations from the allowed list.
5) If evidence is missing or conflicting, say so.

Allowed citations:
{allowed_list}
"""
        else:
            system_prompt = "You are a helpful research assistant."

        user_prompt = f"""Answer the question using only the evidence below.

Question:
{query}

Evidence:
{evidence}

Write a structured answer with inline citations in (source_id, chunk_id) format.
"""
        full_prompt = f"{system_prompt}\n{user_prompt}"
        prompt_hash = hashlib.sha1(full_prompt.encode("utf-8")).hexdigest()

        answer = (self._gen_text(full_prompt, retries=2) or "").strip()

        validation = self._validate_citations(answer, allowed_set)

        # Detect citation bundling like "(a,b; c,d)" which is hard to resolve
        malformed_multi = bool(re.search(r"\([^)]*;[^)]*\)", answer or ""))

        need_repair = (
            use_strict_citations
            and (
                not validation["citations"]
                or validation["invalid"]
                or validation["missing_sentence_citations"]
                or malformed_multi
            )
        )

        if need_repair:
            try:
                answer = self._repair_answer_with_allowed_citations(
                    query=query,
                    draft_answer=answer,
                    retrieved_chunks=retrieved_chunks,
                    allowed_citations=allowed_citations,
                ).strip()
                validation = self._validate_citations(answer, allowed_set)
            except Exception as e:
                print(f"Answer repair failed: {e}")

        # Final fallback: if still no citations, append top sources as a minimal trace
        if use_strict_citations and not validation["citations"] and allowed_citations:
            top = allowed_citations[: min(3, len(allowed_citations))]
            answer = (answer + "\n\nSources: " + " ".join(top)).strip()
            validation = self._validate_citations(answer, allowed_set)

        return {
            "run_id": str(uuid.uuid4()),
            "query": query,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "source_id": chunk.get("source_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text", ""),
                    "score": float(score),
                    "section": chunk.get("section", "unknown"),
                    "char_start": chunk.get("char_start"),
                    "char_end": chunk.get("char_end"),
                }
                for chunk, score in retrieved_chunks
            ],
            "model": self.model_name,
            "prompt_version": self.prompt_version,
            "prompt_hash": prompt_hash,
            "citations_used": validation["citations"],
            "invalid_citations": validation["invalid"],
            "missing_sentence_citations": validation["missing_sentence_citations"],
            "timestamp": datetime.now().isoformat(),
        }

    # -- Filtering helpers ------------------------------------------------------
    def _is_reference_like(self, chunk: dict) -> bool:
        sec = (chunk.get("section") or "").strip().lower()
        if "reference" in sec or "bibliograph" in sec:
            return True

        text = (chunk.get("text") or "").strip().lower()
        if text.startswith("references"):
            return True

        doi_hits = len(re.findall(r"\b10\.\d{4,9}/\S+\b", text))
        year_hits = len(re.findall(r"\b(19|20)\d{2}\b", text))
        url_hits = len(re.findall(r"https?://", text))
        bracket_hits = len(re.findall(r"\[\d+\]", text))

        if doi_hits >= 2:
            return True
        if (year_hits >= 8 and bracket_hits >= 5) or url_hits >= 3:
            return True

        return False

    def _filter_retrieved_chunks(self, retrieved: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        return [(c, s) for (c, s) in (retrieved or []) if not self._is_reference_like(c)]

    def _get_allowed_source_ids(
        self,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        source_types: Optional[List[str]] = None,
    ) -> Optional[set]:
        if year_min is None and year_max is None and not source_types:
            return None

        df = self.manifest.copy()
        df["_year_num"] = pd.to_numeric(df.get("year"), errors="coerce")

        if year_min is not None:
            df = df[df["_year_num"] >= year_min]
        if year_max is not None:
            df = df[df["_year_num"] <= year_max]
        if source_types:
            df = df[df["source_type"].isin(source_types)]

        return set(df["source_id"].tolist())

    # -- Main query ------------------------------------------------------------
    def query(
        self,
        query: str,
        k: int = 10,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        top_k_after_rerank: int = 5,
        query_id: Optional[str] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        log: bool = True,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        source_types: Optional[List[str]] = None,
        **_ignored_kwargs,
    ) -> Dict:
        """
        Main query interface.

        Extra filter params
        year_min / year_max: restrict to sources in this year range.
        source_types: restrict to selected source_type values.
        """
        k = int(k)
        k_raw = max(k * 6, 40)

        # Retrieve candidates
        if use_hybrid:
            retrieved_raw = self.hybrid_retrieve(
                query,
                k=k_raw,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )
        else:
            retrieved_raw = self.retrieve_vector(query, k=k_raw)

        # Filter out reference-like chunks for better grounding
        retrieved = self._filter_retrieved_chunks(retrieved_raw)

        # Metadata filter
        allowed_ids = self._get_allowed_source_ids(year_min, year_max, source_types)
        filter_hit_zero = False

        if allowed_ids is not None:
            filtered = [(c, s) for c, s in retrieved if c.get("source_id") in allowed_ids]
            if filtered:
                retrieved = filtered
            else:
                filter_hit_zero = True
                retrieved = []

        # Rerank
        need_n = int(top_k_after_rerank if use_reranking else k)
        if retrieved and len(retrieved) < need_n:
            # If filtering removed too much, extend with raw candidates as a last resort
            retrieved = (retrieved + retrieved_raw)[: max(need_n * 4, need_n)]

        if use_reranking and retrieved:
            candidate_cap = min(len(retrieved), max(need_n * 4, 20), 40)
            retrieved = self.rerank_with_llm(query, retrieved[:candidate_cap], top_k=need_n)
        else:
            retrieved = retrieved[:k]

        result = self.generate_answer(query, retrieved)
        result["query_id"] = query_id
        result["retrieval_config"] = {
            "k": k,
            "k_raw": k_raw,
            "use_hybrid": use_hybrid,
            "use_reranking": use_reranking,
            "top_k_after_rerank": top_k_after_rerank,
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "year_min": year_min,
            "year_max": year_max,
            "source_types": source_types,
        }
        result["filter_hit_zero"] = filter_hit_zero

        if log:
            self._log_query(result)

        return result

    # -- Logging ---------------------------------------------------------------
    def _log_query(self, result: Dict):
        log_file = self.logs_dir / f"query_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # -- Resolvers for UI / debugging ------------------------------------------
    def get_source_metadata(self, source_id: str) -> Dict:
        row = self.manifest[self.manifest["source_id"] == source_id]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def resolve_citation(self, source_id: str, chunk_id: str) -> Optional[Dict]:
        """Resolve a (source_id, chunk_id) pair to the ingested chunk dict."""
        return self.chunks_by_key.get((source_id, chunk_id))

    def get_raw_path(self, source_id: str) -> str:
        meta = self.get_source_metadata(source_id)
        return (meta.get("raw_path") or "").strip()

    def _normalize_for_display(self, text: str) -> str:
        t = (text or "")
        t = t.replace("\u00a0", " ").replace("\n", " ")
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
        t = re.sub(r"([A-Za-z])\.(?=[A-Za-z])", r"\1. ", t)
        t = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", t)
        return t


def main():
    """CLI entrypoint for running a single RAG query."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single query against the research-grade RAG system and write a log entry."
    )
    parser.add_argument("--query", type=str, default="", help="Question to ask")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--use_hybrid", action="store_true", help="Enable hybrid retrieval (BM25 + vector via RRF)")
    parser.add_argument("--use_reranking", action="store_true", help="Enable LLM-based reranking")
    parser.add_argument("--top_k_after_rerank", type=int, default=5, help="How many chunks to keep after reranking")
    parser.add_argument("--vector_weight", type=float, default=0.5, help="RRF weight for vector retriever")
    parser.add_argument("--bm25_weight", type=float, default=0.5, help="RRF weight for BM25 retriever")
    parser.add_argument("--show_evidence", action="store_true", help="Print retrieved evidence blocks")
    args = parser.parse_args()

    rag = ResearchRAG()
    if not args.query.strip():
        print("Please provide --query")
        return

    result = rag.query(
        query=args.query,
        k=args.k,
        use_hybrid=args.use_hybrid,
        use_reranking=args.use_reranking,
        top_k_after_rerank=args.top_k_after_rerank,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        log=True,
    )

    print("\nAnswer:\n")
    print(result.get("answer", ""))

    if args.show_evidence:
        print("\nRetrieved evidence:\n")
        for c in result.get("retrieved_chunks", [])[:8]:
            print(f"({c['source_id']}, {c['chunk_id']}) [{c.get('section', 'unknown')}] score={c.get('score')}")
            print((c.get("text") or "")[:900])
            print("-" * 60)


if __name__ == "__main__":
    main()
