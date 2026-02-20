"""
Research Artifact Generator

Phase 3 requires at least one exportable research artifact. This module generates:
- Evidence table (Claim | Evidence snippet | Citation | Confidence | Notes)
- Annotated bibliography (8-12 sources with claim/method/limitations/why it matters)
- Synthesis memo (target 800-1200 words with inline citations + reference list)

The app passes gen_fn=rag._gen_text to enable LLM-powered artifact generation.
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import pandas as pd


class ArtifactGenerator:
    """Generates research artifacts from RAG results and the manifest."""

    def __init__(self, manifest_path: str = "data/data_manifest.csv"):
        self.manifest_path = manifest_path
        self.manifest = None
        self._load_manifest()

    def _load_manifest(self):
        encodings = ["utf-8-sig", "utf-8", "gbk", "cp936", "cp1252", "latin-1"]
        last_err = None
        for enc in encodings:
            try:
                self.manifest = pd.read_csv(self.manifest_path, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_err = e
        if self.manifest is None and last_err:
            raise last_err

    # -- Helpers ---------------------------------------------------------------
    def _extract_citations(self, text: str) -> List[Tuple[str, str]]:
        pattern = r"\(([^,\)]+),\s*([^,\)]+)\)"
        matches = re.findall(pattern, text or "")
        return [(source.strip(), chunk.strip()) for source, chunk in matches]

    def _get_source_metadata(self, source_id: str) -> Dict:
        if self.manifest is None:
            return {}
        row = self.manifest[self.manifest["source_id"] == source_id]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def _safe_json_from_text(self, text: str):
        """
        Extract JSON object or array from an LLM response.
        """
        if not text:
            return None
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try extract first {...} or [...]
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return None

    def _normalize_scores(self, retrieved_chunks: List[Dict]) -> Dict[Tuple[str, str], float]:
        """
        Normalize retrieval scores to [0, 1] per query result for display.
        """
        scores = []
        keys = []
        for ch in retrieved_chunks or []:
            try:
                s = float(ch.get("score", 0.0))
            except Exception:
                s = 0.0
            scores.append(s)
            keys.append((ch.get("source_id"), ch.get("chunk_id")))

        if not scores:
            return {}

        mn, mx = min(scores), max(scores)
        if mx == mn:
            return {k: 0.5 for k in keys}

        return {k: (s - mn) / (mx - mn) for k, s in zip(keys, scores)}

    def _confidence_label(self, norm_score: float) -> str:
        if norm_score >= 0.75:
            return "High"
        if norm_score >= 0.45:
            return "Medium"
        return "Low"

    def _reference_lines(self, source_ids: List[str]) -> List[str]:
        refs = []
        for sid in source_ids:
            meta = self._get_source_metadata(sid) or {}
            authors = meta.get("authors", "Unknown")
            year = meta.get("year", "Unknown")
            title = meta.get("title", sid)
            venue = meta.get("venue", "")
            url = meta.get("url_or_doi", "")
            parts = [f"- {authors} ({year}). *{title}*."]
            if venue and str(venue).lower() not in ("nan", "unknown"):
                parts.append(f"{venue}.")
            if url and str(url).lower() not in ("nan",):
                parts.append(str(url))
            refs.append(" ".join(parts).strip())
        return refs

    # -- Evidence table --------------------------------------------------------
    def generate_evidence_table(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict],
        citations: List[Tuple[str, str]],
        max_snippet_chars: int = 420,
    ) -> pd.DataFrame:
        """
        Evidence table schema (recommended):
        Claim | Evidence snippet | Citation (source_id, chunk_id) | Confidence | Notes
        """
        chunk_map = {
            (chunk.get("source_id"), chunk.get("chunk_id")): chunk for chunk in (retrieved_chunks or [])
        }
        norm = self._normalize_scores(retrieved_chunks)

        # Pull claim sentences that contain citations
        sentences = re.split(r"(?<=[\.\!\?])\s+|\n+", answer or "")
        rows = []

        for sentence in sentences:
            sentence = (sentence or "").strip()
            if not sentence:
                continue

            sent_citations = self._extract_citations(sentence)
            if not sent_citations:
                continue

            claim = re.sub(r"\([^)]+\)", "", sentence).strip()
            claim = re.sub(r"\s+", " ", claim)
            if not claim:
                continue

            for source_id, chunk_id in sent_citations:
                chunk = chunk_map.get((source_id, chunk_id))
                if not chunk:
                    continue

                txt = (chunk.get("text") or "").strip().replace("\n", " ")
                txt = re.sub(r"\s+", " ", txt)
                snippet = txt[:max_snippet_chars] + ("..." if len(txt) > max_snippet_chars else "")

                meta = self._get_source_metadata(source_id) or {}
                ns = float(norm.get((source_id, chunk_id), 0.5))
                rows.append(
                    {
                        "Claim": claim,
                        "Evidence Snippet": snippet,
                        "Citation": f"({source_id}, {chunk_id})",
                        "Confidence": self._confidence_label(ns),
                        "Notes": chunk.get("section", "unknown"),
                        "Source Title": meta.get("title", source_id),
                        "Year": meta.get("year", "Unknown"),
                    }
                )

        # Fallback if the answer is short or has no citations
        if not rows:
            for chunk in (retrieved_chunks or [])[:10]:
                source_id = chunk.get("source_id")
                chunk_id = chunk.get("chunk_id")
                meta = self._get_source_metadata(source_id) or {}
                txt = (chunk.get("text") or "").strip().replace("\n", " ")
                txt = re.sub(r"\s+", " ", txt)
                snippet = txt[:max_snippet_chars] + ("..." if len(txt) > max_snippet_chars else "")
                ns = float(norm.get((source_id, chunk_id), 0.5))
                rows.append(
                    {
                        "Claim": "Supporting evidence (answer had no extractable claim-citation sentences)",
                        "Evidence Snippet": snippet,
                        "Citation": f"({source_id}, {chunk_id})",
                        "Confidence": self._confidence_label(ns),
                        "Notes": chunk.get("section", "unknown"),
                        "Source Title": meta.get("title", source_id),
                        "Year": meta.get("year", "Unknown"),
                    }
                )

        return pd.DataFrame(rows)

    # -- Annotated bibliography ------------------------------------------------
    def generate_annotated_bibliography(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        citations_used: Optional[List[Tuple[str, str]]] = None,
        max_sources: int = 12,
        gen_fn=None,
    ) -> List[Dict]:
        """
        8-12 sources with fields: claim, method, limitations, why it matters.
        When gen_fn is provided, each entry is generated from source-specific evidence.
        """
        retrieved_chunks = retrieved_chunks or []
        citations_used = citations_used or []

        # Group chunks by source
        by_source: Dict[str, List[Dict]] = {}
        for ch in retrieved_chunks:
            by_source.setdefault(ch.get("source_id"), []).append(ch)

        # Prefer sources that were actually cited in the answer
        cited_sources = [sid for sid, _cid in citations_used]
        cited_sources = [s for s in cited_sources if s in by_source]

        # Fill with remaining sources by #chunks
        remaining = sorted(by_source.keys(), key=lambda s: len(by_source[s]), reverse=True)
        selected: List[str] = []
        for sid in cited_sources:
            if sid not in selected:
                selected.append(sid)
        for sid in remaining:
            if sid not in selected:
                selected.append(sid)
            if len(selected) >= max_sources:
                break

        # Ensure a minimum size if possible
        selected = selected[:max_sources]
        if len(selected) < 8:
            selected = remaining[: min(max_sources, max(8, len(remaining)))]

        bibliography: List[Dict] = []

        for source_id in selected:
            chunks = by_source.get(source_id, [])
            meta = self._get_source_metadata(source_id) or {}

            # Pick top evidence chunks for this source (by score if present)
            def _score(ch):
                try:
                    return float(ch.get("score", 0.0))
                except Exception:
                    return 0.0

            chunks_sorted = sorted(chunks, key=_score, reverse=True)
            top_chunks = chunks_sorted[:3]

            allowed_citations = [f"({c.get('source_id')}, {c.get('chunk_id')})" for c in top_chunks]
            evidence = "\n\n".join(
                f"{allowed_citations[i]}\n{(c.get('text') or '')[:900]}"
                for i, c in enumerate(top_chunks)
                if i < len(allowed_citations)
            ).strip()

            # Default placeholders
            entry = {
                "Source ID": source_id,
                "Title": meta.get("title", source_id),
                "Authors": meta.get("authors", "Unknown"),
                "Year": meta.get("year", "Unknown"),
                "Venue": meta.get("venue", "Unknown"),
                "DOI/URL": meta.get("url_or_doi", ""),
                "Claim": "",
                "Method": "",
                "Limitations": "",
                "Why it matters": "",
                "Key Citation": allowed_citations[0] if allowed_citations else "",
                "Chunks Retrieved": len(chunks),
            }

            if gen_fn is None or not evidence:
                # Heuristic fallback: compress evidence into a crude "claim"
                all_text = " ".join([(c.get("text") or "") for c in top_chunks]).strip()
                all_text = re.sub(r"\s+", " ", all_text)
                entry["Claim"] = all_text[:280] + ("..." if len(all_text) > 280 else "")
                entry["Method"] = "Not extracted (LLM disabled)."
                entry["Limitations"] = "Not extracted (LLM disabled)."
                entry["Why it matters"] = f"Relevant to: {query[:120]}"
                bibliography.append(entry)
                continue

            prompt = f"""You are writing an annotated bibliography entry for a personal research portal.

Research question:
{query}

Source metadata:
- source_id: {source_id}
- title: {meta.get('title', source_id)}
- year: {meta.get('year', 'Unknown')}
- venue: {meta.get('venue', 'Unknown')}

Evidence from this source (only use these excerpts):
{evidence}

Allowed citations (use exactly one as Key Citation):
{chr(10).join('- '+c for c in allowed_citations if c)}

Task:
Write 4 short fields grounded in the evidence:
- claim: 1-2 sentences summarizing the main contribution relevant to the research question
- method: 1 sentence describing the method or approach used
- limitations: 1 sentence about limitations or threats to validity (say "not stated in evidence" if missing)
- why_it_matters: 1 sentence explaining relevance to the research question
Also pick:
- key_citation: choose exactly one citation from the allowed list.

Return ONLY a JSON object with keys:
claim, method, limitations, why_it_matters, key_citation
No markdown, no commentary."""
            try:
                raw = gen_fn(prompt, retries=2)
                obj = self._safe_json_from_text(raw)
                if isinstance(obj, dict):
                    entry["Claim"] = (obj.get("claim") or "").strip()
                    entry["Method"] = (obj.get("method") or "").strip()
                    entry["Limitations"] = (obj.get("limitations") or "").strip()
                    entry["Why it matters"] = (obj.get("why_it_matters") or "").strip()
                    kc = (obj.get("key_citation") or "").strip()
                    if kc in allowed_citations:
                        entry["Key Citation"] = kc
                bibliography.append(entry)
            except Exception as e:
                entry["Claim"] = entry["Claim"] or "Extraction failed."
                entry["Method"] = f"Extraction error: {e}"
                entry["Limitations"] = ""
                entry["Why it matters"] = f"Relevant to: {query[:120]}"
                bibliography.append(entry)

        return bibliography

    # -- Synthesis memo --------------------------------------------------------
    def generate_synthesis_memo(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict],
        citations: List[Tuple[str, str]],
        word_count_target: int = 1000,
        gen_fn=None,
    ) -> str:
        """
        Synthesis memo: target 800-1200 words with inline citations and a reference list.
        When gen_fn is provided, the memo is regenerated from evidence to increase usefulness.
        """
        retrieved_chunks = retrieved_chunks or []
        citations = citations or []

        # Build allowed citations from retrieved evidence
        allowed = [(ch.get("source_id"), ch.get("chunk_id")) for ch in retrieved_chunks]
        allowed_set = set(allowed)
        allowed_list = [f"({a}, {b})" for a, b in allowed]

        def _validate(text: str):
            cits = self._extract_citations(text or "")
            invalid = [c for c in cits if c not in allowed_set]
            return cits, invalid

        # Evidence blocks
        max_chunks = 8
        evidence_blocks = []
        for ch in retrieved_chunks[:max_chunks]:
            evidence_blocks.append(
                f"({ch.get('source_id')}, {ch.get('chunk_id')})\n{(ch.get('text') or '')[:1100]}"
            )
        evidence = "\n\n".join(evidence_blocks).strip()

        if gen_fn is None or not evidence:
            # Simple fallback: wrap the existing answer as the memo
            source_ids = list(dict.fromkeys([c[0] for c in citations]))
            references = self._reference_lines(source_ids)
            memo = f"""# Synthesis Memo: {query}

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Research Question

{query}

## Synthesis

{answer}

## References

{chr(10).join(references)}

---
*Generated by Personal Research Portal*
"""
            return memo

        # LLM-powered memo generation
        target_low = int(os.getenv("MEMO_WORD_MIN", "800"))
        target_high = int(os.getenv("MEMO_WORD_MAX", "1200"))
        if not (500 <= target_low <= 2000 and 500 <= target_high <= 2500 and target_low < target_high):
            target_low, target_high = 800, 1200

        prompt = f"""You are producing a research synthesis memo for a personal research portal.

Rules:
1) Use ONLY information supported by the Evidence excerpts.
2) Every claim sentence must end with at least one citation in the format (source_id, chunk_id).
3) Use ONLY citations from the allowed list.
4) If evidence is missing or conflicting, say so and recommend what to retrieve next.

Research question:
{query}

Allowed citations:
{chr(10).join('- ' + c for c in allowed_list)}

Evidence excerpts:
{evidence}

Write a memo of {target_low}-{target_high} words with sections:
- Overview
- Key findings
- Where sources disagree (if applicable)
- Practical implications
- Remaining gaps and next retrieval steps

Return ONLY the memo body in Markdown (do not include a reference list)."""
        memo_body = ""
        try:
            memo_body = (gen_fn(prompt, retries=2) or "").strip()
        except Exception as e:
            memo_body = f"Memo generation failed: {e}"

        # Validate citations and repair once if needed
        cits, invalid = _validate(memo_body)
        if invalid or (not cits and allowed_list):
            repair_prompt = f"""Revise the memo so all citations are valid and drawn ONLY from the allowed list.
Every claim sentence must end with at least one citation.

Allowed citations:
{chr(10).join('- ' + c for c in allowed_list)}

Original memo:
{memo_body}

Return ONLY the revised memo body in Markdown."""
            try:
                memo_body = (gen_fn(repair_prompt, retries=2) or "").strip()
                cits, invalid = _validate(memo_body)
            except Exception:
                pass

        # Build references from citations in the memo
        used_source_ids = []
        seen = set()
        for sid, _cid in cits:
            if sid not in seen:
                seen.add(sid)
                used_source_ids.append(sid)

        references = self._reference_lines(used_source_ids)

        memo = f"""# Synthesis Memo: {query}

**Date**: {datetime.now().strftime('%Y-%m-%d')}

{memo_body}

## References

{chr(10).join(references)}

---
*Generated by Personal Research Portal*
"""
        return memo

    # -- Disagreement map and gap finder (stretch) -----------------------------
    def generate_disagreement_map(self, query: str, retrieved_chunks: List[Dict], gen_fn=None) -> pd.DataFrame:
        empty_cols = ["Aspect", "Source A", "Position A", "Source B", "Position B", "Conflict Type"]
        if not retrieved_chunks or gen_fn is None:
            return pd.DataFrame(columns=empty_cols)

        sources_text: Dict[str, List[str]] = {}
        for chunk in (retrieved_chunks or [])[:16]:
            sid = chunk.get("source_id")
            sources_text.setdefault(sid, []).append((chunk.get("text") or "")[:350])

        context_parts = []
        for sid, texts in list(sources_text.items())[:8]:
            meta = self._get_source_metadata(sid) or {}
            title = (meta.get("title") or sid)[:55]
            year = meta.get("year", "?")
            body = " ".join(texts)[:500]
            context_parts.append(f"[{sid}] {title} ({year}):\n{body}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are a systematic review analyst. Identify genuine disagreements or tensions between the sources below on the research topic.

Query: {query}

Sources:
{context}

Find up to 5 real disagreements (not just different emphasis). Return a JSON array where each item is:
{{
  "aspect": "short topic name (≤8 words)",
  "source_a_id": "exact source_id from above",
  "position_a": "what this source claims (max 25 words)",
  "source_b_id": "exact different source_id",
  "position_b": "what this source claims (max 25 words)",
  "conflict_type": "one of: Contradiction / Different scope / Methodological difference / Empirical disagreement"
}}

If fewer than 2 genuine disagreements exist, return [].
Return ONLY valid JSON array, no markdown, no explanation."""
        try:
            raw = gen_fn(prompt, retries=2)
            items = self._safe_json_from_text(raw)
            if not isinstance(items, list):
                return pd.DataFrame(columns=empty_cols)

            rows = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                meta_a = self._get_source_metadata(item.get("source_a_id", "")) or {}
                meta_b = self._get_source_metadata(item.get("source_b_id", "")) or {}
                label_a = f"{(meta_a.get('title') or item.get('source_a_id', ''))[:45]} ({meta_a.get('year', '?')})"
                label_b = f"{(meta_b.get('title') or item.get('source_b_id', ''))[:45]} ({meta_b.get('year', '?')})"
                rows.append(
                    {
                        "Aspect": item.get("aspect", ""),
                        "Source A": label_a,
                        "Position A": item.get("position_a", ""),
                        "Source B": label_b,
                        "Position B": item.get("position_b", ""),
                        "Conflict Type": item.get("conflict_type", ""),
                    }
                )
            return pd.DataFrame(rows, columns=empty_cols) if rows else pd.DataFrame(columns=empty_cols)
        except Exception as e:
            return pd.DataFrame(
                [
                    {
                        "Aspect": f"Analysis error: {e}",
                        "Source A": "",
                        "Position A": "",
                        "Source B": "",
                        "Position B": "",
                        "Conflict Type": "",
                    }
                ],
                columns=empty_cols,
            )

    def find_research_gaps(self, query: str, answer: str, retrieved_chunks: List[Dict], gen_fn=None) -> List[Dict]:
        if gen_fn is None:
            return []

        chunk_summaries = []
        for chunk in (retrieved_chunks or [])[:10]:
            meta = self._get_source_metadata(chunk.get("source_id")) or {}
            chunk_summaries.append(
                f"[{chunk.get('source_id')}, {chunk.get('chunk_id')}] "
                f"{(meta.get('title') or chunk.get('source_id'))[:50]}: "
                f"{(chunk.get('text') or '')[:220]}…"
            )
        context = "\n\n".join(chunk_summaries)

        prompt = f"""You are a research gap analyst. Given a research question and the evidence found, identify important aspects NOT covered by the retrieved evidence.

Research Question: {query}

Evidence retrieved:
{context}

Answer generated from evidence:
{(answer or '')[:700]}

Identify 3-5 specific research gaps. For each gap return a JSON object:
{{
  "gap_title": "concise gap name (5-8 words)",
  "what_is_missing": "what evidence or knowledge is absent (2 sentences max)",
  "evidence_would_help": "what type of source or study could fill this gap (1 sentence)",
  "suggested_query": "a concrete follow-up search query string (≤15 words)"
}}

Return ONLY a valid JSON array of gap objects, no markdown, no explanation."""
        try:
            raw = gen_fn(prompt, retries=2)
            gaps = self._safe_json_from_text(raw)
            if not isinstance(gaps, list):
                return []
            validated = []
            for g in gaps:
                if isinstance(g, dict) and g.get("gap_title"):
                    validated.append(
                        {
                            "gap_title": g.get("gap_title", "Unknown gap"),
                            "what_is_missing": g.get("what_is_missing", ""),
                            "evidence_would_help": g.get("evidence_would_help", ""),
                            "suggested_query": g.get("suggested_query", ""),
                        }
                    )
            return validated
        except Exception as e:
            return [
                {
                    "gap_title": "Analysis error",
                    "what_is_missing": str(e),
                    "evidence_would_help": "",
                    "suggested_query": "",
                }
            ]

    # -- All artifacts ---------------------------------------------------------
    def generate_all_artifacts(self, query: str, result: Dict, gen_fn=None) -> Dict:
        answer = result.get("answer", "")
        retrieved_chunks = result.get("retrieved_chunks", []) or []
        citations = result.get("citations_used", []) or []

        evidence_table = self.generate_evidence_table(query, answer, retrieved_chunks, citations)
        bibliography = self.generate_annotated_bibliography(
            query=query,
            retrieved_chunks=retrieved_chunks,
            citations_used=citations,
            max_sources=12,
            gen_fn=gen_fn,
        )
        synthesis_memo = self.generate_synthesis_memo(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            word_count_target=1000,
            gen_fn=gen_fn,
        )
        disagreement_map = self.generate_disagreement_map(query, retrieved_chunks, gen_fn=gen_fn)

        return {
            "evidence_table": evidence_table,
            "bibliography": bibliography,
            "synthesis_memo": synthesis_memo,
            "disagreement_map": disagreement_map,
        }
