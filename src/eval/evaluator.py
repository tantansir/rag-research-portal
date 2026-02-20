"""
Evaluation Framework for Research-Grade RAG
Implements groundedness, citation precision, and answer relevance metrics
"""

import json
import re
from datetime import datetime
import pandas as pd
import google.generativeai as genai
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import inspect
import time
import random

try:
    from google.api_core import exceptions as gexc
except Exception:
    gexc = None

class RAGEvaluator:
    """Evaluates RAG system performance on research tasks"""

    def __init__(self, rag_system, eval_dir: str = "src/eval", output_dir: str = "outputs/eval"):
        self.rag = rag_system
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Gemini for evaluation
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.eval_model_name = os.getenv('GEMINI_EVAL_MODEL', 'gemini-2.5-pro')
        self.eval_model = genai.GenerativeModel(self.eval_model_name)

        # Load or create query set
        self.queries = self._load_query_set()

    def _eval_text(self, prompt: str, retries: int = 2) -> str:
        import time
        import random

        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self.eval_model.generate_content(prompt)
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

    def _filter_kwargs_for_callable(self, fn, kwargs: Dict) -> Dict:
        """Drop kwargs that the callable does not accept to avoid unexpected keyword errors."""
        if not kwargs:
            return {}
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return dict(kwargs)
            allowed = set(params.keys())
            return {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            return {}

    def _trim_chunks_for_eval_context(self, retrieved_chunks: List[Dict]) -> str:
        """Build a smaller context for judge prompts to reduce latency and timeouts."""
        max_chunks = int(os.getenv("EVAL_MAX_CONTEXT_CHUNKS", "6"))
        max_chars_per_chunk = int(os.getenv("EVAL_MAX_CHARS_PER_CHUNK", "800"))
        max_total_chars = int(os.getenv("EVAL_MAX_CONTEXT_CHARS", "6000"))

        parts: List[str] = []
        total = 0
        for chunk in (retrieved_chunks or [])[:max_chunks]:
            txt = (chunk.get("text") or "").strip().replace("\n", " ")
            if len(txt) > max_chars_per_chunk:
                txt = txt[:max_chars_per_chunk] + "..."
            block = f"[{chunk.get('source_id')}, {chunk.get('chunk_id')}]\n{txt}"
            if total + len(block) > max_total_chars:
                remaining = max_total_chars - total
                if remaining <= 0:
                    break
                block = block[:remaining]
            parts.append(block)
            total += len(block) + 2
            if total >= max_total_chars:
                break
        return "\n\n".join(parts)

    def _load_query_set(self) -> List[Dict]:
        """
        Load evaluation query set
        20+ queries: 10 direct, 5 synthesis/multi-hop, 5 edge cases
        """
        query_set_path = self.eval_dir / "query_set.json"
        
        if query_set_path.exists():
            with open(query_set_path, 'r') as f:
                return json.load(f)
        
        # Create comprehensive query set for city digital twins
        queries = [
            # Direct queries (10)
            {
                "id": "Q01",
                "type": "direct",
                "query": "What architectural components are critical for operational city digital twins?",
                "difficulty": "medium"
            },
            {
                "id": "Q02",
                "type": "direct",
                "query": "What benchmark datasets exist for evaluating city digital twin systems?",
                "difficulty": "medium"
            },
            {
                "id": "Q03",
                "type": "direct",
                "query": "How is real-time sensor data integrated into digital twin platforms?",
                "difficulty": "medium"
            },
            {
                "id": "Q04",
                "type": "direct",
                "query": "What human-centered design practices are used in digital twin interfaces?",
                "difficulty": "medium"
            },
            {
                "id": "Q05",
                "type": "direct",
                "query": "What are the most common use cases for city digital twins in urban planning?",
                "difficulty": "easy"
            },
            {
                "id": "Q06",
                "type": "direct",
                "query": "How do digital twins handle model drift and recalibration over time?",
                "difficulty": "hard"
            },
            {
                "id": "Q07",
                "type": "direct",
                "query": "What evaluation protocols are used to validate digital twin predictions?",
                "difficulty": "medium"
            },
            {
                "id": "Q08",
                "type": "direct",
                "query": "How do city digital twins support resilience and emergency response?",
                "difficulty": "medium"
            },
            {
                "id": "Q09",
                "type": "direct",
                "query": "What data pipeline architectures are commonly used in digital twin systems?",
                "difficulty": "medium"
            },
            {
                "id": "Q10",
                "type": "direct",
                "query": "What methods are used for uncertainty quantification in digital twin predictions?",
                "difficulty": "hard"
            },
            
            # Synthesis/multi-hop queries (5)
            {
                "id": "Q11",
                "type": "synthesis",
                "query": "Compare the evaluation methodologies used for digital twin systems across different application domains",
                "difficulty": "hard"
            },
            {
                "id": "Q12",
                "type": "synthesis",
                "query": "How do participatory design approaches differ across reported digital twin implementations?",
                "difficulty": "medium"
            },
            {
                "id": "Q13",
                "type": "synthesis",
                "query": "What are the key differences between building-level and city-level digital twin architectures?",
                "difficulty": "hard"
            },
            {
                "id": "Q14",
                "type": "synthesis",
                "query": "What tensions exist between real-time responsiveness and prediction accuracy in digital twin systems?",
                "difficulty": "hard"
            },
            {
                "id": "Q15",
                "type": "synthesis",
                "query": "How do different sources define the boundary between a digital twin and a traditional urban simulation?",
                "difficulty": "hard"
            },
            
            # Edge cases / ambiguity (5)
            {
                "id": "Q16",
                "type": "edge_case",
                "query": "Do city digital twins use blockchain for data provenance?",
                "difficulty": "hard",
                "expected_behavior": "Should acknowledge if evidence is absent"
            },
            {
                "id": "Q17",
                "type": "edge_case",
                "query": "What privacy concerns are raised about digital twin surveillance?",
                "difficulty": "medium",
                "expected_behavior": "May have limited evidence, should cite what exists"
            },
            {
                "id": "Q18",
                "type": "edge_case",
                "query": "Are there standardized APIs for interoperability between different digital twin platforms?",
                "difficulty": "hard",
                "expected_behavior": "Should check for standards/frameworks"
            },
            {
                "id": "Q19",
                "type": "edge_case",
                "query": "What is the typical computational cost of running a city-scale digital twin?",
                "difficulty": "hard",
                "expected_behavior": "Specific metrics may be sparse"
            },
            {
                "id": "Q20",
                "type": "edge_case",
                "query": "How do digital twins incorporate citizen-generated data from social media?",
                "difficulty": "medium",
                "expected_behavior": "Should note if evidence is limited"
            },
            {
                "id": "Q21",
                "type": "direct",
                "query": "What visualization techniques are used in digital twin dashboards?",
                "difficulty": "easy"
            },
            {
                "id": "Q22",
                "type": "synthesis",
                "query": "What validation strategies are recommended across multiple sources?",
                "difficulty": "medium"
            },
        ]
        
        # Save query set
        with open(query_set_path, 'w') as f:
            json.dump(queries, f, indent=2)
        
        print(f"Created query set with {len(queries)} queries")
        return queries

    def _normalize_for_report(self, text: str) -> str:
        """
        Normalize text for report readability.
        This does not change the stored corpus, only how snippets are shown in the report.
        """
        t = (text or "")
        t = t.replace("\u00a0", " ")
        t = t.replace("\n", " ")
        t = re.sub(r"\s+", " ", t).strip()

        # Light spacing fixes for common PDF extraction artifacts
        t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)  # camelCase -> camel Case
        t = re.sub(r"([A-Za-z])\.(?=[A-Za-z])", r"\1. ", t)  # "end.Start" -> "end. Start"
        t = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", t)  # "word,Next" -> "word, Next"
        t = re.sub(r"([A-Za-z])(\d{2,})", r"\1 \2", t)  # keep CO2, split long numbers
        t = re.sub(r"(\d{2,})([A-Za-z])", r"\1 \2", t)

        return t

    def extract_citations(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract citations from answer text
        Returns list of (source_id, chunk_id) tuples
        """
        # Pattern: (source_id, chunk_id)
        pattern = r'\(([^,\)]+),\s*([^,\)]+)\)'
        matches = re.findall(pattern, text)
        return [(source.strip(), chunk.strip()) for source, chunk in matches]

    def evaluate_groundedness(self, query: str, answer: str, retrieved_chunks):
        # Use trimmed context to avoid huge prompts and timeouts
        context = self._trim_chunks_for_eval_context(retrieved_chunks)

        prompt = f"""Evaluate groundedness of the answer against the evidence.

    Query: {query}

    Evidence:
    {context}

    Answer:
    {answer}

    Return EXACTLY:
    SCORE: <1-4>
    JUSTIFICATION: <2-3 sentences>

    Scoring guide:
    4 = Every factual claim is directly supported by evidence above; citations are correct.
    3 = Most claims supported; minor gaps or citation issues.
    2 = Some claims supported but key ones lack evidence or have wrong citations.
    1 = Hallucinated claims or fabricated citations not present in evidence.
    """
        try:
            txt = self._eval_text(prompt, retries=2)
            m = re.search(r"SCORE\s*[:：]\s*([1-4])", txt)
            if not m:
                return {"score": 1, "justification": "Judge output missing SCORE", "eval_response": txt}
            score = int(m.group(1))
            jm = re.search(r"JUSTIFICATION\s*[:：]\s*(.+)", txt, re.DOTALL)
            just = (jm.group(1).strip() if jm else "")
            return {"score": score, "justification": just, "eval_response": txt}
        except Exception as e:
            return {"score": 1, "justification": "Evaluation failed", "eval_response": str(e)}

    def evaluate_citation_precision(self, answer: str, retrieved_chunks: List[Dict]) -> Dict:
        """
        Check if citations in answer actually point to retrieved chunks
        """
        # Extract citations from answer
        citations = self.extract_citations(answer)
        
        # Build set of available citations
        available_citations = {
            (chunk['source_id'], chunk['chunk_id'])
            for chunk in retrieved_chunks
        }
        
        if not citations:
            return {
                'score': 1,
                'precision': 0.0,
                'total_citations': 0,
                'valid_citations': 0,
                'invalid_citations': []
            }
        
        # Check which citations are valid
        valid = []
        invalid = []
        
        for citation in citations:
            if citation in available_citations:
                valid.append(citation)
            else:
                invalid.append(citation)
        
        precision = len(valid) / len(citations) if citations else 0.0
        
        # Convert to 1-4 scale
        if precision >= 0.95:
            score = 4
        elif precision >= 0.80:
            score = 3
        elif precision >= 0.50:
            score = 2
        else:
            score = 1
        
        return {
            'score': score,
            'precision': precision,
            'total_citations': len(citations),
            'valid_citations': len(valid),
            'invalid_citations': invalid
        }

    def evaluate_answer_relevance(self, query: str, answer: str):
        prompt = f"""Evaluate relevance.

    Question: {query}
    Answer: {answer}

    Return EXACTLY:
    SCORE: <1-4>
    REASON: <1-2 sentences>
    """
        try:
            txt = self._eval_text(prompt, retries=2)
            m = re.search(r"SCORE\s*[:：]\s*([1-4])", txt)
            if not m:
                return {"score": 1, "eval_response": txt}
            return {"score": int(m.group(1)), "eval_response": txt}
        except Exception as e:
            return {"score": 1, "eval_response": str(e)}

    def evaluate_single_query(self, query_info: Dict, run_rag: bool = True,
                              rag_query_kwargs: Optional[Dict] = None) -> Dict:
        """Evaluate a single query and return a metrics row."""
        query_id = query_info.get("id", "")
        query_text = query_info.get("query", "")

        rag_query_kwargs = rag_query_kwargs or {}
        if "log" not in rag_query_kwargs:
            rag_query_kwargs["log"] = False

        safe_kwargs = self._filter_kwargs_for_callable(self.rag.query, rag_query_kwargs)

        try:
            result = self.rag.query(query_text, query_id=query_id, **safe_kwargs)
            answer = result.get("answer", "")
            retrieved_chunks = result.get("retrieved_chunks", []) or []
            rag_error = ""
        except Exception as e:
            answer = ""
            retrieved_chunks = []
            rag_error = str(e)

        if rag_error:
            groundedness = {"score": 1, "justification": f"RAG query failed: {rag_error}", "eval_response": rag_error}
            answer_relevance = {"score": 1, "eval_response": rag_error}
        else:
            groundedness = self.evaluate_groundedness(query_text, answer, retrieved_chunks)
            answer_relevance = self.evaluate_answer_relevance(query_text, answer)

        citation_precision = self.evaluate_citation_precision(answer, retrieved_chunks)

        return {
            "query_id": query_id,
            "query": query_text,
            "query_type": query_info.get("type", ""),
            "answer": answer,
            "num_retrieved": len(retrieved_chunks),
            "groundedness_score": groundedness.get("score", 2),
            "groundedness_justification": groundedness.get("justification", ""),
            "citation_precision_score": citation_precision.get("score", 1),
            "citation_precision_value": citation_precision.get("precision", 0.0),
            "total_citations": citation_precision.get("total_citations", 0),
            "valid_citations": citation_precision.get("valid_citations", 0),
            "answer_relevance_score": answer_relevance.get("score", 2),
            "timestamp": datetime.now().isoformat(),
            "error": rag_error,
        }

    def run_evaluation(
            self,
            query_ids: List[str] = None,
            run_label: str = "enhanced",
            rag_query_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        if query_ids is None:
            queries_to_eval = self.queries
        else:
            queries_to_eval = [q for q in self.queries if q["id"] in query_ids]

        rag_query_kwargs = rag_query_kwargs or {}

        if "log" not in rag_query_kwargs:
            rag_query_kwargs["log"] = False

        print(f"Running evaluation on {len(queries_to_eval)} queries ({run_label})...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_path = self.output_dir / f"eval_runs_{run_label}_{timestamp}.jsonl"

        results: List[Dict] = []
        run_records: List[Dict] = []

        for query_info in queries_to_eval:
            query_id = query_info["id"]
            query_text = query_info["query"]

            print()
            print(f"Evaluating {query_id}: {query_text}...")

            rag_result = self.rag.query(query_text, query_id=query_id, **rag_query_kwargs)
            answer = rag_result["answer"]
            retrieved_chunks = rag_result["retrieved_chunks"]

            retrieval_cfg = rag_result.get("retrieval_config") or {}
            answer_has_citations = bool(self.extract_citations(answer))

            groundedness = self.evaluate_groundedness(query_text, answer, retrieved_chunks)
            citation_precision = self.evaluate_citation_precision(answer, retrieved_chunks)
            answer_relevance = self.evaluate_answer_relevance(query_text, answer)

            eval_row = {
                "query_id": query_id,
                "query": query_text,
                "query_type": query_info["type"],
                "run_label": run_label,

                "use_hybrid": retrieval_cfg.get("use_hybrid"),
                "use_reranking": retrieval_cfg.get("use_reranking"),
                "k": retrieval_cfg.get("k"),
                "k_raw": retrieval_cfg.get("k_raw"),
                "answer_has_citations": answer_has_citations,

                "answer": answer,
                "num_retrieved": len(retrieved_chunks),
                "groundedness_score": groundedness["score"],
                "groundedness_parse_failed": groundedness.get("parse_failed", False),
                "groundedness_justification": groundedness.get("justification", ""),
                "citation_precision_score": citation_precision["score"],
                "citation_precision_value": citation_precision["precision"],
                "total_citations": citation_precision["total_citations"],
                "valid_citations": citation_precision["valid_citations"],
                "invalid_citations_count": len(citation_precision["invalid_citations"]),
                "invalid_citations": citation_precision["invalid_citations"],
                "answer_relevance_score": answer_relevance["score"],
                "answer_relevance_parse_failed": answer_relevance.get("parse_failed", False),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(eval_row)

            run_record = {
                "run_label": run_label,
                "query_id": query_id,
                "query": query_text,
                "query_type": query_info["type"],
                "rag_result": {
                    "run_id": rag_result.get("run_id"),
                    "prompt_version": rag_result.get("prompt_version"),
                    "prompt_hash": rag_result.get("prompt_hash"),
                    "model": rag_result.get("model"),
                    "retrieval_config": rag_result.get("retrieval_config"),
                    "rag_query_kwargs": rag_query_kwargs,
                    "answer": answer,
                    "retrieved_chunks": retrieved_chunks,
                    "citations_used": rag_result.get("citations_used"),
                    "invalid_citations": rag_result.get("invalid_citations"),
                },
                "metrics": {
                    "groundedness": groundedness,
                    "citation_precision": citation_precision,
                    "answer_relevance": answer_relevance,
                },
                "timestamp": eval_row["timestamp"],
            }
            run_records.append(run_record)

        with open(runs_path, "w", encoding="utf-8") as rf:
            for rec in run_records:
                rf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        df = pd.DataFrame(results)

        results_path = self.output_dir / f"eval_results_{run_label}_{timestamp}.csv"
        df.to_csv(results_path, index=False, encoding="utf-8")

        self.latest_runs_path = str(runs_path)
        self.latest_results_path = str(results_path)

        print()
        print(f"Results saved to: {results_path}")
        print(f"Runs saved to: {runs_path}")

        return df

    def run_enhancement_comparison(self, query_ids: List[str] = None, fast: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Compare baseline vs enhanced retrieval settings.

        Baseline: vector-only, no reranking.
        Enhanced: hybrid RRF retrieval + LLM reranking.
          (Demonstrates two distinct improvements over baseline.)

        Args:
            query_ids: subset of query IDs to run; None = full set.
            fast: if True, run only the first 10 queries (for quick iteration).
        """
        if fast and query_ids is None:
            # Use a balanced fast subset: ~4 direct + 3 synthesis + 3 edge_case
            type_counts = {"direct": 0, "synthesis": 0, "edge_case": 0}
            limits = {"direct": 4, "synthesis": 3, "edge_case": 3}
            fast_ids = []
            for q in self.queries:
                qt = q.get("type", "direct")
                if type_counts.get(qt, 0) < limits.get(qt, 0):
                    fast_ids.append(q["id"])
                    type_counts[qt] = type_counts.get(qt, 0) + 1
            query_ids = fast_ids
            print(f"[fast mode] Running {len(query_ids)} queries: {query_ids}")

        baseline_kwargs = {
            "use_hybrid": False,
            "use_reranking": False,
            "log": False,
        }
        enhanced_kwargs = {
            "use_hybrid": True,  # Enhancement 1: hybrid RRF retrieval
            "use_reranking": True,  # Enhancement 2: LLM reranking
            "top_k_after_rerank": 5,
            "log": False,
        }

        print("\n--- Running BASELINE evaluation ---")
        baseline_df = self.run_evaluation(
            query_ids=query_ids,
            run_label="baseline",
            rag_query_kwargs=baseline_kwargs,
        )
        print("\n--- Running ENHANCED evaluation (hybrid RRF + reranking) ---")
        enhanced_df = self.run_evaluation(
            query_ids=query_ids,
            run_label="enhanced",
            rag_query_kwargs=enhanced_kwargs,
        )
        return {"baseline": baseline_df, "enhanced": enhanced_df}

    def generate_eval_report(
            self,
            results_df: pd.DataFrame,
            output_path: str = None,
            runs_path: str = None,
            baseline_df: Optional[pd.DataFrame] = None,
            enhanced_df: Optional[pd.DataFrame] = None,
    ):
        """Generate evaluation report (Markdown).

        If baseline_df and enhanced_df are provided, the report will include an
        Enhancement Impact section with before/after metric deltas.
        """
        if output_path is None:
            output_path = self.output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        out = Path(output_path)

        run_map: Dict[str, Dict] = {}
        chosen_runs: Optional[Path] = None

        if runs_path:
            chosen_runs = Path(runs_path)
        elif hasattr(self, "latest_runs_path"):
            chosen_runs = Path(getattr(self, "latest_runs_path"))
        else:
            candidates = sorted(self.output_dir.glob("eval_runs_*.jsonl"))
            if candidates:
                chosen_runs = candidates[-1]

        if chosen_runs and chosen_runs.exists():
            try:
                with open(chosen_runs, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        qid = rec.get("query_id")
                        if qid:
                            run_map[qid] = rec
            except Exception as e:
                print(f"Warning: failed to load runs file: {e}")

        qdf = pd.DataFrame(self.queries)
        q_type_counts = qdf.groupby("type").size().to_dict() if not qdf.empty else {}
        q_diff_counts = (
            qdf.groupby("difficulty").size().to_dict()
            if (not qdf.empty and "difficulty" in qdf.columns)
            else {}
        )

        example_run = None
        if not results_df.empty:
            first_qid = str(results_df.iloc[0]["query_id"])
            example_run = run_map.get(first_qid)

        example_cfg = None
        example_model = None
        if example_run:
            rr = (example_run.get("rag_result", {}) or {})
            example_cfg = rr.get("retrieval_config")
            example_model = rr.get("model")

        report = f"""# RAG System Evaluation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Total Queries Evaluated**: {len(results_df)}

## Query Set Design

This evaluation uses a fixed query set stored at `src/eval/query_set.json`.

- Query type counts: {q_type_counts}
- Difficulty counts: {q_diff_counts}

The set includes direct questions, cross-source synthesis questions, and edge cases that should trigger explicit missing-evidence behavior.

## System Configuration (This Run)

- Generator model: {example_model}
- Retrieval configuration snapshot: {example_cfg}

## Metrics

Groundedness is judged against retrieved evidence. Citation precision checks whether citations in the answer match retrieved chunks. Answer relevance rates whether the answer addresses the question.

## Implementation Notes

Retrieval uses a hybrid of semantic vector search and lexical BM25. Answers are generated with strict citation constraints that only allow citing retrieved chunks, and a repair pass runs when the model outputs citations outside the allowed set.

## Summary Metrics (This Run)

| Metric | Average Score | Notes |
|--------|---------------|-------|
| Groundedness | {results_df['groundedness_score'].mean():.2f}/4 | LLM-judged faithfulness to retrieved evidence |
| Citation Precision | {results_df['citation_precision_value'].mean():.2%} | Fraction of citations that match retrieved chunks |
| Answer Relevance | {results_df['answer_relevance_score'].mean():.2f}/4 | LLM-judged relevance to the question |

"""

        if baseline_df is not None and enhanced_df is not None and not baseline_df.empty and not enhanced_df.empty:
            def _avg(df, col):
                return float(df[col].mean())

            b_g = _avg(baseline_df, "groundedness_score")
            e_g = _avg(enhanced_df, "groundedness_score")
            b_c = _avg(baseline_df, "citation_precision_value")
            e_c = _avg(enhanced_df, "citation_precision_value")
            b_r = _avg(baseline_df, "answer_relevance_score")
            e_r = _avg(enhanced_df, "answer_relevance_score")

            report += f"""## Enhancement Impact (Baseline vs Enhanced)

Baseline settings: vector-only retrieval, no LLM reranking.

Enhanced settings: hybrid retrieval (BM25 + vector) plus LLM reranking.

| Metric | Baseline | Enhanced | Delta |
|--------|----------:|---------:|------:|
| Groundedness (avg /4) | {b_g:.2f} | {e_g:.2f} | {(e_g - b_g):+.2f} |
| Citation Precision (avg) | {b_c:.2%} | {e_c:.2%} | {(e_c - b_c):+.2%} |
| Answer Relevance (avg /4) | {b_r:.2f} | {e_r:.2f} | {(e_r - b_r):+.2f} |

"""

            # By query type deltas (helps explain which query families improved)
            try:
                metrics_cols = [
                    "groundedness_score",
                    "citation_precision_value",
                    "answer_relevance_score",
                ]

                b_by = (
                    baseline_df.groupby("query_type")[metrics_cols]
                    .mean()
                    .rename(
                        columns={
                            "groundedness_score": "groundedness_baseline",
                            "citation_precision_value": "citation_precision_baseline",
                            "answer_relevance_score": "relevance_baseline",
                        }
                    )
                )
                e_by = (
                    enhanced_df.groupby("query_type")[metrics_cols]
                    .mean()
                    .rename(
                        columns={
                            "groundedness_score": "groundedness_enhanced",
                            "citation_precision_value": "citation_precision_enhanced",
                            "answer_relevance_score": "relevance_enhanced",
                        }
                    )
                )

                by_type = b_by.join(e_by, how="outer")
                by_type["groundedness_delta"] = by_type["groundedness_enhanced"] - by_type["groundedness_baseline"]
                by_type["citation_precision_delta"] = (
                        by_type["citation_precision_enhanced"] - by_type["citation_precision_baseline"]
                )
                by_type["relevance_delta"] = by_type["relevance_enhanced"] - by_type["relevance_baseline"]

                by_type_fmt = by_type.copy()

                for c in [
                    "citation_precision_baseline",
                    "citation_precision_enhanced",
                    "citation_precision_delta",
                ]:
                    if c in by_type_fmt.columns:
                        by_type_fmt[c] = by_type_fmt[c].map(lambda x: f"{float(x):.2%}" if pd.notna(x) else "")

                for c in [
                    "groundedness_baseline",
                    "groundedness_enhanced",
                    "groundedness_delta",
                    "relevance_baseline",
                    "relevance_enhanced",
                    "relevance_delta",
                ]:
                    if c in by_type_fmt.columns:
                        by_type_fmt[c] = by_type_fmt[c].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")

                report += """### Enhancement Impact by Query Type

        """
                report += by_type_fmt.to_markdown()
                report += """

        """

            except Exception as e:
                report += f"""

        (Warning: could not compute per-type deltas: {e})

        """

        report += "## Breakdown by Query Type\n\n"

        type_breakdown = results_df.groupby("query_type").agg(
            groundedness_avg=("groundedness_score", "mean"),
            citation_precision_avg=("citation_precision_value", "mean"),
            relevance_avg=("answer_relevance_score", "mean"),
            count=("query_id", "count"),
        )
        report += type_breakdown.to_markdown()

        report += "\n\n## Per-Query Summary\n\n"
        per_query = results_df[
            ["query_id", "query_type", "groundedness_score", "citation_precision_value", "answer_relevance_score"]
        ].copy()
        per_query = per_query.sort_values("query_id")
        per_query["citation_precision_value"] = per_query["citation_precision_value"].map(
            lambda x: f"{float(x):.2%}")
        report += per_query.to_markdown(index=False)

        report += "\n\n## Best Performing Queries\n\n"

        best_tmp = results_df.copy()
        best_tmp["composite_score"] = (
            best_tmp["groundedness_score"].astype(float)
            + best_tmp["answer_relevance_score"].astype(float)
            + best_tmp["citation_precision_value"].astype(float) * 4.0
        )
        best_queries = best_tmp.nlargest(3, "composite_score")[
            ["query_id", "query", "groundedness_score", "citation_precision_value", "answer_relevance_score"]
        ]

        for _, row in best_queries.iterrows():
            report += (
                f"\n**{row['query_id']}**: {row['query']}\n"
                f"- Groundedness: {row['groundedness_score']} | "
                f"Citation precision: {row['citation_precision_value']:.2%} | "
                f"Relevance: {row['answer_relevance_score']}\n"
            )

        report += "\n\n## Failure Cases (Representative)\n\n"

        failures = results_df[
            (results_df["groundedness_score"] < 4)
            | (results_df["citation_precision_value"] < 0.80)
            | (results_df["answer_relevance_score"] < 3)
            ].copy()

        if failures.empty:
            failures = results_df.nsmallest(3, "citation_precision_value").copy()

        failures = failures.sort_values(
            ["citation_precision_value", "groundedness_score", "answer_relevance_score"],
            ascending=[True, True, True],
        ).head(3)

        for _, row in failures.iterrows():
            qid = row["query_id"]
            report += f"\n**{qid}**: {row['query']}\n"
            report += (
                f"- Groundedness: {row['groundedness_score']} | "
                f"Citation precision: {row['citation_precision_value']:.2%} | "
                f"Relevance: {row['answer_relevance_score']}\n"
            )
            just = str(row.get("groundedness_justification", "")).strip()
            if just:
                report += f"- Grounding note: {self._normalize_for_report(just)}\n"

            invalid = row.get("invalid_citations", [])
            report += f"- Invalid citations: {invalid}\n"

            rec = run_map.get(qid)
            if rec:
                chunks = (rec.get("rag_result", {}) or {}).get("retrieved_chunks", [])[:2]
                if chunks:
                    report += "- Evidence snippets:\n"
                    for c in chunks:
                        snippet = self._normalize_for_report(c.get("text") or "")
                        report += f"  - ({c.get('source_id')}, {c.get('chunk_id')}): {snippet}\n"

        if chosen_runs and chosen_runs.exists():
            report += f"\n\n## Run Logs\n\nMachine-readable runs: {chosen_runs.as_posix()}\n"

        report += "\n\n## Limitations and Next Steps\n\nThe evaluation uses an LLM judge, so scores can be noisy. For follow-up, consider adding deterministic retrieval metrics, expanding edge cases for missing evidence, and validating citation alignment against longer evidence spans.\n"

        if baseline_df is None and enhanced_df is None:
            report += (
                "\n\n## Notes\n\n"
                "To quantify the impact of enhancements, run evaluation with `--compare` to produce baseline and enhanced runs and include the Enhancement Impact section.\n"
            )

        with open(out, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Report saved to: {out}")
        return str(out)


def main():
    import sys, argparse
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.rag.rag_system import ResearchRAG

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run the full query set")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated query IDs, e.g. Q01,Q02")
    parser.add_argument("--compare", action="store_true", help="Run baseline vs enhanced comparison")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: run only a 10-query subset (for quick iteration)")
    args = parser.parse_args()

    rag = ResearchRAG()
    evaluator = RAGEvaluator(rag)

    query_ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None

    if args.compare:
        dfs = evaluator.run_enhancement_comparison(query_ids=query_ids, fast=args.fast)
        evaluator.generate_eval_report(
            results_df=dfs["enhanced"],
            baseline_df=dfs["baseline"],
            enhanced_df=dfs["enhanced"],
        )
    else:
        results_df = evaluator.run_evaluation(query_ids=query_ids, run_label="enhanced")
        evaluator.generate_eval_report(results_df)

if __name__ == "__main__":
    main()
