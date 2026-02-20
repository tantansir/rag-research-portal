# Personal Research Portal (PRP)

A research-grade RAG portal for the **City Digital Twins** domain. The system ingests a domain corpus, retrieves evidence with hybrid search, produces citation-backed answers, and generates exportable research artifacts.

**Domain**: City Digital Twins, Urban AI, and Spatial Intelligence  
**Main Research Question**: How do city digital twins enable evidence-backed decision support for urban planning and management?

---

## Quick Start

### Prerequisites
- Python 3.10 or 3.11  
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Set environment variables
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY    = "your_key_here"      
$env:GEMINI_MODEL      = "gemini-2.5-flash"   # You can switch to other models such as gemini-2.5-pro, gemini-3-pro-preview
$env:GEMINI_EVAL_MODEL = "gemini-2.5-flash"
```
```bash
# macOS / Linux
export GEMINI_API_KEY="your_key_here"
export GEMINI_MODEL="gemini-2.5-flash"
export GEMINI_EVAL_MODEL="gemini-2.5-flash"
```

### 3 · Launch the web portal
```bash
python run_app.py
# or
streamlit run src/app/app.py
```

Opens at **http://localhost:8501**.

---

## Repository Structure

```
phase2-prp/
├── data/
│   ├── raw/                       # Downloaded PDFs (22 sources)
│   ├── processed/                 # Extracted text + chunk JSON per source
│   ├── embeddings/                # FAISS index, BM25 index, embeddings cache
│   └── data_manifest.csv          # Source metadata (22 sources, all required fields)
├── src/
│   ├── ingest/
│   │   ├── collect_corpus.py      # arXiv search + download; AuthorYear source_id generation
│   │   └── ingest_pipeline.py     # PDF parsing, cleaning, section-aware chunking
│   ├── rag/
│   │   └── rag_system.py          # Hybrid RRF retrieval + LLM reranking + citation generation
│   ├── eval/
│   │   ├── evaluator.py           # Groundedness / citation precision / answer relevance
│   │   └── query_set.json         # 22 fixed evaluation queries
│   └── app/
│       ├── app.py                 # Streamlit portal (Phase 3)
│       ├── thread_manager.py      # Save / search / delete research threads
│       ├── artifact_generator.py  # Evidence table, bibliography, synthesis memo, disagreement map
│       └── export_manager.py      # Markdown / CSV / PDF / BibTeX export
├── outputs/
│   ├── eval/                      # Evaluation CSVs, JSONL run logs, Markdown reports
│   ├── threads/                   # Saved research threads (JSON)
│   └── exports/                   # Generated and exported research artifacts
├── logs/                          # Per-day JSONL query logs
├── report/
│   ├── Phase1_Framing_Report.pdf
│   └── Phase2_Evaluation_Report.md
├── run_app.py                     # Launch script
├── requirements.txt               # Pinned dependencies
├── AI_USAGE_LOG.md
└── README.md
```

---

## Phase 3 — Research Portal (MVP + Stretch Goals)

### Core MVP Features

| Feature | Location in UI |
|---------|----------------|
| **Search & Ask** — hybrid retrieval, citation-backed answers | Navigation → Search & Ask |
| **Metadata filters** — year range + source type | Search & Ask → Filter panel |
| **Research Threads** — save / search / delete sessions | Navigation → Research Threads |
| **Evidence Table** — Claim · Evidence · Citation · Confidence · Notes | Artifacts → Evidence Table |
| **Annotated Bibliography** — 8–12 sources with 4 fields | Artifacts → Annotated Bibliography |
| **Synthesis Memo** — 800–1200 words with inline citations | Artifacts → Synthesis Memo |
| **Export** — Markdown / CSV / PDF | All artifact types |
| **Evaluation View** — run query set, metrics, baseline vs enhanced | Navigation → Evaluation |

### Stretch Goals Implemented

| Stretch Goal | Spec Reference | Location in UI |
|---|---|---|
| 🔭 **Gap Finder** | "Gap finder: missing evidence + targeted next retrieval actions" | Navigation → Gap Finder |
| ⚡ **Disagreement Map** | "Automatic disagreement map (conflicts surfaced with citations)" | Artifacts → Disagreement Map |
| 📄 **BibTeX Export** | "BibTeX export" | Artifacts → Annotated Bibliography → Export BibTeX |
| 🔧 **Metadata Filters** | "Filters by year/venue/type" | Search & Ask → Filter panel |

### Artifact Schemas

**Evidence Table** (CSV / PDF):  
`Claim | Evidence Snippet | Citation (source_id, chunk_id) | Confidence | Notes`

**Annotated Bibliography** (CSV / BibTeX):  
`Source ID | Title | Authors | Year | Venue | DOI/URL | Claim | Method | Limitations | Why it matters | Chunks Retrieved`

**Synthesis Memo** (Markdown / PDF):  
800–1200 words with inline `(source_id, chunk_id)` citations and a full reference list generated from the data manifest.

**Disagreement Map** (CSV):  
`Aspect | Source A | Position A | Source B | Position B | Conflict Type`

### Trust Behavior

- Every answer cites only retrieved chunks using `(source_id, chunk_id)`.
- A post-generation validation pass detects invalid citations; a repair pass corrects them.
- If evidence is absent the system explicitly states this rather than hallucinating.
- Filter zero-hit: if year/type filters eliminate all results, the app shows a warning instead of returning silently degraded output.

---

## Phase 2 — RAG System

### Corpus

| Property | Value |
|----------|-------|
| Total sources | 22 |
| Peer-reviewed journal articles | 11 |
| arXiv preprints | 10 |
| Technical reports | 1 |
| Year range | 2021–2025 |

All sources have: `source_id · title · authors · year · source_type · venue · url_or_doi · raw_path · processed_path · tags · relevance_note`

Source IDs follow the **AuthorYear** convention (e.g. `Luo2024`, `WEF2022`). Every citation maps back to a row in `data/data_manifest.csv`.

### Retrieval Pipeline

```
Query
  │
  ├─ Vector search  (FAISS, all-MiniLM-L6-v2)  ─┐
  │                                               ├─ RRF fusion → top-k candidates
  └─ Lexical search (BM25Okapi)                  ─┘
                │
                ▼
         Reference-section filter
                │
                ▼
         LLM reranker (Gemini, 400-char previews)
                │
                ▼
         Answer generation with strict citation constraint
                │
                ▼
         Citation validation + repair pass
```

**Enhancement**: Hybrid retrieval via **Reciprocal Rank Fusion (RRF)** + **LLM reranking**.  
RRF is robust to score-scale mismatch between BM25 and cosine similarity, which is why it outperforms the linear-score fusion used in the baseline.

### Evaluation

**Query set**: 22 queries — 11 direct, 6 synthesis/multi-hop, 5 edge-case / ambiguity  
**Metrics**: Groundedness (LLM judge 1–4) · Citation precision (exact match %) · Answer relevance (LLM judge 1–4)

Latest comparison results (see `outputs/eval/`):

| Metric | Baseline | Enhanced | Delta |
|---|---:|---:|---:|
| Groundedness (avg /4) | 2.91 | 3.91 | 1.00 |
| Citation precision (avg) | 99.43% | 99.65% | 0.22% |
| Answer relevance (avg /4) | 3.41 | 3.55 | 0.14 |

### Logging & Reproducibility

- Every query writes a JSONL record to `logs/query_log_YYYYMMDD.jsonl` containing: `run_id · query · prompt_version · prompt_hash · model · retrieval_config · retrieved_chunks · answer · citations_used`
- All dependencies are pinned in `requirements.txt`
- Embedding cache is validated against a SHA-1 of `all_chunks.json`; stale cache auto-rebuilds

---

## CLI Usage (Phase 2)

```bash
# Single query — vector-only baseline
python -m src.rag.rag_system --query "What is an urban digital twin?" --show_evidence

# Single query — enhanced (hybrid RRF + reranking)
python -m src.rag.rag_system \
  --query "What is an urban digital twin?" \
  --use_hybrid --use_reranking --top_k_after_rerank 5 \
  --show_evidence

# Run full evaluation (22 queries, enhanced)
python -m src.eval.evaluator

# Run baseline vs enhanced comparison
python -m src.eval.evaluator --compare

# Run comparison on a subset (faster, e.g. first 5 queries)
python -m src.eval.evaluator --compare --ids Q01,Q02,Q03,Q04,Q05
```

## Rebuilding the Corpus from Scratch

```bash
# Step 1: Download arXiv papers (optional — PDFs already in data/raw/)
python -m src.ingest.collect_corpus

# Step 2: Parse PDFs, chunk text, build all_chunks.json
python -m src.ingest.ingest_pipeline

# Step 3: Rebuild FAISS + BM25 index
#   (happens automatically on next app or rag_system startup)
```

---

## Citation Format

All answers use `(source_id, chunk_id)` inline citations.

Example: `(Luo2024, chunk_03)` resolves to:
- **Manifest row**: `data/data_manifest.csv` → row with `source_id = Luo2024`
- **Chunk text**: `data/processed/Luo2024_chunks.json` → item with `chunk_id = chunk_03`
- **Raw source**: `data/raw/Luo2024.pdf`

---

## Phase 1 Report

See `report/Phase1_Framing_Report.pdf` for the research framing, prompt kit (paper triage + claim-evidence extraction tasks), evaluation sheet (16 runs across 2 models × 2 prompts × 2 test cases), and analysis memo.