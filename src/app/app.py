"""
Personal Research Portal - Phase 3
Streamlit Web Application
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import time
from typing import Optional, Dict, Any, Tuple

from src.rag.rag_system import ResearchRAG
from src.app.thread_manager import ThreadManager
from src.app.artifact_generator import ArtifactGenerator
from src.app.export_manager import ExportManager
from src.eval.evaluator import RAGEvaluator


# -- page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Personal Research Portal",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #1f4788; margin-bottom: 0.4rem;
    }
    .sub-header {
        font-size: 1.05rem; color: #666; margin-bottom: 1.6rem;
    }
    .citation {
        background-color: #eef2fa; padding: 0.25rem 0.6rem;
        border-radius: 4px; font-family: monospace; font-size: 0.88em;
    }
    .source-card {
        border-left: 4px solid #1f4788; padding: 0.9rem;
        margin: 0.45rem 0; background-color: #f9f9f9;
    }
    .gap-card {
        border-left: 4px solid #e67e22; padding: 0.9rem;
        margin: 0.6rem 0; background-color: #fdf6ee;
        border-radius: 0 6px 6px 0;
    }
    .conflict-badge {
        display: inline-block; background: #e74c3c;
        color: white; padding: 0.15rem 0.5rem;
        border-radius: 10px; font-size: 0.8em;
    }
    .stAlert { margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# -- cached loaders ------------------------------------------------------------
@st.cache_resource
def load_rag_system():
    return ResearchRAG()

@st.cache_resource
def load_thread_manager():
    return ThreadManager()

@st.cache_resource
def load_artifact_generator():
    return ArtifactGenerator()

@st.cache_resource
def load_export_manager():
    return ExportManager()


def _resolve_repo_path(path_str: str) -> Optional[Path]:
    """Resolve a manifest path (usually under data/raw/) to a local file path."""
    if not path_str:
        return None
    p = str(path_str).strip().replace('\\', '/')
    cand = (ROOT / p).resolve()
    if cand.exists():
        return cand
    # Sometimes manifests store just the filename
    cand2 = (ROOT / 'data' / 'raw' / Path(p).name).resolve()
    if cand2.exists():
        return cand2
    return None


def _to_url(url_or_doi: str) -> Optional[str]:
    if not url_or_doi:
        return None
    u = str(url_or_doi).strip()
    if u.startswith("10."):
        return "https://doi.org/" + u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return None


def _resolve_cited_snippet(rag: ResearchRAG, citation: tuple, retrieved_chunks: list) -> Optional[dict]:
    """Return a resolved chunk dict for a citation, prefer retrieved chunks then full corpus."""
    sid, cid = citation
    for ch in (retrieved_chunks or []):
        if ch.get("source_id") == sid and ch.get("chunk_id") == cid:
            return ch
    full = rag.resolve_citation(sid, cid)
    if full:
        return {"source_id": sid, "chunk_id": cid, "text": full.get("text", ""), "section": full.get("section", "unknown")}
    return None


# -- main ---------------------------------------------------------------------
def main():
    try:
        rag          = load_rag_system()
        thread_mgr   = load_thread_manager()
        artifact_gen = load_artifact_generator()
        export_mgr   = load_export_manager()
    except Exception as e:
        st.error(f"Failed to initialise system: {e}")
        st.info("Make sure GEMINI_API_KEY is set.")
        return

    st.sidebar.title("Personal Research Portal")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🔍 Search & Ask", "🔭 Gap Finder", "📚 Research Threads",
         "📊 Artifacts", "📈 Evaluation", "ℹ️ About"],
    )

    if page == "🔍 Search & Ask":
        show_search_page(rag, thread_mgr, artifact_gen, export_mgr)
    elif page == "🔭 Gap Finder":
        show_gap_finder_page(rag, artifact_gen, thread_mgr)
    elif page == "📚 Research Threads":
        show_threads_page(thread_mgr, artifact_gen, export_mgr)
    elif page == "📊 Artifacts":
        show_artifacts_page(rag, thread_mgr, artifact_gen, export_mgr)
    elif page == "📈 Evaluation":
        show_evaluation_page(rag)
    elif page == "ℹ️ About":
        show_about_page()


# -- Search & Ask --------------------------------------------------------------
def show_search_page(rag, thread_mgr, artifact_gen, export_mgr):
    st.markdown('<div class="main-header">🔍 Search & Ask</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask a research question and get evidence-backed answers</div>',
                unsafe_allow_html=True)

    query = st.text_area(
        "Enter your research question:",
        height=100,
        placeholder="E.g., What architectural components are critical for operational city digital twins?",
    )

    # -- retrieval options --
    col1, col2, col3 = st.columns(3)
    with col1:
        use_hybrid   = st.checkbox("Hybrid retrieval (RRF)", value=True,
                                   help="BM25 + vector via Reciprocal Rank Fusion")
    with col2:
        use_reranking = st.checkbox("LLM reranking", value=True,
                                    help="Re-order candidates with an LLM pass")
    with col3:
        top_k = st.slider("Chunks to retrieve", 5, 20, 10)

    # -- metadata filters (stretch goal #4) --
    with st.expander("🔧 Filter by source metadata", expanded=False):
        years = sorted(
            pd.to_numeric(rag.manifest["year"], errors="coerce").dropna().astype(int).unique()
        )
        year_min_val, year_max_val = (min(years), max(years)) if years else (2020, 2025)

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            year_range = st.slider(
                "Publication year range",
                min_value=year_min_val,
                max_value=year_max_val,
                value=(year_min_val, year_max_val),
                key="search_year_range",
            )
        with f_col2:
            all_types = sorted(rag.manifest["source_type"].dropna().unique().tolist())
            sel_types = st.multiselect(
                "Source types",
                options=all_types,
                default=all_types,
                key="search_source_types",
                help="Leave all selected to search across all types",
            )

    active_year_min = year_range[0] if year_range[0] > year_min_val else None
    active_year_max = year_range[1] if year_range[1] < year_max_val else None
    active_types    = sel_types if sel_types and set(sel_types) != set(all_types) else None

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a research question.")
        else:
            with st.spinner("Retrieving evidence and generating answer…"):
                try:
                    result = rag.query(
                        query=query,
                        k=top_k,
                        use_hybrid=use_hybrid,
                        use_reranking=use_reranking,
                        top_k_after_rerank=min(top_k, 10),
                        year_min=active_year_min,
                        year_max=active_year_max,
                        source_types=active_types,
                    )
                    st.session_state["last_result"] = result
                    st.session_state["last_query"]  = query
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    st.exception(e)

    result          = st.session_state.get("last_result")
    query_for_result = st.session_state.get("last_query", query)

    if result:
        # -- filter zero-hit warning --------------------------------------
        if result.get("filter_hit_zero"):
            st.warning(
                "⚠️ **No sources matched your filters** - the year range or source type "
                "combination returned zero chunks. Showing an empty result. "
                "Try widening the filter or clearing it."
            )
            st.stop()

        # Show active filter badge
        filter_parts = []
        if active_year_min or active_year_max:
            filter_parts.append(f"years {year_range[0]}-{year_range[1]}")
        if active_types:
            filter_parts.append(f"types: {', '.join(active_types)}")
        if filter_parts:
            st.caption(f"🔧 Filtered by: {' · '.join(filter_parts)}")

        st.markdown("### 📝 Answer")
        st.markdown(result["answer"])

        invalid = result.get("invalid_citations", []) or []
        missing_cov = result.get("missing_sentence_citations", []) or []

        if invalid:
            bad = " · ".join(f"({c[0]}, {c[1]})" for c in invalid[:8])
            st.warning(f"Found invalid citations not present in retrieved evidence: {bad}")

        if missing_cov:
            with st.expander(f"⚠️ {len(missing_cov)} sentence(s) missing end-of-sentence citations", expanded=False):
                for s in missing_cov[:10]:
                    st.write(f"- {s}")
                if len(missing_cov) > 10:
                    st.caption("Showing first 10 only.")

        citations = result.get("citations_used", [])
        if citations:
            st.markdown("### 📚 Citations")
            cit_text = " &nbsp;·&nbsp; ".join(f"({c[0]}, {c[1]})" for c in citations)
            st.markdown(f'<div class="citation">{cit_text}</div>', unsafe_allow_html=True)

            with st.expander("🔎 Cited evidence (resolved snippets)", expanded=False):
                seen_cited: set = set()
                for _ci, c in enumerate(citations):
                    sid, cid = c
                    if (sid, cid) in seen_cited:
                        continue  # skip duplicate citations — same chunk, same content
                    seen_cited.add((sid, cid))
                    meta = rag.get_source_metadata(sid)
                    title = meta.get("title", sid)
                    resolved = _resolve_cited_snippet(rag, c, result.get("retrieved_chunks", []))
                    st.markdown(f"**{title[:90]}**  `{sid}, {cid}`")
                    if resolved:
                        txt = (resolved.get("text") or "")[:800]
                        st.text_area(
                            label=f"{sid}, {cid}",
                            value=txt,
                            height=140,
                            disabled=True,
                            key=f"cited_{sid}_{cid}",
                        )
                    else:
                        st.caption("Chunk text not found in retrieved set or corpus index.")

        st.markdown("### 📄 Retrieved Sources")
        retrieved_chunks = result.get("retrieved_chunks", [])

        sources_dict: dict = {}
        for chunk in retrieved_chunks[:10]:
            sid = chunk["source_id"]
            sources_dict.setdefault(sid, []).append(chunk)

        for sid, chunks in list(sources_dict.items())[:5]:
            meta = rag.get_source_metadata(sid)
            title = meta.get("title", sid)
            with st.expander(f"📖 {title[:70]} ({len(chunks)} chunks)"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Authors**: {meta.get('authors', 'Unknown')}")
                c2.write(f"**Year**: {meta.get('year', '?')}")
                c3.write(f"**Type**: {meta.get('source_type', '?')}")

                url = _to_url(meta.get("url_or_doi", ""))
                if url:
                    st.markdown(f"[Open source link]({url})")

                raw_path = meta.get("raw_path", "")
                local = _resolve_repo_path(raw_path)
                if local and local.exists():
                    try:
                        data = local.read_bytes()
                        st.download_button(
                            "⬇️ Download source file",
                            data=data,
                            file_name=local.name,
                            mime="application/pdf",
                            use_container_width=False,
                            key=f"dl_{sid}",
                        )
                    except Exception:
                        pass

                for _ci, chunk in enumerate(chunks[:2]):
                    txt = chunk.get("text", "")
                    st.text_area(
                        f"({chunk['source_id']}, {chunk['chunk_id']})",
                        value=txt,
                        height=150,
                        disabled=True,
                        key=f"chunk_{chunk['source_id']}_{chunk['chunk_id']}_{_ci}",
                    )
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            thread_title = st.text_input(
                "Save as research thread (optional title):",
                value=(query_for_result or "")[:50],
                key="thread_title_input",
            )
        with col2:
            if st.button("💾 Save thread", use_container_width=True, key="save_thread_button"):
                tid = thread_mgr.save_thread(query_for_result, result, thread_title)
                st.success(f"Saved! Thread ID: {tid[:8]}…")
                st.session_state["last_thread_id"] = tid


# Gap Finder
def show_gap_finder_page(rag, artifact_gen, thread_mgr):
    """
    Stretch goal: Gap finder - missing evidence + targeted next retrieval actions.
    """
    st.markdown('<div class="main-header">🔭 Gap Finder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        "Discover what the corpus can't answer and get targeted follow-up query suggestions"
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "**How it works:** Enter a research question → the system retrieves evidence from the corpus "
        "→ an LLM analyses what aspects are *missing* and suggests follow-up queries you can run immediately."
    )

    query = st.text_area(
        "Research question to analyse for gaps:",
        height=90,
        placeholder="E.g., How do city digital twins handle model drift and recalibration?",
        key="gap_query_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        use_hybrid_gap = st.checkbox("Hybrid retrieval", value=True, key="gap_hybrid")
    with col2:
        use_rerank_gap = st.checkbox("LLM reranking", value=True, key="gap_rerank")

    if st.button("🔭 Find Research Gaps", type="primary", use_container_width=True, key="gap_btn"):
        if not query.strip():
            st.warning("Please enter a research question.")
        else:
            with st.spinner("Retrieving evidence and analysing gaps - this may take ~20 s…"):
                try:
                    result = rag.query(
                        query=query,
                        k=12,
                        use_hybrid=use_hybrid_gap,
                        use_reranking=use_rerank_gap,
                        top_k_after_rerank=8,
                        log=False,
                    )
                    gaps = artifact_gen.find_research_gaps(
                        query=query,
                        answer=result.get("answer", ""),
                        retrieved_chunks=result.get("retrieved_chunks", []),
                        gen_fn=rag._gen_text,
                    )
                    st.session_state["gap_result"]  = result
                    st.session_state["gap_gaps"]    = gaps
                    st.session_state["gap_query"]   = query
                except Exception as e:
                    st.error(f"Gap analysis failed: {e}")
                    st.exception(e)

    # -- display results --
    gaps   = st.session_state.get("gap_gaps")
    result = st.session_state.get("gap_result")
    shown_q = st.session_state.get("gap_query", "")

    if gaps is None:
        return

    st.markdown("---")

    # Answer context
    with st.expander("📝 What the corpus *does* cover (current answer)", expanded=False):
        st.markdown(result.get("answer", ""))
        cits = result.get("citations_used", [])
        if cits:
            cit_text = " · ".join(f"({c[0]}, {c[1]})" for c in cits)
            st.markdown(f'<div class="citation">{cit_text}</div>', unsafe_allow_html=True)

    if not gaps:
        st.success("✅ No significant gaps detected - the corpus covers this question well.")
        return

    st.markdown(f"### 🕳️ {len(gaps)} Research Gap(s) Identified")
    st.caption(f"For query: *{shown_q}*")

    for i, gap in enumerate(gaps, 1):
        with st.container():
            st.markdown(
                f'<div class="gap-card">'
                f'<strong>Gap {i}: {gap["gap_title"]}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**🔍 What's missing**")
                st.write(gap["what_is_missing"])
            with c2:
                st.markdown("**📄 Evidence that would help**")
                st.write(gap["evidence_would_help"])

            sug_q = gap.get("suggested_query", "")
            if sug_q:
                st.markdown(f"**💡 Suggested follow-up query:** `{sug_q}`")
                btn_cols = st.columns([1, 2])
                with btn_cols[0]:
                    if st.button(f"▶ Run this query", key=f"gap_run_{i}"):
                        st.session_state["last_query"]  = sug_q
                        st.session_state["last_result"] = None
                        # Immediately run it
                        with st.spinner(f"Running: {sug_q}…"):
                            try:
                                follow_result = rag.query(sug_q, k=10, use_hybrid=True,
                                                          use_reranking=True, top_k_after_rerank=5)
                                st.session_state["last_result"] = follow_result
                                st.session_state["last_query"]  = sug_q
                                st.success("Done! Go to 🔍 Search & Ask to see the answer.")
                            except Exception as e:
                                st.error(f"Query failed: {e}")
                with btn_cols[1]:
                    if st.button(f"💾 Save as thread", key=f"gap_save_{i}"):
                        if result:
                            tid = thread_mgr.save_thread(
                                sug_q, result,
                                f"[Gap follow-up] {gap['gap_title'][:60]}"
                            )
                            st.success(f"Saved thread {tid[:8]}…")
            st.markdown("---")

    # Export gaps as CSV
    gaps_df = pd.DataFrame(gaps)
    st.download_button(
        "📥 Download gaps as CSV",
        data=gaps_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name=f"research_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="gap_download_csv",
    )


# -- Research Threads ----------------------------------------------------------
def show_threads_page(thread_mgr, artifact_gen, export_mgr):
    st.markdown('<div class="main-header">📚 Research Threads</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">View and manage saved research threads</div>',
                unsafe_allow_html=True)

    search_query = st.text_input("Search threads:", placeholder="Enter keyword…")
    threads = thread_mgr.search_threads(search_query) if search_query else thread_mgr.list_threads(limit=50)

    if not threads:
        st.info("No threads saved yet.")
        return

    st.markdown(f"### Found {len(threads)} thread(s)")

    for tinfo in threads:
        tid    = tinfo["thread_id"]
        thread = thread_mgr.get_thread(tid)
        if not thread:
            continue

        with st.expander(f"📌 {tinfo['title']} - {tinfo['timestamp'][:10]}"):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**Query**: {thread['query']}")
                st.write(f"**Saved**: {thread['timestamp']}")
            with col2:
                if st.button("Delete", key=f"del_{tid}", type="secondary"):
                    thread_mgr.delete_thread(tid)
                    st.rerun()

            st.markdown("**Answer:**")
            st.markdown(thread["answer"])

            cits = thread.get("citations_used", [])
            if cits:
                st.markdown("**Citations:**")
                st.markdown(
                    f'<div class="citation">'
                    + " · ".join(f"({c[0]}, {c[1]})" for c in cits)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**Export:**")
            if st.button("Export Markdown", key=f"exp_md_{tid}", use_container_width=True):
                arts = artifact_gen.generate_all_artifacts(thread["query"], thread)
                fp   = export_mgr.export_markdown(
                    arts["synthesis_memo"],
                    f"thread_{tid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tinfo["title"],
                )
                st.success(f"Exported to: {fp}")


# -- Artifacts -----------------------------------------------------------------
def show_artifacts_page(rag, thread_mgr, artifact_gen, export_mgr):
    st.markdown('<div class="main-header">📊 Generate Artifacts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        "Evidence tables · Annotated bibliographies · Synthesis memos · Disagreement maps"
        "</div>",
        unsafe_allow_html=True,
    )

    use_last = st.checkbox("Use last query result", value=True)
    query = ""
    result = None
    source_label = ""

    if use_last and "last_result" in st.session_state:
        query        = st.session_state.get("last_query", "")
        result       = st.session_state.get("last_result", {})
        source_label = "last query result"
        st.info(f"Using query: *{query}*")
    else:
        threads = thread_mgr.list_threads(limit=50)
        if not threads:
            st.warning("No research threads found. Run a query first.")
            return
        opts     = {f"{t['title']} ({t['timestamp'][:10]})": t["thread_id"] for t in threads}
        sel_title = st.selectbox("Select thread:", list(opts.keys()))
        tid       = opts[sel_title]
        thread    = thread_mgr.get_thread(tid)
        if not thread:
            st.error("Thread not found.")
            return
        query        = thread["query"]
        result       = thread
        source_label = f"thread {tid[:8]}"

    artifact_type = st.radio(
        "Select artifact type:",
        ["Evidence Table", "Annotated Bibliography", "Synthesis Memo",
         "Disagreement Map", "All"],  # ← Disagreement Map added
        key="artifact_type_radio",
        horizontal=True,
    )

    # Clear cache if source changed
    cached = st.session_state.get("artifacts_payload")
    if cached and cached.get("query") != query:
        for k in ["artifacts_payload", "artifacts_base_name", "export_status"]:
            st.session_state.pop(k, None)

    if st.button("⚙️ Generate", type="primary", use_container_width=True, key="artifacts_btn_generate"):
        with st.spinner("Generating artifacts…"):
            try:
                arts = artifact_gen.generate_all_artifacts(
                    query,
                    result or {},
                    gen_fn=rag._gen_text,          # enables LLM-powered artifacts
                )
                st.session_state["artifacts_payload"] = {
                    "query":            query,
                    "artifact_type":    artifact_type,
                    "generated_at":     datetime.now().isoformat(timespec="seconds"),
                    "source":           source_label,
                    "evidence_table":   arts["evidence_table"],
                    "bibliography":     arts["bibliography"],
                    "synthesis_memo":   arts["synthesis_memo"],
                    "disagreement_map": arts["disagreement_map"],
                }
                st.session_state["artifacts_base_name"] = \
                    f"artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state["export_status"] = {}
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate artifacts: {e}")
                st.exception(e)

    payload = st.session_state.get("artifacts_payload")
    if not payload:
        return

    st.markdown("---")
    st.caption(f"Generated at {payload.get('generated_at')} from {payload.get('source')}.")

    base_name     = st.session_state.get("artifacts_base_name",
                                         f"artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    export_status = st.session_state.get("export_status", {})

    def _set_status(key: str, filepath: str):
        export_status[key] = filepath
        st.session_state["export_status"] = export_status

    def _dl_from_path(label, filepath, mime, file_name, key):
        try:
            p = Path(filepath)
            if p.exists():
                st.download_button(label, data=p.read_bytes(), file_name=file_name,
                                   mime=mime, key=key, use_container_width=True)
            else:
                st.caption(f"File missing: {filepath}")
        except Exception as e:
            st.caption(f"Download unavailable: {e}")

    # -- Evidence Table --
    if payload["artifact_type"] in ("Evidence Table", "All"):
        st.markdown("### 📋 Evidence Table")
        ev_df = payload["evidence_table"]
        st.dataframe(ev_df, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Export CSV", key="ev_export_csv", use_container_width=True):
                fp = export_mgr.export_csv(ev_df, f"{base_name}_evidence_table")
                _set_status("ev_csv", fp); st.rerun()
        with c2:
            if st.button("Export PDF", key="ev_export_pdf", use_container_width=True):
                fp = export_mgr.export_evidence_table_pdf(ev_df, f"{base_name}_evidence_table", payload["query"])
                _set_status("ev_pdf", fp); st.rerun()
        with c3:
            st.download_button("Download CSV",
                data=ev_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"{base_name}_evidence_table.csv", mime="text/csv",
                key="ev_dl_csv", use_container_width=True)
        with c4:
            if export_status.get("ev_pdf"):
                _dl_from_path("Download PDF", export_status["ev_pdf"],
                              "application/pdf", f"{base_name}_evidence_table.pdf", "ev_dl_pdf")
            else:
                st.caption("Generate PDF to download")
        for sk, sl in [("ev_csv", "CSV"), ("ev_pdf", "PDF")]:
            if export_status.get(sk):
                st.success(f"Saved {sl}: {export_status[sk]}")

    # -- Annotated Bibliography --
    if payload["artifact_type"] in ("Annotated Bibliography", "All"):
        st.markdown("### 📚 Annotated Bibliography")
        bib      = payload["bibliography"]
        bib_df   = pd.DataFrame(bib)
        st.dataframe(bib_df, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Export CSV", key="bib_export_csv", use_container_width=True):
                fp = export_mgr.export_bibliography_csv(bib, f"{base_name}_bibliography")
                _set_status("bib_csv", fp); st.rerun()
        with c2:
            # -- BibTeX export (stretch goal #3) --
            if st.button("Export BibTeX", key="bib_export_bib", use_container_width=True):
                fp = export_mgr.export_bibtex(bib, f"{base_name}_bibliography")
                _set_status("bib_bib", fp); st.rerun()
        with c3:
            st.download_button("Download CSV",
                data=bib_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"{base_name}_bibliography.csv", mime="text/csv",
                key="bib_dl_csv", use_container_width=True)
        with c4:
            if export_status.get("bib_bib"):
                _dl_from_path("Download .bib", export_status["bib_bib"],
                              "text/plain", f"{base_name}_bibliography.bib", "bib_dl_bib")
            else:
                st.caption("Export BibTeX to download")
        for sk, sl in [("bib_csv", "CSV"), ("bib_bib", "BibTeX")]:
            if export_status.get(sk):
                st.success(f"Saved {sl}: {export_status[sk]}")

    # -- Synthesis Memo --
    if payload["artifact_type"] in ("Synthesis Memo", "All"):
        st.markdown("### 📝 Synthesis Memo")
        memo = payload["synthesis_memo"]
        st.markdown(memo)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Export Markdown", key="memo_export_md", use_container_width=True):
                fp = export_mgr.export_markdown(memo, f"{base_name}_synthesis_memo",
                                                f"Synthesis Memo: {payload['query']}")
                _set_status("memo_md", fp); st.rerun()
        with c2:
            if st.button("Export PDF", key="memo_export_pdf", use_container_width=True):
                fp = export_mgr.export_pdf(memo, f"{base_name}_synthesis_memo",
                                           f"Synthesis Memo: {payload['query']}")
                _set_status("memo_pdf", fp); st.rerun()
        with c3:
            st.download_button("Download Markdown",
                data=memo.encode("utf-8"), file_name=f"{base_name}_synthesis_memo.md",
                mime="text/markdown", key="memo_dl_md", use_container_width=True)
        with c4:
            if export_status.get("memo_pdf"):
                _dl_from_path("Download PDF", export_status["memo_pdf"],
                              "application/pdf", f"{base_name}_synthesis_memo.pdf", "memo_dl_pdf")
            else:
                st.caption("Generate PDF to download")
        for sk, sl in [("memo_md", "Markdown"), ("memo_pdf", "PDF")]:
            if export_status.get(sk):
                st.success(f"Saved {sl}: {export_status[sk]}")

    # -- Disagreement Map (stretch goal #2) --
    if payload["artifact_type"] in ("Disagreement Map", "All"):
        st.markdown("### ⚡ Disagreement Map")
        st.caption(
            "Surfaces contradictions and tensions between sources retrieved for this query. "
            "Powered by LLM analysis."
        )
        dis_df = payload["disagreement_map"]

        if dis_df.empty:
            st.success("✅ No significant disagreements found among retrieved sources.")
        else:
            # Colour-code by conflict type
            st.dataframe(dis_df, use_container_width=True, height=280)
            st.download_button(
                "📥 Download Disagreement Map (CSV)",
                data=dis_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"{base_name}_disagreement_map.csv",
                mime="text/csv",
                key="dis_dl_csv",
                use_container_width=True,
            )
            st.markdown("---")
            st.markdown("**Detailed view:**")
            for _, row in dis_df.iterrows():
                with st.expander(f"⚡ {row['Aspect']}  -  *{row['Conflict Type']}*"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**{row['Source A']}**")
                        st.info(row["Position A"])
                    with col_b:
                        st.markdown(f"**{row['Source B']}**")
                        st.warning(row["Position B"])


# -- Evaluation ----------------------------------------------------------------
def show_evaluation_page(rag):
    st.markdown('<div class="main-header">📈 Evaluation View</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Run the evaluation query set and inspect metrics</div>',
                unsafe_allow_html=True)

    if "evaluator" not in st.session_state:
        st.session_state["evaluator"] = RAGEvaluator(rag)
    evaluator = st.session_state["evaluator"]

    total_queries = len(getattr(evaluator, "queries", []) or [])
    if total_queries == 0:
        st.warning("No evaluation queries found.")
        return

    def _make_display_df(df: pd.DataFrame) -> pd.DataFrame:
        preferred = [
            "query_id", "query_type", "use_hybrid", "use_reranking",
            "groundedness_score", "citation_precision_value",
            "answer_relevance_score", "answer_has_citations",
        ]
        cols = [c for c in preferred if c in df.columns]
        out  = (df[cols].copy() if cols else df.copy()).reset_index(drop=True)
        if "citation_precision_value" in out.columns:
            out["citation_precision_value"] = out["citation_precision_value"].apply(
                lambda v: f"{float(v):.2%}" if str(v).replace(".", "").isdigit() else str(v)
            )
        return out

    with st.expander("📋 Preset evaluation queries", expanded=False):
        qdf  = pd.DataFrame(evaluator.queries)
        cols = [c for c in ["id", "type", "difficulty", "query"] if c in qdf.columns]
        st.dataframe(qdf[cols] if cols else qdf, use_container_width=True, height=260)
        st.caption(f"Total: {total_queries} queries")

    with st.form("eval_controls_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            run_type = st.radio(
                "Evaluation type:",
                ["Enhanced (hybrid + reranking)", "Baseline (vector only)",
                 "Compare baseline vs enhanced"],
                key="eval_run_type",
            )
        with col2:
            default_limit = max(0, min(int(st.session_state.get("eval_limit", 1) or 1), total_queries))
            query_limit   = st.number_input(
                "Query limit (0 = all):",
                min_value=0, max_value=total_queries,
                value=default_limit, key="eval_limit",
            )
        run_clicked = st.form_submit_button("🚀 Run evaluation", type="primary",
                                            use_container_width=True)

    if run_clicked:
        for k in ["eval_results_df", "eval_display_df", "eval_report_path"]:
            st.session_state[k] = None
        st.session_state["eval_report_path"] = ""

        llm_before = getattr(rag, "llm_call_count", 0)
        t0 = time.time()

        query_ids = None
        if int(query_limit) > 0:
            n = min(int(query_limit), total_queries)
            query_ids = [f"Q{i:02d}" for i in range(1, n + 1)]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with st.spinner("Running evaluation…"):
            try:
                if run_type == "Enhanced (hybrid + reranking)":
                    results_df = evaluator.run_evaluation(
                        query_ids=query_ids, run_label="enhanced",
                        rag_query_kwargs={"use_hybrid": True, "use_reranking": True,
                                          "top_k_after_rerank": 5, "log": False},
                    )
                    report_path = evaluator.generate_eval_report(
                        results_df,
                        output_path=Path("outputs/eval") / f"eval_report_enhanced_{timestamp}.md",
                    )
                    st.session_state["eval_mode"]        = "enhanced"
                    st.session_state["eval_baseline_df"] = None
                    st.session_state["eval_enhanced_df"] = None

                elif run_type == "Baseline (vector only)":
                    results_df = evaluator.run_evaluation(
                        query_ids=query_ids, run_label="baseline",
                        rag_query_kwargs={"use_hybrid": False, "use_reranking": False, "log": False},
                    )
                    report_path = evaluator.generate_eval_report(
                        results_df,
                        output_path=Path("outputs/eval") / f"eval_report_baseline_{timestamp}.md",
                    )
                    st.session_state["eval_mode"]        = "baseline"
                    st.session_state["eval_baseline_df"] = None
                    st.session_state["eval_enhanced_df"] = None

                else:
                    dfs = evaluator.run_enhancement_comparison(
                        query_ids=query_ids,
                    )
                    baseline_df = dfs.get("baseline")
                    enhanced_df = dfs.get("enhanced")
                    results_df  = enhanced_df
                    report_path = evaluator.generate_eval_report(
                        results_df=enhanced_df, baseline_df=baseline_df,
                        enhanced_df=enhanced_df,
                        output_path=Path("outputs/eval") / f"eval_report_compare_{timestamp}.md",
                    )
                    st.session_state["eval_mode"]        = "compare"
                    st.session_state["eval_baseline_df"] = baseline_df
                    st.session_state["eval_enhanced_df"] = enhanced_df

                st.session_state["eval_results_df"]  = results_df
                st.session_state["eval_display_df"]  = (
                    _make_display_df(results_df)
                    if isinstance(results_df, pd.DataFrame) else None
                )
                st.session_state["eval_report_path"] = str(report_path) if report_path else ""

                elapsed   = time.time() - t0
                llm_after = getattr(rag, "llm_call_count", 0)
                st.success(
                    f"Evaluation complete - {llm_after - llm_before} LLM calls · {elapsed:.1f} s"
                )

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)
                return

    results_df = st.session_state.get("eval_results_df")
    display_df = st.session_state.get("eval_display_df")
    mode       = st.session_state.get("eval_mode", "")

    if isinstance(results_df, pd.DataFrame) and not results_df.empty and isinstance(display_df, pd.DataFrame):
        st.markdown("### 📊 Evaluation Summary")

        def _m(col):
            try:    return float(results_df[col].mean())
            except: return None

        g, cp, ar = _m("groundedness_score"), _m("citation_precision_value"), _m("answer_relevance_score")

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Groundedness",      f"{g:.2f}/4"  if g  is not None else "n/a")
        c2.metric("Avg Citation Precision", f"{cp:.2%}"  if cp is not None else "n/a")
        c3.metric("Avg Answer Relevance",  f"{ar:.2f}/4" if ar is not None else "n/a")

        if mode == "compare":
            b_df = st.session_state.get("eval_baseline_df")
            e_df = st.session_state.get("eval_enhanced_df")
            if isinstance(b_df, pd.DataFrame) and isinstance(e_df, pd.DataFrame):
                def _row(df, label):
                    def _mm(col):
                        try:    return float(df[col].mean())
                        except: return None
                    return {"run": label, "avg_groundedness": _mm("groundedness_score"),
                            "avg_citation_precision": _mm("citation_precision_value"),
                            "avg_answer_relevance": _mm("answer_relevance_score")}
                comp = pd.DataFrame([_row(b_df, "baseline"), _row(e_df, "enhanced (hybrid RRF + rerank)")])
                st.markdown("### 🔎 Baseline vs Enhanced")
                st.dataframe(comp, use_container_width=True, height=140)

        if mode == "compare":
            st.info(
                "ℹ️ **Compare mode**: the Summary metrics and Detailed results below show the "
                "**Enhanced** run only. The Baseline vs Enhanced table above shows averages for both runs. "
                "Switch to 'Baseline (vector only)' mode to inspect baseline results in full detail."
            )

        st.markdown(
            "### 📋 Detailed results"
            + (" - Enhanced run" if mode == "compare" else "")
        )
        st.dataframe(display_df, use_container_width=True, height=400)

        # In compare mode, also offer the baseline detail table
        if mode == "compare":
            b_df = st.session_state.get("eval_baseline_df")
            if isinstance(b_df, pd.DataFrame) and not b_df.empty:
                with st.expander("📋 Detailed results - Baseline run", expanded=False):
                    st.dataframe(_make_display_df(b_df), use_container_width=True, height=400)

        rpt = st.session_state.get("eval_report_path", "")
        if rpt:
            st.success(f"Report saved to: {rpt}")
            try:
                content = Path(rpt).read_text(encoding="utf-8")
                st.download_button("📥 Download report (.md)", data=content.encode("utf-8"),
                                   file_name=Path(rpt).name, mime="text/markdown",
                                   use_container_width=True)
            except Exception:
                pass

    show_rpts = st.checkbox("Show recent evaluation reports", value=False, key="eval_show_reports")
    if show_rpts:
        eval_dir = Path("outputs/eval")
        if eval_dir.exists():
            reports = sorted(eval_dir.glob("eval_report_*.md"),
                             key=lambda x: x.stat().st_mtime, reverse=True)
            if reports:
                selected = st.selectbox("Select report:", [p.name for p in reports[:20]],
                                        key="eval_report_select")
                rpt_path = eval_dir / selected
                try:
                    content = rpt_path.read_text(encoding="utf-8")
                    st.download_button("Download", data=content.encode("utf-8"),
                                       file_name=selected, mime="text/markdown",
                                       use_container_width=True)
                    preview = content[:4000] + "\n\n…(truncated)" if len(content) > 4000 else content
                    st.markdown(preview)
                except Exception as e:
                    st.warning(f"Cannot read report: {e}")
            else:
                st.info("No evaluation reports found.")
        else:
            st.info("outputs/eval does not exist yet.")


# -- About ---------------------------------------------------------------------
def show_about_page():
    st.markdown('<div class="main-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
## Personal Research Portal (PRP) - Phase 3

A research-grade portal that answers questions with evidence, citations, and exportable artifacts.

### Core Features (MVP)
| Feature | Description |
|---------|-------------|
| 🔍 **Search & Ask** | Hybrid RRF + LLM reranking, citation-backed answers |
| 📚 **Research Threads** | Save / search / delete query sessions |
| 📊 **Artifacts** | Evidence table · Annotated bibliography · Synthesis memo |
| 📥 **Export** | Markdown · CSV · PDF · BibTeX |
| 📈 **Evaluation** | 22-query set · Groundedness · Citation precision · Answer relevance |

### Stretch Goals Implemented
| Stretch Goal | Location |
|-------------|----------|
| 🔭 **Gap Finder** | Navigation → Gap Finder |
| ⚡ **Disagreement Map** | Artifacts → Disagreement Map |
| 📄 **BibTeX Export** | Artifacts → Annotated Bibliography |
| 🔧 **Metadata Filters** | Search & Ask → Filter by source metadata |

### Tech Stack
- **Backend**: Python 3.11, Streamlit
- **RAG**: Hybrid retrieval (BM25 + FAISS) via **RRF**, LLM reranking
- **Generator / Evaluator**: Google Gemini
- **Vector Store**: FAISS · `all-MiniLM-L6-v2` embeddings

### Citation Format
All answers use `(source_id, chunk_id)`. Every citation resolves to a source in `data/data_manifest.csv`.

### Domain
City Digital Twins - 22 sources (2021-2025), including peer-reviewed journals,
arXiv preprints, and technical reports.
""")


if __name__ == "__main__":
    main()
