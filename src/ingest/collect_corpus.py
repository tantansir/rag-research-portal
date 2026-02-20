"""
Corpus Collection Script for City Digital Twins Research
Automates the collection of papers from arXiv and manual source list

Source ID format: AuthorYear[Suffix]
  e.g. Talasila2023, Luo2024, Barbie2023a / Barbie2023b
"""

import arxiv
import os
import re
import json
import time
from pathlib import Path
import requests
import pandas as pd
from typing import List, Dict, Optional


def make_source_id(authors, year: int, existing_ids: set = None) -> str:
    """
    Generate a short, human-readable source_id: LastName + Year [+ suffix].

    Rules
    -----
    - Take the last name of the first author.
    - Strip non-ASCII characters and spaces (handles hyphenated names like Do-Bui-Khanh → DoBui).
    - Append the 4-digit year.
    - If the id already exists in `existing_ids`, append 'a', 'b', … to disambiguate.

    Examples
    --------
    Prasad Talasila; Claudio Gomes, 2023  →  Talasila2023
    Junjie Luo,                   2024  →  Luo2024
    Alexander Barbie (first of two 2023 papers)  →  Barbie2023a / Barbie2023b
    """
    existing_ids = existing_ids or set()

    # Parse first author's last name
    if isinstance(authors, list):
        first_author = authors[0] if authors else "Unknown"
    else:
        first_author = str(authors).split(";")[0].strip()

    # Extract last name (last whitespace-delimited token)
    parts = first_author.strip().split()
    last_name = parts[-1] if parts else "Unknown"

    # Keep only ASCII letters; collapse hyphens
    last_name = re.sub(r"[^A-Za-z\-]", "", last_name)
    last_name = last_name.replace("-", "")   # DoBuiKhanh → keep full or shorten
    if len(last_name) > 12:                  # cap length for readability
        last_name = last_name[:12]

    base = f"{last_name}{year}"

    if base not in existing_ids:
        return base

    # Disambiguate with suffix a, b, c …
    for ch in "abcdefghij":
        candidate = f"{base}{ch}"
        if candidate not in existing_ids:
            return candidate

    # Fallback (very unlikely)
    return f"{base}_{int(time.time()) % 10000}"


class CorpusCollector:
    """Collects and organises research papers for the digital twins corpus."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = []

    def search_arxiv(self, query: str, max_results: int = 15) -> List[Dict]:
        """Search arXiv for relevant papers."""
        print(f"Searching arXiv for: {query}")

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        # Collect existing IDs so we can disambiguate within this batch
        existing = {p["source_id"] for p in self.manifest}

        papers = []
        for result in client.results(search):
            authors_list = [author.name for author in result.authors]
            year = result.published.year
            sid = make_source_id(authors_list, year, existing_ids=existing)
            existing.add(sid)

            paper_info = {
                "source_id":      sid,
                "title":          result.title,
                "authors":        authors_list,
                "year":           year,
                "source_type":    "arxiv_preprint",
                "venue":          "arXiv",
                "url_or_doi":     result.entry_id,
                "pdf_url":        result.pdf_url,
                "abstract":       result.summary,
                "tags":           ["digital_twin", "urban_ai"],
                "relevance_note": "ArXiv paper on city digital twins and urban AI",
            }
            papers.append(paper_info)

        print(f"Found {len(papers)} papers")
        return papers

    def download_arxiv_papers(self, papers: List[Dict]) -> None:
        """Download PDFs from arXiv."""
        for paper in papers:
            try:
                filename = f"{paper['source_id']}.pdf"
                filepath = self.data_dir / filename

                if filepath.exists():
                    print(f"Already downloaded: {filename}")
                    paper["raw_path"] = filepath.as_posix()
                    continue

                print(f"Downloading: {paper['title'][:60]}…")
                response = requests.get(paper["pdf_url"], timeout=30)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)

                paper["raw_path"] = filepath.as_posix()
                print(f"  → {filepath}")
                time.sleep(2)   # be polite to arXiv servers

            except Exception as e:
                print(f"Error downloading {paper['source_id']}: {e}")
                paper["raw_path"] = None

    # ── manifest helpers ──────────────────────────────────────────────────────

    def _read_csv_robust(self, path: str) -> pd.DataFrame:
        for enc in ["utf-8-sig", "utf-8", "gbk", "cp1252"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read {path} with any known encoding")

    def upsert_sources_into_manifest(
        self,
        sources: List[Dict],
        manifest_path: str = "data/data_manifest.csv",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        required_cols = [
            "source_id", "title", "authors", "year", "source_type", "venue",
            "url_or_doi", "pdf_url", "abstract", "tags", "relevance_note",
            "raw_path", "processed_path",
        ]

        base_df = (
            self._read_csv_robust(manifest_path)
            if os.path.exists(manifest_path)
            else pd.DataFrame(columns=required_cols)
        )
        for col in required_cols:
            if col not in base_df.columns:
                base_df[col] = ""

        src_df = pd.DataFrame(sources) if sources else pd.DataFrame(columns=required_cols)
        for col in required_cols:
            if col not in src_df.columns:
                src_df[col] = ""

        if not src_df.empty:
            src_df["authors"] = src_df["authors"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else ("" if pd.isna(x) else str(x))
            )
            src_df["tags"] = src_df["tags"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else ("" if pd.isna(x) else str(x))
            )

        def _is_empty(v) -> bool:
            if v is None:
                return True
            if isinstance(v, float) and pd.isna(v):
                return True
            return str(v).strip() in ("", "nan")

        base = base_df.set_index("source_id", drop=False)
        for _, row in src_df.iterrows():
            sid = str(row.get("source_id", "")).strip()
            if not sid:
                continue
            if sid not in base.index:
                new_row = {c: row.get(c, "") for c in required_cols}
                base = pd.concat(
                    [base, pd.DataFrame([new_row]).set_index("source_id", drop=False)], axis=0
                )
                continue
            for c in required_cols:
                if c == "source_id":
                    continue
                incoming = row.get(c, "")
                current  = base.at[sid, c]
                if _is_empty(incoming):
                    continue
                if overwrite or _is_empty(current):
                    base.at[sid, c] = incoming

        merged_df = base.reset_index(drop=True)
        for col in ["raw_path", "processed_path"]:
            if col in merged_df.columns:
                merged_df[col] = (
                    merged_df[col].fillna("").astype(str).str.replace("\\", "/", regex=False)
                )
        merged_df = merged_df[required_cols]
        merged_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        return merged_df

    def merge_manual_journal_list_into_manifest(
        self,
        manual_csv_path: str = "data/data_manifest_example_journal paper_list.csv",
        manifest_path:   str = "data/data_manifest.csv",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        if not os.path.exists(manual_csv_path):
            print(f"Manual journal list not found: {manual_csv_path}")
            return (
                self._read_csv_robust(manifest_path)
                if os.path.exists(manifest_path)
                else pd.DataFrame()
            )

        manual_df = self._read_csv_robust(manual_csv_path)

        if "raw_path" in manual_df.columns:
            def _fix_raw(p):
                if pd.isna(p):
                    return ""
                s = str(p).strip().replace("\\", "/")
                if not s:
                    return ""
                if "/" not in s and s.lower().endswith(".pdf"):
                    return f"data/raw/{s}"
                return s
            manual_df["raw_path"] = manual_df["raw_path"].apply(_fix_raw)

        # If the manual CSV has long legacy IDs, regenerate them
        if "source_id" in manual_df.columns:
            existing = set()
            new_ids = []
            for _, row in manual_df.iterrows():
                sid = str(row.get("source_id", "")).strip()
                # If the id looks like a long slug (contains underscores + year at end), regenerate
                if re.search(r"_\d{4}$", sid) or len(sid) > 20:
                    authors = str(row.get("authors", "Unknown"))
                    year    = int(str(row.get("year", "2000"))[:4])
                    sid = make_source_id(authors, year, existing_ids=existing)
                existing.add(sid)
                new_ids.append(sid)
            manual_df["source_id"] = new_ids

        sources = manual_df.to_dict(orient="records")
        merged  = self.upsert_sources_into_manifest(
            sources=sources,
            manifest_path=manifest_path,
            overwrite=overwrite,
        )

        # Warn about missing PDFs
        if "raw_path" in merged.columns:
            missing = [
                (r["source_id"], r["raw_path"])
                for _, r in merged.iterrows()
                if str(r.get("raw_path", "")).strip().lower().endswith(".pdf")
                and not os.path.exists(str(r.get("raw_path", "")))
            ]
            if missing:
                print("Warning: some raw_path PDFs are missing:")
                for sid, rp in missing[:10]:
                    print(f"  {sid}: {rp}")

        return merged


def main():
    """Main corpus collection workflow."""
    collector = CorpusCollector()

    queries = [
        "urban digital twin decision support",
        "digital twin participatory planning",
        "urban AI predictive analytics",
    ]

    all_papers: List[Dict] = []
    for query in queries[:3]:
        papers = collector.search_arxiv(query, max_results=5)
        all_papers.extend(papers)
        time.sleep(3)

    # Deduplicate by source_id
    unique_papers = list({p["source_id"]: p for p in all_papers}.values())

    collector.download_arxiv_papers(unique_papers)

    collector.upsert_sources_into_manifest(
        unique_papers,
        manifest_path="data/data_manifest.csv",
        overwrite=False,
    )

    manifest_df = collector.merge_manual_journal_list_into_manifest(
        manual_csv_path="data/data_manifest_example_journal paper_list.csv",
        manifest_path="data/data_manifest.csv",
        overwrite=False,
    )

    print("\n" + "=" * 60)
    print("CORPUS COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total sources collected: {len(manifest_df)}")
    print(f"ArXiv papers downloaded: {sum(1 for p in unique_papers if p.get('raw_path'))}")
    print("\nSample source IDs:")
    for sid in manifest_df["source_id"].head(6):
        print(f"  {sid}")


if __name__ == "__main__":
    main()
