"""
Export Manager

Phase 3 requires exportable artifacts (Markdown/CSV/PDF).
This module exports:
- Markdown (.md)
- CSV (.csv)
- PDF (.pdf) for memos and for evidence tables with cell wrapping
- BibTeX (.bib) for bibliography (stretch goal)
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER


class ExportManager:
    """Manages export of research artifacts to various formats."""

    def __init__(self, output_dir: str = "outputs/exports"):
        root = Path(__file__).resolve().parents[2]
        out = Path(output_dir)
        self.output_dir = out if out.is_absolute() else (root / out)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- Markdown --------------------------------------------------------------
    def export_markdown(self, content: str, filename: str, title: str = "Research Export") -> str:
        filepath = self.output_dir / f"{filename}.md"
        header = f"# {title}\n\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + (content or ""))
        return str(filepath)

    # -- CSV -------------------------------------------------------------------
    def export_csv(self, df: pd.DataFrame, filename: str) -> str:
        filepath = self.output_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        return str(filepath)

    # -- PDF (text) ------------------------------------------------------------
    def export_pdf(self, content: str, filename: str, title: str = "Research Export") -> str:
        """
        Export Markdown-like content to a simple PDF.
        """
        filepath = self.output_dir / f"{filename}.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#1f4788"),
            spaceAfter=30,
            alignment=TA_CENTER,
        )

        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.15 * inch))

        timestamp_style = ParagraphStyle(
            "Timestamp",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER,
        )
        story.append(
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", timestamp_style)
        )
        story.append(Spacer(1, 0.25 * inch))

        lines = (content or "").split("\n")
        for line in lines:
            ln = (line or "").strip()
            if not ln:
                story.append(Spacer(1, 0.1 * inch))
                continue

            if ln.startswith("# "):
                story.append(Paragraph(ln[2:], styles["Heading1"]))
                story.append(Spacer(1, 0.08 * inch))
                continue
            if ln.startswith("## "):
                story.append(Paragraph(ln[3:], styles["Heading2"]))
                story.append(Spacer(1, 0.06 * inch))
                continue
            if ln.startswith("### "):
                story.append(Paragraph(ln[4:], styles["Heading3"]))
                story.append(Spacer(1, 0.04 * inch))
                continue

            # Bullets
            if ln.startswith("- ") or ln.startswith("* ") or ln.startswith("• "):
                story.append(Paragraph("• " + ln[2:].replace("**", ""), styles["Normal"]))
                continue

            para_text = ln.replace("**", "").replace("`", "")
            story.append(Paragraph(para_text, styles["Normal"]))
            story.append(Spacer(1, 0.06 * inch))

        doc.build(story)
        return str(filepath)

    # -- PDF (evidence table) --------------------------------------------------
    def export_evidence_table_pdf(self, df: pd.DataFrame, filename: str, query: str) -> str:
        """
        Export evidence table DataFrame to PDF with wrapping and sane column widths.
        """
        filepath = self.output_dir / f"{filename}.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#1f4788"),
            spaceAfter=16,
        )
        story.append(Paragraph("Evidence Table", title_style))
        story.append(Paragraph(f"<b>Query:</b> {query}", styles["Normal"]))
        story.append(Spacer(1, 0.18 * inch))

        # Keep the table compact
        preferred_cols = ["Claim", "Evidence Snippet", "Citation", "Confidence", "Notes"]
        cols = [c for c in preferred_cols if c in df.columns]
        if not cols:
            cols = df.columns.tolist()[:5]

        sub = df[cols].copy()

        cell_style = ParagraphStyle(
            "Cell",
            parent=styles["Normal"],
            fontSize=8,
            leading=9,
        )
        header_style = ParagraphStyle(
            "HeaderCell",
            parent=styles["Normal"],
            fontSize=9,
            leading=10,
            textColor=colors.whitesmoke,
        )

        # Build table data with Paragraph cells for wrapping
        table_data = [[Paragraph(f"<b>{c}</b>", header_style) for c in cols]]
        for _, row in sub.iterrows():
            table_data.append([Paragraph(str(v) if v is not None else "", cell_style) for v in row.values])

        # Column widths based on available doc width
        w = doc.width
        col_width_map = {
            "Claim": 0.22,
            "Evidence Snippet": 0.45,
            "Citation": 0.13,
            "Confidence": 0.08,
            "Notes": 0.12,
        }
        col_widths = []
        for c in cols:
            frac = col_width_map.get(c, 1.0 / max(1, len(cols)))
            col_widths.append(w * frac)

        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 0.15 * inch))
        story.append(
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])
        )

        doc.build(story)
        return str(filepath)

    # -- Bibliography exports --------------------------------------------------
    def export_bibliography_csv(self, bibliography: List[Dict], filename: str) -> str:
        df = pd.DataFrame(bibliography)
        return self.export_csv(df, filename)

    def export_bibtex(self, bibliography: List[Dict], filename: str) -> str:
        """
        Export bibliography to BibTeX .bib format.
        """
        import re as _re

        def _make_key(entry: Dict) -> str:
            authors = str(entry.get("Authors", "Unknown"))
            year = str(entry.get("Year", "0000"))
            first_author = authors.split(";")[0].split(",")[0].strip()
            last_name = first_author.split()[-1] if first_author.split() else "Unknown"
            last_name = _re.sub(r"[^a-zA-Z]", "", last_name)
            title_words = str(entry.get("Title", "")).split()[:2]
            title_part = "".join(_re.sub(r"[^a-zA-Z]", "", w) for w in title_words)
            return (f"{last_name}{year}{title_part}" or "unknown").lower()

        entries = []
        seen_keys: Dict[str, int] = {}

        for entry in bibliography:
            key = _make_key(entry)
            if key in seen_keys:
                seen_keys[key] += 1
                suffix = chr(97 + seen_keys[key])
                key = f"{key}{suffix}"
            else:
                seen_keys[key] = 0

            source_id = str(entry.get("Source ID", ""))
            venue = str(entry.get("Venue", ""))
            entry_type = "article"
            if "arxiv" in source_id.lower():
                entry_type = "misc"

            doi_url = str(entry.get("DOI/URL", ""))
            is_doi = doi_url.startswith("10.")
            is_url = doi_url.startswith("http")

            lines = [f"@{entry_type}{{{key},"]
            lines.append(f"  title   = {{{entry.get('Title', '')}}},")
            lines.append(f"  author  = {{{entry.get('Authors', '')}}},")
            lines.append(f"  year    = {{{entry.get('Year', '')}}},")
            if venue and venue.lower() not in ("unknown", "nan", ""):
                field = "journal" if entry_type == "article" else "howpublished"
                lines.append(f"  {field} = {{{venue}}},")
            if is_doi:
                lines.append(f"  doi     = {{{doi_url}}},")
            elif is_url:
                lines.append(f"  url     = {{{doi_url}}},")
            lines.append("}")
            entries.append("\n".join(lines))

        content = "\n\n".join(entries)
        filepath = self.output_dir / f"{filename}.bib"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return str(filepath)
