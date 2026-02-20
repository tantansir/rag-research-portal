"""
Ingestion Pipeline for Personal Research Portal
Processes PDFs, extracts text, chunks with section awareness, and prepares for RAG
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import pdfplumber
from tqdm import tqdm


class DocumentProcessor:
    """Processes documents and creates chunks with metadata"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=2, y_tolerance=3, use_text_flow=True)
                    if not text:
                        text = page.extract_text()
                    if text:
                        text_parts.append(text)

            full_text = "\n\n".join(text_parts)
            return self.clean_text(full_text)

        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean extracted text - fix ligatures and formatting"""
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        text = text.replace("ﬀ", "ff")
        text = text.replace("ﬃ", "ffi")
        text = text.replace("ﬄ", "ffl")
        text = text.replace("\u00a0", " ")

        # Dehyphenate line breaks: "informa-\\ntion" -> "information"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Convert single newlines into spaces, keep paragraph breaks
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Normalize excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Light spacing fixes for common PDF artifacts
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Za-z])\.(?=[A-Za-z])", r"\1. ", text)
        text = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", text)
        text = re.sub(r"([A-Za-z])(\d{2,})", r"\1 \2", text)
        text = re.sub(r"(\d{2,})([A-Za-z])", r"\1 \2", text)

        return text.strip()

    def detect_sections(self, text: str) -> List[Dict]:
        """
        Detect section boundaries in academic papers
        Returns list of sections with start positions
        """
        sections = []
        
        # Common section headers in academic papers
        section_patterns = [
            r'\n\s*(\d+\.?\s+)?Abstract\s*\n',
            r'\n\s*(\d+\.?\s+)?Introduction\s*\n',
            r'\n\s*(\d+\.?\s+)?Related Work\s*\n',
            r'\n\s*(\d+\.?\s+)?Background\s*\n',
            r'\n\s*(\d+\.?\s+)?Methodology\s*\n',
            r'\n\s*(\d+\.?\s+)?Methods\s*\n',
            r'\n\s*(\d+\.?\s+)?Approach\s*\n',
            r'\n\s*(\d+\.?\s+)?System\s*\n',
            r'\n\s*(\d+\.?\s+)?Architecture\s*\n',
            r'\n\s*(\d+\.?\s+)?Implementation\s*\n',
            r'\n\s*(\d+\.?\s+)?Experiments?\s*\n',
            r'\n\s*(\d+\.?\s+)?Evaluation\s*\n',
            r'\n\s*(\d+\.?\s+)?Results\s*\n',
            r'\n\s*(\d+\.?\s+)?Discussion\s*\n',
            r'\n\s*(\d+\.?\s+)?Conclusion\s*\n',
            r'\n\s*(\d+\.?\s+)?References\s*\n',
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_name = match.group(0).strip()
                sections.append({
                    'name': section_name,
                    'start': match.start(),
                    'end': None
                })
        
        # Sort by position and set end positions
        sections.sort(key=lambda x: x['start'])
        for i in range(len(sections) - 1):
            sections[i]['end'] = sections[i + 1]['start']
        
        if sections:
            sections[-1]['end'] = len(text)
        
        return sections
    
    def chunk_with_section_awareness(
        self, 
        text: str, 
        source_id: str,
        chunk_size: int = 800,
        overlap: int = 100
    ) -> List[Dict]:
        """
        Chunk text with section awareness
        Strategy: Try to keep chunks within section boundaries when possible
        """
        chunks = []
        sections = self.detect_sections(text)
        
        if not sections:
            # Fallback to simple chunking if no sections detected
            return self.simple_chunk(text, source_id, chunk_size, overlap)
        
        chunk_id = 1
        
        for section in sections:
            section_text = text[section['start']:section['end']]
            section_name = section['name']
            
            # If section is small enough, keep as single chunk
            if len(section_text) <= chunk_size:
                chunks.append({
                    'source_id': source_id,
                    'chunk_id': f"chunk_{chunk_id:02d}",
                    'text': section_text.strip(),
                    'section': section_name,
                    'char_start': section['start'],
                    'char_end': section['end']
                })
                chunk_id += 1
            else:
                # Split large section into chunks with overlap
                section_chunks = self.simple_chunk(
                    section_text, 
                    source_id, 
                    chunk_size, 
                    overlap,
                    start_chunk_id=chunk_id
                )
                # Add section metadata
                for chunk in section_chunks:
                    chunk['section'] = section_name
                    chunks.append(chunk)
                chunk_id += len(section_chunks)
        
        return chunks
    
    def simple_chunk(
        self, 
        text: str, 
        source_id: str, 
        chunk_size: int = 800, 
        overlap: int = 100,
        start_chunk_id: int = 1
    ) -> List[Dict]:
        """Simple sliding window chunking"""
        chunks = []
        chunk_id = start_chunk_id
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'source_id': source_id,
                    'chunk_id': f"chunk_{chunk_id:02d}",
                    'text': chunk_text,
                    'section': 'unknown',
                    'char_start': start,
                    'char_end': end
                })
                chunk_id += 1
            
            start += (chunk_size - overlap)
        
        return chunks
    
    def process_document(self, source_id: str, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Process a single document: extract text and create chunks"""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            return "", []
        
        # Save processed text
        text_path = self.processed_dir / f"{source_id}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Create chunks
        chunks = self.chunk_with_section_awareness(text, source_id)
        
        # Save chunks
        chunks_path = self.processed_dir / f"{source_id}_chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        return str(text_path), chunks
    
    def process_corpus(self, manifest_path: str = "data/data_manifest.csv") -> pd.DataFrame:
        """Process entire corpus from manifest"""
        print("Loading manifest...")
        encodings = ["utf-8-sig", "utf-8", "gbk", "cp1252"]
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(manifest_path, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_err = e
        else:
            raise last_err
        
        all_chunks = []
        
        print("\nProcessing documents...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            source_id = row['source_id']
            raw_path = row['raw_path']

            if pd.isna(raw_path) or not os.path.exists(raw_path):
                print(f"Skipping {source_id}: PDF not found")
                continue
            
            try:
                text_path, chunks = self.process_document(source_id, raw_path)
                
                # Update manifest with processed path
                df.at[idx, 'processed_path'] = text_path
                
                # Collect all chunks
                all_chunks.extend(chunks)
                
                print(f"Processed {source_id}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {source_id}: {e}")
        
        # Save updated manifest
        for col in ["raw_path", "processed_path"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.replace("\\", "/", regex=False)

        df.to_csv(manifest_path, index=False, encoding='utf-8')
        
        # Save all chunks to a single file for easy loading
        all_chunks_path = self.processed_dir / "all_chunks.json"
        with open(all_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing complete!")
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"All chunks saved to: {all_chunks_path}")
        
        return df, all_chunks


def main():
    """Run ingestion pipeline"""
    processor = DocumentProcessor()
    df, chunks = processor.process_corpus()
    
    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"Documents processed: {df['processed_path'].notna().sum()}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunks per document: {len(chunks) / max(df['processed_path'].notna().sum(), 1):.1f}")


if __name__ == "__main__":
    main()
