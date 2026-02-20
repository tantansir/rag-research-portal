"""
Research Thread Manager
Manages saved research threads (query + retrieved evidence + answer)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class ThreadManager:
    """Manages research threads (saved queries with evidence and answers)"""
    
    def __init__(self, threads_dir: str = "outputs/threads"):
        self.threads_dir = Path(threads_dir)
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        self.threads_index_path = self.threads_dir / "threads_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load threads index"""
        if self.threads_index_path.exists():
            with open(self.threads_index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = []
    
    def _save_index(self):
        """Save threads index"""
        with open(self.threads_index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)
    
    def save_thread(self, query: str, result: Dict, title: Optional[str] = None) -> str:
        """
        Save a research thread
        
        Args:
            query: The research query
            result: RAG result dictionary (from rag.query())
            title: Optional title for the thread
        
        Returns:
            thread_id: Unique identifier for the saved thread
        """
        thread_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        thread_data = {
            "thread_id": thread_id,
            "title": title or query[:100],
            "query": query,
            "timestamp": timestamp,
            "answer": result.get("answer", ""),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "citations_used": result.get("citations_used", []),
            "model": result.get("model"),
            "prompt_version": result.get("prompt_version"),
            "retrieval_config": result.get("retrieval_config", {}),
            "run_id": result.get("run_id"),
        }
        
        # Save thread file
        thread_file = self.threads_dir / f"{thread_id}.json"
        with open(thread_file, 'w', encoding='utf-8') as f:
            json.dump(thread_data, f, ensure_ascii=False, indent=2)
        
        # Update index
        index_entry = {
            "thread_id": thread_id,
            "title": thread_data["title"],
            "query": query,
            "timestamp": timestamp,
        }
        self.index.insert(0, index_entry)  # Most recent first
        self._save_index()
        
        return thread_id
    
    def get_thread(self, thread_id: str) -> Optional[Dict]:
        """Retrieve a thread by ID"""
        thread_file = self.threads_dir / f"{thread_id}.json"
        if not thread_file.exists():
            return None
        
        with open(thread_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_threads(self, limit: Optional[int] = None) -> List[Dict]:
        """List all threads (most recent first)"""
        threads = self.index[:limit] if limit else self.index
        return threads
    
    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread"""
        thread_file = self.threads_dir / f"{thread_id}.json"
        if thread_file.exists():
            thread_file.unlink()
        
        # Remove from index
        self.index = [t for t in self.index if t["thread_id"] != thread_id]
        self._save_index()
        return True
    
    def search_threads(self, query: str) -> List[Dict]:
        """Search threads by query text or title"""
        query_lower = query.lower()
        matching = [
            t for t in self.index
            if query_lower in t.get("title", "").lower() or query_lower in t.get("query", "").lower()
        ]
        return matching
