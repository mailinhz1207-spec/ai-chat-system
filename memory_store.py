# memory_store.py
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sqlite3
import os
import uuid
from datetime import datetime

EMBED_MODEL = "all-MiniLM-L6-v2"

class MemoryStore:
    def __init__(self, namespace="global", faiss_index_path="faiss.index", db_path="memory.db"):
        self.namespace = namespace
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.dim = self.encoder.get_sentence_embedding_dimension()
        # init faiss
        if os.path.exists(faiss_index_path):
            self.index = faiss.read_index(faiss_index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.fpath = faiss_index_path
        # sqlite for metadata
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            content TEXT,
            summary TEXT,
            embedding_idx INTEGER,
            created_at TEXT,
            tags TEXT,
            importance REAL
        )
        """)
        self.conn.commit()

    def add_memory(self, user_id: str, content: str, metadata: dict = None):
        metadata = metadata or {}
        vec = self.encoder.encode([content], convert_to_numpy=True)
        # normalize then add
        faiss.normalize_L2(vec)
        idx_before = self.index.ntotal
        self.index.add(vec)
        self._save_index()
        idx_added = idx_before  # embedding_idx
        mem_id = str(uuid.uuid4())
        cur = self.conn.cursor()
        cur.execute("INSERT INTO memories (id, user_id, content, summary, embedding_idx, created_at, tags, importance) VALUES (?,?,?,?,?,?,?,?)",
                    (mem_id, user_id, content, metadata.get("summary",""), idx_added, datetime.utcnow().isoformat(), ",".join(metadata.get("tags",[])), metadata.get("importance", 0.5)))
        self.conn.commit()
        return mem_id

    def _save_index(self):
        faiss.write_index(self.index, self.fpath)

    def retrieve(self, user_id: str, query: str, k: int = 5) -> List[dict]:
        vec = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(vec, k)
        cur = self.conn.cursor()
        results = []
        for idx in I[0]:
            # find memory row by embedding_idx = idx
            cur.execute("SELECT id, content, summary, created_at, tags, importance FROM memories WHERE embedding_idx=?", (idx,))
            row = cur.fetchone()
            if row:
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "summary": row[2],
                    "created_at": row[3],
                    "tags": row[4].split(",") if row[4] else [],
                    "importance": row[5]
                })
        return results

    def condense(self, user_id: str, keep_top_n: int = 200):
        """
        A simple condense: keep top importance, merge older low-importance into summary entries.
        For production: call an LLM to summarize many memories into fewer.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, content, importance, created_at FROM memories WHERE user_id=? ORDER BY importance DESC, created_at DESC", (user_id,))
        rows = cur.fetchall()
        # simplistic: keep first keep_top_n, delete rest (but create a merged summary)
        keep = rows[:keep_top_n]
        drop = rows[keep_top_n:]
        if not drop:
            return
        merged = " ".join([r[1] for r in drop])
        # create condensed memory
        self.add_memory(user_id, content=f"Condensed summary: {merged[:2000]}", metadata={"summary":"auto-condensed", "tags":["condensed"], "importance":0.6})
        # delete dropped
        for r in drop:
            cur.execute("DELETE FROM memories WHERE id=?", (r[0],))
        self.conn.commit()
        self._save_index()
