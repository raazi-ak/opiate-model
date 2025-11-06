import os
import json
from typing import List, Dict, Optional, Tuple

import faiss  # type: ignore
import numpy as np


def read_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # lazy import
        except Exception:
            return ""
        try:
            reader = PdfReader(path)
            txt = []
            for pg in reader.pages:
                try:
                    txt.append(pg.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(txt)
        except Exception:
            return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as h:
            return h.read()
    except Exception:
        return ""


def chunk_text(txt: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if not txt:
        return []
    chunks = []
    i = 0
    n = len(txt)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(txt[i:j])
        if j == n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


class RagPipeline:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        # Lazy initialization to avoid startup hangs on macOS
        self.embed = None  # type: ignore
        self.dim = None  # type: ignore
        self.index_path = os.path.join(self.index_dir, "faiss.index")
        self.meta_path = os.path.join(self.index_dir, "meta.json")
        self.index = None
        self.meta: List[Dict[str, str]] = []
        self._load_index()

    def _ensure_embed(self) -> None:
        if self.embed is None:
            from sentence_transformers import SentenceTransformer  # local import
            # keep threads low to avoid mutex issues
            try:
                import torch  # type: ignore
                torch.set_num_threads(1)
            except Exception:
                pass
            m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.embed = m
            self.dim = self.embed.get_sentence_embedding_dimension()

    def _load_index(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as h:
                    self.meta = json.load(h)
            except Exception:
                self.index = None
                self.meta = []
        else:
            self.index = None
            self.meta = []

    def _save_index(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as h:
            json.dump(self.meta, h, ensure_ascii=False)

    def build_or_update_index(self, file_paths: List[str]) -> int:
        self._ensure_embed()
        texts: List[str] = []
        metas: List[Dict[str, str]] = []

        for p in file_paths:
            src = os.path.basename(p)
            txt = read_text_from_file(p)
            ch = chunk_text(txt)
            for c in ch:
                texts.append(c)
                metas.append({"source": src, "text": c})

        if not texts:
            return 0

        embs = self.embed.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        vecs = np.array(embs).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(int(self.dim))
            self.meta = []

        self.index.add(vecs)
        self.meta.extend(metas)
        self._save_index()
        return len(texts)

    def retrieve(self, query: str, k: int = 5, objective: Optional[str] = None, stress_level: Optional[str] = None) -> List[Dict[str, str]]:
        self._ensure_embed()
        if self.index is None or (self.index.ntotal or 0) == 0:
            return []
        parts = [query]
        if objective:
            parts.append(f"objective: {objective}")
        if stress_level:
            parts.append(f"stress: {stress_level}")
        q = " | ".join(parts)
        qv = self.embed.encode([q], normalize_embeddings=True)
        qv = np.array(qv).astype("float32")
        D, I = self.index.search(qv, k)
        out: List[Dict[str, str]] = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.meta):
                continue
            meta = self.meta[idx]
            out.append({"source": meta.get("source", "unknown"), "text": meta.get("text", "")})
        return out

    def format_context(self, docs: List[Dict[str, str]]) -> str:
        if not docs:
            return ""
        # Build a concise context with top passages and their sources
        context_parts: List[str] = []
        total_chars = 0
        max_chars = 1800  # keep prompt size reasonable
        for d in docs:
            source = d.get("source", "unknown")
            text = (d.get("text", "") or "").strip()
            if not text:
                continue
            snippet = text[:600]
            block = f"[source] {source}\n{snippet}"
            if total_chars + len(block) > max_chars:
                break
            context_parts.append(block)
            total_chars += len(block)
        return "\n\n".join(context_parts)


