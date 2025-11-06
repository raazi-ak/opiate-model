"""
RAG pipeline using Pinecone vector database + Firestore for metadata
Uses Gemini Embeddings API and semantic chunking
"""
import os
import uuid
from typing import List, Dict, Optional
import numpy as np
from firebase_admin import firestore
from pinecone import Pinecone, ServerlessSpec

from .firebase_config import get_firestore
from .firebase_config import get_user_stress_data


def read_text_from_file(path: str) -> str:
    """Read text from PDF, TXT, or MD files"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
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


def chunk_text_semantic(txt: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Semantic chunking: Split by paragraphs first, then sentences, then fallback to characters
    This preserves semantic coherence better than fixed-size chunks
    """
    if not txt:
        return []
    
    chunks = []
    
    # First, try to split by double newlines (paragraphs)
    paragraphs = txt.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + para
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If we got very few chunks or chunks are too large, fall back to sentence-based splitting
    if len(chunks) == 0 or (len(chunks) == 1 and len(chunks[0]) > chunk_size * 1.5):
        chunks = []
        # Split by sentences (period, exclamation, question mark)
        import re
        sentences = re.split(r'([.!?]\s+)', txt)
        # Recombine sentences with their punctuation
        sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                    for i in range(0, len(sentences), 2) if sentences[i].strip()]
        
        current_chunk = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            if len(current_chunk) + len(sent) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sent
            else:
                if current_chunk:
                    current_chunk += " " + sent
                else:
                    current_chunk = sent
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    # Final fallback: if still too large, use character-based splitting
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            # Split large chunk by characters
            i = 0
            while i < len(chunk):
                j = min(i + chunk_size, len(chunk))
                final_chunks.append(chunk[i:j])
                if j == len(chunk):
                    break
                i = j - overlap
                if i < 0:
                    i = 0
    
    return final_chunks if final_chunks else [txt]


class FirebaseRagPipeline:
    """RAG pipeline using Pinecone vector DB + Firestore for metadata"""
    
    COLLECTION_NAME = "document_chunks"
    PINECONE_INDEX_NAME = "study-assistant-rag"
    
    def __init__(self):
        self.db = get_firestore()
        self.collection = self.db.collection(self.COLLECTION_NAME)
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY must be set. Get free API key from: https://www.pinecone.io/"
            )
        
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Create or connect to index
        index_name = os.getenv("PINECONE_INDEX_NAME", self.PINECONE_INDEX_NAME)
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            # Create index if it doesn't exist
            pc.create_index(
                name=index_name,
                dimension=768,  # Gemini text-embedding-004 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Free tier region
                )
            )
            print(f"Created Pinecone index: {index_name}")
        
        self.pinecone_index = pc.Index(index_name)
        self.pinecone_index_name = index_name  # Store for clear_index
        
        # Lazy initialization of embedding model
        self.embed = None
        self.dim = 768  # Gemini text-embedding-004 dimension
    
    def _ensure_embed(self):
        """Initialize Gemini Embeddings API client lazily"""
        if self.embed is None:
            import google.generativeai as genai
            from dotenv import load_dotenv
            import os
            
            # Load API key
            APP_DIR = os.path.dirname(os.path.abspath(__file__))
            ENV_PATH = os.path.abspath(os.path.join(APP_DIR, "..", ".env"))
            if os.path.exists(ENV_PATH):
                load_dotenv(ENV_PATH)
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY must be set for embeddings")
            
            genai.configure(api_key=api_key)
            # Store both genai module and API key for embedding calls
            self.embed = genai
            self.embed_api_key = api_key
            self.dim = 768  # text-embedding-004 has 768 dimensions
    
    def build_or_update_index(self, file_paths: List[str], session_id: Optional[str] = None) -> int:
        """
        Build or update the Firestore index with document chunks
        
        Args:
            file_paths: List of file paths to process
            session_id: Optional session ID to associate documents with a session
            
        Returns:
            Number of chunks indexed
        """
        self._ensure_embed()
        
        texts: List[str] = []
        metas: List[Dict[str, str]] = []
        
        # Track chunk index per file to ensure unique labeling
        file_chunk_counters = {}
        
        # Process files and create chunks using semantic chunking
        for p in file_paths:
            src = os.path.basename(p)
            file_path = p  # Store full path for uniqueness
            
            # Initialize counter for this file if not exists
            if src not in file_chunk_counters:
                file_chunk_counters[src] = 0
            
            txt = read_text_from_file(p)
            ch = chunk_text_semantic(txt, chunk_size=1000, overlap=200)
            
            for c in ch:
                # Get file-specific chunk index
                chunk_idx = file_chunk_counters[src]
                
                texts.append(c)
                metas.append({
                    "source": src,
                    "text": c,
                    "file_path": file_path,  # Full path for uniqueness
                    "chunk_index": chunk_idx  # Per-file chunk index
                })
                
                # Increment counter for this file
                file_chunk_counters[src] += 1
        
        if not texts:
            return 0
        
        # Generate embeddings using Gemini Embeddings API via REST
        print(f"Generating embeddings for {len(texts)} chunks using Gemini Embeddings API...")
        embs = []
        
        # Use REST API for embeddings (more reliable than SDK method)
        import requests
        
        for i, text in enumerate(texts):
            try:
                # Call Gemini Embeddings API REST endpoint
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.embed_api_key}"
                payload = {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_DOCUMENT"
                }
                
                response = requests.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Extract embedding from response
                if "embedding" in result and "values" in result["embedding"]:
                    emb = np.array(result["embedding"]["values"])
                elif "embedding" in result:
                    emb = np.array(result["embedding"])
                else:
                    raise ValueError("Unexpected embedding response format")
                
                if len(emb) == 0:
                    raise ValueError("Empty embedding returned")
                
                embs.append(emb)
                
            except Exception as e:
                print(f"Error generating embedding for chunk {i+1}: {e}")
                # Fallback: use zeros (will be filtered out in similarity search)
                embs.append(np.zeros(self.dim))
            
            if (i + 1) % 50 == 0:
                print(f"Generated embeddings for {i+1}/{len(texts)} chunks...")
        
        # Convert to numpy array and normalize
        if not embs:
            raise ValueError("No embeddings generated")
        
        embs = np.array(embs)
        # Normalize to unit vectors for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        embs = embs / norms
        
        # Store embeddings in Pinecone and metadata in Firestore
        # Prepare vectors for Pinecone (upsert in batches)
        vectors_to_upsert = []
        firestore_batch = self.db.batch()
        firestore_count = 0
        total_added = 0
        
        for i, (text, meta, emb) in enumerate(zip(texts, metas, embs)):
            # Generate unique ID for this chunk
            # Format: source_filename_chunkIndex_uniqueHash
            # This ensures uniqueness even if same filename appears multiple times
            source_safe = meta['source'].replace(' ', '_').replace('.', '_')[:50]  # Sanitize filename
            chunk_idx = meta['chunk_index']
            unique_hash = uuid.uuid4().hex[:8]  # 8-char unique hash
            
            chunk_id = f"{source_safe}_{chunk_idx}_{unique_hash}"
            
            # Prepare vector for Pinecone
            pinecone_metadata = {
                "source": meta["source"],
                "text_preview": text[:200]  # Store preview for debugging
            }
            if session_id:
                pinecone_metadata["session_id"] = session_id
            
            vectors_to_upsert.append({
                "id": chunk_id,
                "values": emb.tolist(),
                "metadata": pinecone_metadata
            })
            
            # Store full text and metadata in Firestore (for retrieval)
            doc_ref = self.collection.document(chunk_id)
            firestore_doc = {
                "text": text,
                "source": meta["source"],  # Filename for display
                "file_path": meta.get("file_path", ""),  # Full path for uniqueness
                "chunk_index": meta["chunk_index"],  # Chunk index within file
                "chunk_id": chunk_id,
                "created_at": firestore.SERVER_TIMESTAMP
            }
            if session_id:
                firestore_doc["session_id"] = session_id
            
            firestore_batch.set(doc_ref, firestore_doc)
            
            firestore_count += 1
            
            # Upsert to Pinecone in batches of 100 (Pinecone batch limit)
            if len(vectors_to_upsert) >= 100:
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
            
            # Commit Firestore batch every 500 (Firestore limit)
            if firestore_count >= 500:
                try:
                    firestore_batch.commit()
                    firestore_batch = self.db.batch()
                    firestore_count = 0
                    total_added += 500
                    print(f"Indexed {total_added}/{len(texts)} chunks...")
                except Exception as firestore_error:
                    print(f"⚠️  Firestore write error (chunks still stored in Pinecone): {firestore_error}")
                    # Continue with Pinecone storage even if Firestore fails
                    firestore_batch = self.db.batch()
                    firestore_count = 0
        
        # Upsert remaining vectors to Pinecone
        if vectors_to_upsert:
            try:
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
            except Exception as pinecone_error:
                print(f"✗ Pinecone upsert error: {pinecone_error}")
        
        # Commit remaining Firestore documents
        if firestore_count > 0:
            try:
                firestore_batch.commit()
                total_added += firestore_count
            except Exception as firestore_error:
                print(f"⚠️  Firestore final commit error (chunks still in Pinecone): {firestore_error}")
                # Don't count Firestore failures in total_added since Pinecone succeeded
                print(f"  Note: {firestore_count} chunks stored in Pinecone but not in Firestore")
        
        print(f"✓ Indexed {total_added} chunks in Pinecone + Firestore")
        return total_added
    
    def retrieve(self, query: str, k: int = 5, objective: Optional[str] = None, 
                        stress_level: Optional[str] = None, user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Retrieve relevant documents using vector similarity search
        
        Args:
            query: User query
            k: Number of results to return
            objective: Optional learning objective
            stress_level: Optional stress level (can also be fetched from Realtime DB)
            user_id: Optional user ID to fetch real-time stress data
            session_id: Optional session ID to filter documents by session
            
        Returns:
            List of relevant document chunks
        """
        self._ensure_embed()
        
        # Check if query is asking about available files or what was uploaded
        query_lower = query.lower()
        file_query_phrases = [
            "what files", "what file", "which files", "list files", 
            "files do you have", "files are available", "what did i upload",
            "what's the file", "what is the file", "tell me about the file",
            "what's in the file", "what is in the file", "what is it about",
            "what's it about", "what is this about", "what's this about",
            "describe the file", "describe it", "summarize the file", "summarize it"
        ]
        
        # Track if we should use aggressive fallback (return documents even with low/no similarity)
        # This happens when we have a session_id but vector search fails
        if any(phrase in query_lower for phrase in file_query_phrases):
            # Check if asking "what is it about" vs "what files"
            is_about_query = any(phrase in query_lower for phrase in [
                "what is it about", "what's it about", "what is this about", "what's this about",
                "describe the file", "describe it", "summarize the file", "summarize it",
                "what's the file about", "what is the file about"
            ])
            
            try:
                # Query Firestore directly to get unique sources for this session
                firestore_query = self.collection
                if session_id:
                    firestore_query = firestore_query.where("session_id", "==", session_id)
                else:
                    # If no session_id, return empty (shouldn't happen but handle it)
                    print("⚠️  File query but no session_id provided")
                    return []
                
                # Get documents from this session
                all_docs = list(firestore_query.stream())
                
                if not all_docs:
                    print(f"No documents found in Firestore for session: {session_id}")
                    return []
                
                if is_about_query:
                    # For "what is it about" queries, return MORE content from ALL files for summarization
                    print(f"📄 'About' query detected - returning content from all files for summarization")
                    sources_seen = set()
                    results = []
                    
                    # Group by source and get multiple chunks per file
                    for doc in all_docs:
                        data = doc.to_dict()
                        source = data.get("source", "unknown")
                        text = data.get("text", "")
                        
                        if source not in sources_seen:
                            sources_seen.add(source)
                            # Get first chunk from this file
                            results.append({
                                "doc_id": doc.id,
                                "similarity": 0.95,  # High similarity for "about" queries
                                "source": source,
                                "text": text[:2000] if text else ""  # More text for summarization
                            })
                        else:
                            # Add more chunks from the same file for better context
                            if len([r for r in results if r["source"] == source]) < 3:  # Max 3 chunks per file
                                results.append({
                                    "doc_id": doc.id,
                                    "similarity": 0.9,
                                    "source": source,
                                    "text": text[:2000] if text else ""
                                })
                    
                    if results:
                        print(f"Found {len(sources_seen)} file(s) with {len(results)} chunks for summarization: {list(sources_seen)}")
                        return results[:k * 2]  # Return more chunks for "about" queries
                else:
                    # For "what files" queries, just list unique files with preview
                    sources_seen = set()
                    results = []
                    
                    for doc in all_docs:
                        data = doc.to_dict()
                        source = data.get("source", "unknown")
                        if source not in sources_seen:
                            sources_seen.add(source)
                            text = data.get("text", "")
                            results.append({
                                "doc_id": doc.id,
                                "similarity": 0.9,  # High similarity for file listing
                                "source": source,
                                "text": text[:1000] if text else ""  # First 1000 chars as preview
                            })
                            if len(results) >= k:
                                break
                    
                    if results:
                        print(f"Found {len(results)} files in session {session_id[:30]}...: {[r['source'] for r in results]}")
                        return results
                
                print(f"No unique files found in session")
                return []
            except Exception as e:
                print(f"Error retrieving file list: {e}")
                import traceback
                traceback.print_exc()
                return []
        
        # Get real-time stress data if user_id provided
        if user_id:
            stress_data = get_user_stress_data(user_id)
            if stress_data.get("stress_level") and not stress_level:
                stress_level = str(stress_data.get("stress_level"))
        
        # Build enhanced query
        parts = [query]
        if objective:
            parts.append(f"objective: {objective}")
        if stress_level:
            parts.append(f"stress: {stress_level}")
        enhanced_query = " | ".join(parts)
        
        # Generate query embedding using Gemini Embeddings API via REST
        try:
            import requests
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.embed_api_key}"
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": enhanced_query}]},
                "taskType": "RETRIEVAL_QUERY"
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract embedding from response
            if "embedding" in result and "values" in result["embedding"]:
                query_emb = np.array(result["embedding"]["values"])
            elif "embedding" in result:
                query_emb = np.array(result["embedding"])
            else:
                raise ValueError("Unexpected embedding response format")
            
            if len(query_emb) == 0:
                raise ValueError("Empty query embedding returned")
            
            # Normalize query embedding
            norm = np.linalg.norm(query_emb)
            if norm > 0:
                query_emb = query_emb / norm
            else:
                raise ValueError("Zero norm query embedding")
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
        
        # Use Pinecone for fast vector similarity search
        try:
            # Build filter for session_id if provided
            # Pinecone filter syntax: {"session_id": {"$eq": "value"}}
            filter_dict = None
            if session_id:
                filter_dict = {"session_id": {"$eq": session_id}}
            
            # Query Pinecone with the query embedding
            query_kwargs = {
                "vector": query_emb.tolist(),
                "top_k": k * 3,  # Get more results for better recall
                "include_metadata": True
            }
            if filter_dict:
                query_kwargs["filter"] = filter_dict
                print(f"🔍 Querying Pinecone with session_id filter: {session_id[:30]}...")
            
            query_response = self.pinecone_index.query(**query_kwargs)
            
            print(f"📊 Pinecone returned {len(query_response.matches)} matches")
            
            # Extract results from Pinecone
            results = []
            chunk_ids = []
            
            # Use a very low threshold when we have session_id (better recall)
            min_threshold = 0.1 if session_id else 0.2
            
            for match in query_response.matches:
                chunk_id = match.id
                similarity = float(match.score)
                
                if similarity < min_threshold:
                    continue
                
                print(f"  ✓ Match: {chunk_id[:30]}... (similarity: {similarity:.3f})")
                
                chunk_ids.append(chunk_id)
                # Don't trust Pinecone metadata for source - we'll get it from Firestore which is filtered by session_id
                results.append({
                    "doc_id": chunk_id,
                    "similarity": similarity,
                    "source": "unknown",  # Will be set from Firestore (source of truth)
                    "text": ""  # Will fetch from Firestore
                })
            
            # ROOT CAUSE FIX: If we have a session_id but got no/few results, 
            # ALWAYS fall back to returning documents from that session
            # This ensures the LLM has context to work with, even if vector similarity is low
            if session_id and len(chunk_ids) < k:
                print(f"⚠️  Only {len(chunk_ids)} results from vector search, using Firestore fallback for session: {session_id[:30]}...")
                
                try:
                    # Query Firestore directly for this session
                    firestore_query = self.collection.where("session_id", "==", session_id)
                    
                    # Get enough chunks to fill up to k results (or more for better context)
                    # If we got no results, get k*3 for better coverage. Otherwise just fill the gap.
                    initial_count = len(chunk_ids)
                    needed = (k * 3) if initial_count == 0 else (k - initial_count)
                    fallback_docs = list(firestore_query.limit(needed * 2).stream())  # Get extra to filter duplicates
                    
                    if fallback_docs:
                        print(f"Found {len(fallback_docs)} documents in Firestore fallback (need {needed} more)")
                        
                        # Group by source to get diverse content
                        sources_seen = {}
                        for doc in fallback_docs:
                            data = doc.to_dict()
                            source = data.get("source", "unknown")
                            chunk_id = doc.id
                            
                            # Skip if already in results
                            if chunk_id in chunk_ids:
                                continue
                            
                            if source not in sources_seen:
                                sources_seen[source] = []
                            
                            sources_seen[source].append({
                                "doc_id": chunk_id,
                                "similarity": 0.3,  # Low but acceptable for fallback
                                "source": source,
                                "text": data.get("text", "")
                            })
                        
                        # Get up to 3 chunks per file to ensure diversity
                        added_count = 0
                        for source, chunks in sources_seen.items():
                            for chunk in chunks[:3]:
                                if len(chunk_ids) >= initial_count + needed:
                                    break
                                results.append(chunk)
                                chunk_ids.append(chunk["doc_id"])
                                added_count += 1
                            
                            if len(chunk_ids) >= initial_count + needed:
                                break
                        
                        print(f"✓ Added {added_count} additional chunks from fallback (total: {len(chunk_ids)})")
                except Exception as e:
                    print(f"Error in Firestore fallback: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Fetch full text from Firestore for the retrieved chunk IDs
            if chunk_ids:
                # Batch fetch from Firestore by directly getting documents by ID
                firestore_docs = {}
                # Firestore batch get is more efficient than querying
                for chunk_id in chunk_ids:
                    try:
                        doc_ref = self.collection.document(chunk_id)
                        doc = doc_ref.get()
                        if doc.exists:
                            data = doc.to_dict()
                            # Verify session_id matches if provided
                            if session_id and data.get("session_id") != session_id:
                                continue  # Skip if session doesn't match
                            firestore_docs[chunk_id] = {
                                "text": data.get("text", ""),
                                "source": data.get("source", "unknown")
                            }
                    except Exception as e:
                        print(f"  ⚠️  Error fetching document {chunk_id[:30]}...: {e}")
                        continue
                
                # Add text and source to results (only from Firestore, which is filtered by session_id)
                for result in results:
                    chunk_id = result["doc_id"]
                    doc_data = firestore_docs.get(chunk_id, {})
                    result["text"] = doc_data.get("text", "")
                    result["source"] = doc_data.get("source", "unknown")
                    
                    # Remove results that don't exist in Firestore (old Pinecone vectors)
                    if not result["text"]:
                        print(f"  ⚠️  Removing result {chunk_id[:30]}... (not found in Firestore for this session)")
                
                # Filter out results without text (old Pinecone vectors not in Firestore)
                results = [r for r in results if r["text"]]
                chunk_ids = [r["doc_id"] for r in results]
            
            # Sort by similarity (should already be sorted, but ensure it)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Log retrieval results for debugging
            if results:
                print(f"Retrieved {len(results)} documents for query: '{query[:50]}...'")
                print(f"  Sources: {set(r['source'] for r in results)}")
            else:
                print(f"No documents retrieved for query: '{query[:50]}...' (session_id: {session_id})")
            
            return results[:k]
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def format_context(self, docs: List[Dict[str, str]]) -> str:
        """
        Format retrieved documents into context string with better structure
        Increased context window for better answers
        """
        if not docs:
            return ""
        
        context_parts: List[str] = []
        total_chars = 0
        max_chars = 8000  # Increased from 1800 to 8000 (Gemini supports large contexts)
        
        # Sort by similarity (highest first) to prioritize most relevant
        sorted_docs = sorted(docs, key=lambda x: x.get("similarity", 0), reverse=True)
        
        for idx, d in enumerate(sorted_docs, 1):
            source = d.get("source", "unknown")
            text = (d.get("text", "") or "").strip()
            similarity = d.get("similarity", 0)
            
            if not text:
                continue
            
            # Include more text per chunk (up to 1000 chars instead of 600)
            # But prioritize higher similarity chunks
            max_chunk_size = 1000 if similarity > 0.7 else 600
            
            snippet = text[:max_chunk_size]
            if len(text) > max_chunk_size:
                snippet += "..."
            
            # Better formatting with source filename clearly shown
            block = f"--- Context Document {idx} (Relevance: {similarity:.2f}) ---\nSource File: {source}\nContent: {snippet}\n"
            
            if total_chars + len(block) > max_chars:
                # Try to fit at least a smaller version
                if idx == 1:  # Always include the top result
                    snippet = text[:500]
                    block = f"--- Context Document {idx} (Relevance: {similarity:.2f}) ---\nSource: {source}\nContent: {snippet}\n"
                    context_parts.append(block)
                break
            
            context_parts.append(block)
            total_chars += len(block)
        
        return "\n".join(context_parts)
    
    def clear_index(self, session_id: Optional[str] = None):
        """
        Clear documents from both Pinecone and Firestore
        
        Args:
            session_id: If provided, only clear documents for this session. Otherwise, clear all.
        """
        # Clear Pinecone index
        try:
            stats = self.pinecone_index.describe_index_stats()
            total_vectors = stats.total_vector_count
            print(f"Pinecone index contains {total_vectors} vectors")
            
            if session_id:
                # Delete vectors for specific session
                # Fetch all vectors with this session_id and delete them
                # Note: Pinecone doesn't support querying by metadata directly for deletion
                # We'll need to fetch from Firestore first, then delete from Pinecone
                print(f"Clearing vectors for session: {session_id}")
            else:
                print("Clearing all vectors from Pinecone...")
                # For full clear, we'll delete the index and recreate it
                # This is simpler than deleting individual vectors
                index_name = getattr(self, 'pinecone_index_name', self.PINECONE_INDEX_NAME)
                from pinecone import Pinecone
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                pc.delete_index(index_name)
                print(f"Deleted Pinecone index: {index_name}")
                # Recreate it
                from pinecone import ServerlessSpec
                pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"Recreated Pinecone index: {index_name}")
                self.pinecone_index = pc.Index(index_name)
        except Exception as e:
            print(f"Error clearing Pinecone: {e}")
        
        # Clear Firestore
        query = self.collection
        if session_id:
            query = query.where("session_id", "==", session_id)
        
        docs = query.stream()
        batch = self.db.batch()
        count = 0
        total_count = 0
        
        chunk_ids_to_delete = []
        
        for doc in docs:
            data = doc.to_dict()
            chunk_ids_to_delete.append(doc.id)
            batch.delete(doc.reference)
            count += 1
            total_count += 1
            if count >= 500:
                batch.commit()
                batch = self.db.batch()
                count = 0
        
        if count > 0:
            batch.commit()
        
        # Delete from Pinecone if session-specific
        if session_id and chunk_ids_to_delete:
            try:
                # Delete in batches of 1000 (Pinecone limit)
                for i in range(0, len(chunk_ids_to_delete), 1000):
                    batch_ids = chunk_ids_to_delete[i:i+1000]
                    self.pinecone_index.delete(ids=batch_ids)
                print(f"Deleted {len(chunk_ids_to_delete)} vectors from Pinecone for session {session_id}")
            except Exception as e:
                print(f"Error deleting Pinecone vectors: {e}")
        
        print(f"Cleared {total_count} documents from Firestore" + (f" for session {session_id}" if session_id else ""))
