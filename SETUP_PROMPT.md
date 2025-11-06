# Complete Setup Prompt for AI Assistant

Copy and paste this entire prompt into Google AI Studio or any AI console to get comprehensive setup guidance:

---

I'm building a study assistant application with RAG (Retrieval-Augmented Generation) capabilities. Here's my complete setup:

## Project Overview
- **Backend**: FastAPI (Python) with Firebase integration
- **Frontend**: Next.js web app + iOS app (already using Firebase Realtime Database)
- **Purpose**: Study assistant that answers questions from uploaded documents (PDFs, TXT, MD)
- **Firebase Project**: `opiate-a4919`
- **Database URL**: `https://opiate-a4919-default-rtdb.asia-southeast1.firebasedatabase.app`

## Current Architecture

### Backend Components:
1. **Firebase RAG Pipeline** (`firebase_rag.py`):
   - Uses Firestore to store document chunks with embeddings
   - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
   - Chunks documents: 900 chars with 150 overlap
   - Currently uses approximate search (fetches 1000 docs, computes cosine similarity in-memory)
   - Collection name: `document_chunks`
   - Stores: `text`, `source`, `embedding` (as list), `created_at`

2. **Gemini LLM Client** (`firebase_llm.py`):
   - Uses Gemini Developer API (free tier, API key based)
   - Model: `gemini-2.5-flash`
   - Supports streaming responses
   - Builds prompts with: user message, objective, retrieved context, stress level

3. **Firebase Config** (`firebase_config.py`):
   - Initializes Firebase Admin SDK
   - Connects to Firestore
   - Connects to Realtime Database (for stress/heart rate data from iOS)
   - Service account: `opiate-a4919-bea6f19622a7.json`

4. **Main API** (`main.py`):
   - `/upload` - Upload documents
   - `/ingest` - Process and index documents in Firestore
   - `/chat` - Non-streaming chat with RAG
   - `/chat/stream` - Streaming chat with RAG
   - Supports: `user_id`, `objective`, `stress_level` parameters

### What's Working:
- ✅ Firebase Admin SDK initialized
- ✅ Firestore connection working
- ✅ Gemini API working (using API key)
- ✅ Document upload working
- ✅ RAG pipeline created (but not optimized)

## What I Need Help With:

### 1. Firestore Vector Search Setup
Currently, my RAG implementation:
- Fetches up to 1000 documents from Firestore
- Computes cosine similarity in-memory
- This won't scale well

**Questions:**
- Should I set up a Firestore vector index? How?
- What's the best approach for vector search in Firestore?
- Should I use a different vector database (Pinecone, Weaviate, etc.)?
- What are the performance implications of each approach?

### 2. RAG Optimization
- How can I improve retrieval accuracy?
- Should I use different embedding models?
- How to handle large document collections efficiently?
- Best practices for chunking strategies?

### 3. Firebase Configuration
- Do I need to enable any specific APIs in Google Cloud Console?
- What Firestore indexes do I need to create?
- Any security rules needed for Firestore?
- How to optimize Firestore queries for RAG?

### 4. Integration with iOS App
- My iOS app writes stress level and heart rate to Realtime Database
- Backend should read this data when `user_id` is provided
- Path structure: `users/{user_id}/vitals` with `stress_level`, `heart_rate`, `timestamp`
- Is this the right approach?

### 5. Production Readiness
- What monitoring/logging should I add?
- How to handle errors gracefully?
- Rate limiting considerations?
- Cost optimization tips?

## Specific Technical Questions:

1. **Vector Search**: Should I:
   - Set up Firestore vector index (if possible)?
   - Use a dedicated vector DB?
   - Keep current in-memory approach (with optimizations)?

2. **Embeddings**: Is `all-MiniLM-L6-v2` good for study materials, or should I use:
   - Larger models for better accuracy?
   - Domain-specific models?
   - Gemini's embedding API?

3. **Chunking**: Current strategy (900 chars, 150 overlap):
   - Is this optimal for study materials?
   - Should I use semantic chunking?
   - How to handle PDFs with tables/diagrams?

4. **Context Window**: Currently limit to 1800 chars:
   - Should I increase this?
   - How to balance context vs. token costs?
   - Best practices for context formatting?

5. **Query Enhancement**: Currently combines query + objective + stress_level:
   - Is this the best approach?
   - Should I use query expansion?
   - How to improve retrieval relevance?

## Environment Setup:
- Python 3.12
- FastAPI backend
- Firebase Admin SDK
- Firestore for vector storage
- Gemini Developer API (free tier)
- Sentence Transformers for embeddings

## Goals:
1. Efficient RAG that scales to thousands of documents
2. Fast retrieval (< 1 second for queries)
3. Accurate results (relevant chunks retrieved)
4. Source attribution (know which document answer came from)
5. Integration with real-time stress data from iOS app

Please provide:
1. Step-by-step setup instructions for optimal RAG
2. Code improvements/optimizations
3. Firebase/Google Cloud configuration checklist
4. Best practices for production deployment
5. Any missing pieces in my current implementation

---

