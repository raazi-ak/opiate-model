from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict
import os
import uvicorn
import time
from threading import Thread
from dotenv import load_dotenv

from .firebase_rag import FirebaseRagPipeline
from .firebase_llm import VertexAIClient
from .firebase_config import initialize_firebase, get_user_stress_data


APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
UPLOAD_DIR = os.path.join(ROOT_DIR, "data", "uploads")
INDEX_DIR = os.path.join(ROOT_DIR, "data", "index")


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Load env from backend/.env
ENV_PATH = os.path.abspath(os.path.join(APP_DIR, "..", ".env"))
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

# Initialize Firebase
try:
    initialize_firebase()
    print("✓ Firebase initialized successfully")
except Exception as e:
    print(f"⚠ Firebase initialization warning: {e}")
    print("  Make sure FIREBASE_SERVICE_ACCOUNT_PATH or GOOGLE_APPLICATION_CREDENTIALS is set")

app = FastAPI(title="Study RAG Backend (Firebase)", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


rag: Optional[FirebaseRagPipeline] = None
llm: Optional[VertexAIClient] = None

# Stress data cache with TTL (1 minute)
stress_cache: Dict[str, Dict] = {}
CACHE_TTL = 60  # 1 minute in seconds

def poll_stress_data():
    """Background thread to poll stress data every 10 seconds"""
    while True:
        try:
            # Poll for the demo user (hardcoded for single-user demo)
            # TODO: Extend to support multiple users if needed
            user_id = "raazifaisal710729"  # Hardcoded for demo
            stress_data = get_user_stress_data(user_id)
            
            # Always update cache if we got any data (even if stress_level is None)
            # This ensures heart rate and other fields are updated
            if stress_data:
                stress_cache[user_id] = {
                    "data": stress_data,
                    "timestamp": time.time(),
                    "is_live": True
                }
                stress_level = stress_data.get('stress_level', 'unknown')
                heart_rate = stress_data.get('heart_rate', 'N/A')
                print(f"✓ Polled stress data for {user_id}: stress={stress_level}, HR={heart_rate}")
        except Exception as e:
            print(f"Error polling stress data: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(10)  # Poll every 10 seconds

# Start background polling thread
polling_thread = Thread(target=poll_stress_data, daemon=True)
polling_thread.start()
print("✓ Started background stress data polling (every 10 seconds)")


def get_rag() -> FirebaseRagPipeline:
    """Get or create Firebase RAG pipeline"""
    global rag
    if rag is None:
        rag = FirebaseRagPipeline()
    return rag


def get_llm() -> VertexAIClient:
    """Get or create Vertex AI client"""
    global llm
    if llm is None:
        llm = VertexAIClient()
    return llm


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/clear")
def clear_session(session_id: str = Form(...)):
    """
    Clear all documents for a specific session from both Firestore and Pinecone
    
    Args:
        session_id: Session ID to clear
    """
    rp = get_rag()
    try:
        rp.clear_index(session_id=session_id)
        return {"status": "success", "message": f"Cleared session {session_id}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/stress/{user_id}")
def get_stress(user_id: str):
    """
    Get stress data for a user (from cache or fetch live)
    Returns whether data is live or cached
    
    Data is considered "live" if:
    - It was polled by the background thread within the last 15 seconds (polling happens every 10s)
    - OR it was just fetched fresh
    
    Data is considered "cached" if:
    - It's in cache but older than 15 seconds (still within 1-minute TTL)
    """
    current_time = time.time()
    LIVE_THRESHOLD = 15  # Consider data "live" if polled within last 15 seconds
    
    # Check cache first
    if user_id in stress_cache:
        cached = stress_cache[user_id]
        age = current_time - cached["timestamp"]
        
        if age < CACHE_TTL:
            # Data is in cache and not expired
            # If age < 15 seconds, it's "live" (polled recently by background thread)
            # If age >= 15 seconds, it's "cached" (old but still valid)
            is_recently_polled = age < LIVE_THRESHOLD
            
            return {
                **cached["data"],
                "is_live": is_recently_polled,  # Live if polled within last 15 seconds
                "is_cached": not is_recently_polled,  # Cached if older than 15 seconds
                "cache_age_seconds": int(age)
            }
        else:
            # Cache expired, fetch fresh
            stress_data = get_user_stress_data(user_id)
            if stress_data.get("stress_level") is not None:
                stress_cache[user_id] = {
                    "data": stress_data,
                    "timestamp": current_time,
                    "is_live": True
                }
                return {
                    **stress_data,
                    "is_live": True,
                    "is_cached": False,
                    "cache_age_seconds": 0
                }
    
    # No cache or expired, fetch fresh
    stress_data = get_user_stress_data(user_id)
    if stress_data.get("stress_level") is not None:
        stress_cache[user_id] = {
            "data": stress_data,
            "timestamp": current_time,
            "is_live": True
        }
        return {
            **stress_data,
            "is_live": True,
            "is_cached": False,
            "cache_age_seconds": 0
        }
    
    return {
        "stress_level": None,
        "heart_rate": None,
        "is_live": False,
        "is_cached": False,
        "cache_age_seconds": None
    }


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        name = os.path.basename(f.filename)
        if not name:
            continue
        out_path = os.path.join(UPLOAD_DIR, name)
        data = await f.read()
        with open(out_path, "wb") as h:
            h.write(data)
        saved.append(name)
    return {"saved": saved}


@app.post("/ingest")
def ingest(
    clear_existing: bool = Form(False), 
    session_id: Optional[str] = Form(None),
    file_names: Optional[str] = Form(None)  # Comma-separated list of file names to ingest
):
    """
    Ingest documents into Firestore vector index
    
    Args:
        clear_existing: If True, clear existing index before ingesting
        session_id: Session ID to associate documents with. If not provided, generates a new one.
        file_names: Optional comma-separated list of specific file names to ingest. 
                   If not provided, ingests all files in uploads directory.
    """
    rp = get_rag()
    
    # Generate session_id if not provided
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
    
    if clear_existing:
        print(f"Clearing existing index for session: {session_id}...")
        rp.clear_index(session_id=session_id)
    
    # If specific file names provided, only ingest those
    if file_names:
        file_list = [name.strip() for name in file_names.split(',') if name.strip()]
        files = [os.path.join(UPLOAD_DIR, name) for name in file_list]
        files = [p for p in files if os.path.isfile(p) and os.path.splitext(p)[1].lower() in {".pdf", ".txt", ".md"}]
    else:
        # Fallback: ingest all files in uploads directory (for backward compatibility)
        files = [os.path.join(UPLOAD_DIR, p) for p in os.listdir(UPLOAD_DIR)]
        files = [p for p in files if os.path.isfile(p) and os.path.splitext(p)[1].lower() in {".pdf", ".txt", ".md"}]
    
    if not files:
        return JSONResponse(status_code=400, content={"error": "no files to ingest"})
    
    print(f"Ingesting {len(files)} file(s): {[os.path.basename(f) for f in files]}")
    
    try:
        n = rp.build_or_update_index(files, session_id=session_id)
        return {"ingested": n, "status": "success", "session_id": session_id, "files": [os.path.basename(f) for f in files]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/chat")
def chat(
    message: str = Form(...),
    objective: Optional[str] = Form(None),
    stress_level: Optional[str] = Form(None),
    k: int = Form(5),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    if not message:
        return JSONResponse(status_code=400, content={"error": "message required"})

    rp = get_rag()
    lc = get_llm()
    
    # Get real-time stress data from cache (polled every 10 seconds) if user_id provided
    heart_rate = None
    if user_id and not stress_level:
        # Use cached data if available (faster than fetching)
        current_time = time.time()
        if user_id in stress_cache:
            cached = stress_cache[user_id]
            age = current_time - cached["timestamp"]
            if age < CACHE_TTL:
                stress_level = str(cached["data"].get("stress_level", "")) if cached["data"].get("stress_level") else stress_level
                heart_rate = cached["data"].get("heart_rate")
            else:
                # Cache expired, fetch fresh
                stress_data = get_user_stress_data(user_id)
                stress_level = str(stress_data.get("stress_level", "")) if stress_data.get("stress_level") else stress_level
                heart_rate = stress_data.get("heart_rate")
        else:
            # No cache, fetch fresh
            stress_data = get_user_stress_data(user_id)
            stress_level = str(stress_data.get("stress_level", "")) if stress_data.get("stress_level") else stress_level
            heart_rate = stress_data.get("heart_rate")
    
    # Retrieve relevant documents (filtered by session_id if provided)
    docs = []
    try:
        docs = rp.retrieve(message, k=k, objective=objective, stress_level=stress_level, user_id=user_id, session_id=session_id)
    except Exception as e:
        print(f"Retrieval error: {e}")
        docs = []
    
    ctx = rp.format_context(docs)

    # Build prompt and generate response
    prompt = lc.build_prompt(
        user_message=message,
        objective=objective,
        retrieved_context=ctx,
        stress_level=stress_level,
        heart_rate=heart_rate,
        session_id=session_id,  # Pass session_id for fresh context
    )
    answer = lc.generate_text(prompt)
    
    return {
        "answer": answer,
        "references": [{"source": d.get("source", ""), "text": d.get("text", "")} for d in docs],
    }


@app.post("/chat/stream")
def chat_stream(
    message: str = Form(...),
    objective: Optional[str] = Form(None),
    stress_level: Optional[str] = Form(None),
    k: int = Form(5),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    if not message:
        return JSONResponse(status_code=400, content={"error": "message required"})

    rp = get_rag()
    lc = get_llm()
    
    # Get real-time stress data from cache (polled every 10 seconds) if user_id provided
    heart_rate = None
    if user_id and not stress_level:
        # Use cached data if available (faster than fetching)
        current_time = time.time()
        if user_id in stress_cache:
            cached = stress_cache[user_id]
            age = current_time - cached["timestamp"]
            if age < CACHE_TTL:
                stress_level = str(cached["data"].get("stress_level", "")) if cached["data"].get("stress_level") else stress_level
                heart_rate = cached["data"].get("heart_rate")
            else:
                # Cache expired, fetch fresh
                stress_data = get_user_stress_data(user_id)
                stress_level = str(stress_data.get("stress_level", "")) if stress_data.get("stress_level") else stress_level
                heart_rate = stress_data.get("heart_rate")
        else:
            # No cache, fetch fresh
            stress_data = get_user_stress_data(user_id)
            stress_level = str(stress_data.get("stress_level", "")) if stress_data.get("stress_level") else stress_level
            heart_rate = stress_data.get("heart_rate")

    # Retrieve relevant documents (filtered by session_id if provided)
    docs = []
    try:
        docs = rp.retrieve(message, k=k, objective=objective, stress_level=stress_level, user_id=user_id, session_id=session_id)
    except Exception as e:
        print(f"Retrieval error: {e}")
        docs = []
    
    ctx = rp.format_context(docs)

    prompt = lc.build_prompt(
        user_message=message,
        objective=objective,
        retrieved_context=ctx,
        stress_level=stress_level,
        heart_rate=heart_rate,
        session_id=session_id,  # Pass session_id for fresh context
    )
    
    def generator():
        # Stream response from Vertex AI
        for chunk in lc.stream_text(prompt):
            if chunk:
                yield chunk
    
    return StreamingResponse(generator(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


