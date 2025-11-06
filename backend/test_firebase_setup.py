#!/usr/bin/env python3
"""
Test script for Firebase + Pinecone + Gemini setup
Verifies all components are working correctly
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    print("⚠️  .env file not found. Make sure to create it with required variables.")
    sys.exit(1)


def test_environment_variables():
    """Test that all required environment variables are set"""
    print("\n=== Testing Environment Variables ===")
    
    required_vars = {
        "FIREBASE_PROJECT_ID": "Firebase project ID",
        "GEMINI_API_KEY": "Gemini API key (get from https://aistudio.google.com/app/apikey)",
        "PINECONE_API_KEY": "Pinecone API key (get from https://www.pinecone.io/)",
    }
    
    optional_vars = {
        "FIREBASE_SERVICE_ACCOUNT_PATH": "Firebase service account JSON path",
        "FIREBASE_DATABASE_URL": "Firebase Realtime Database URL",
        "PINECONE_INDEX_NAME": "Pinecone index name (default: study-assistant-rag)",
        "GEMINI_MODEL": "Gemini model name (default: gemini-2.5-flash)",
    }
    
    all_set = True
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "PASSWORD" in var:
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"✓ {var}: {masked}")
            else:
                print(f"✓ {var}: {value}")
        else:
            print(f"✗ {var}: NOT SET - {description}")
            all_set = False
    
    print("\nOptional variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if "KEY" in var or "PASSWORD" in var or "PATH" in var:
                print(f"  ✓ {var}: [set]")
            else:
                print(f"  ✓ {var}: {value}")
        else:
            print(f"  - {var}: not set (using default)")
    
    return all_set


def test_firebase():
    """Test Firebase Admin SDK initialization"""
    print("\n=== Testing Firebase ===")
    try:
        from app.firebase_config import initialize_firebase, get_firestore
        
        app = initialize_firebase()
        print("✓ Firebase initialized successfully")
        
        db = get_firestore()
        print("✓ Firestore client created")
        
        # Test a simple read (should not fail even if collection is empty)
        try:
            test_collection = db.collection("_test")
            docs = list(test_collection.limit(1).stream())
            print("✓ Firestore connection working")
        except Exception as e:
            print(f"⚠️  Firestore connection test: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Firebase initialization failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure FIREBASE_SERVICE_ACCOUNT_PATH points to valid JSON file")
        print("  2. Or ensure GOOGLE_APPLICATION_CREDENTIALS is set")
        print("  3. Check that Firebase Admin SDK is installed: pip install firebase-admin")
        return False


def test_pinecone():
    """Test Pinecone connection and index"""
    print("\n=== Testing Pinecone ===")
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("✗ PINECONE_API_KEY not set")
            return False
        
        pc = Pinecone(api_key=api_key)
        print("✓ Pinecone client created")
        
        # List existing indexes
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        print(f"✓ Found {len(index_names)} existing index(es)")
        
        # Check if our index exists or will be created
        index_name = os.getenv("PINECONE_INDEX_NAME", "study-assistant-rag")
        if index_name in index_names:
            print(f"✓ Index '{index_name}' exists")
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"  - Total vectors: {stats.total_vector_count}")
            print(f"  - Dimension: {stats.dimension}")
        else:
            print(f"ℹ️  Index '{index_name}' will be created automatically on first use")
            print(f"  - Dimension: 768 (Gemini text-embedding-004)")
            print(f"  - Metric: cosine")
            print(f"  - Spec: serverless (free tier)")
        
        return True
    except ImportError:
        print("✗ Pinecone client not installed")
        print("  Install with: pip install pinecone-client==5.0.1")
        return False
    except Exception as e:
        print(f"✗ Pinecone test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure PINECONE_API_KEY is correct")
        print("  2. Check your internet connection")
        print("  3. Verify Pinecone account is active at https://www.pinecone.io/")
        return False


def test_gemini_api():
    """Test Gemini API client"""
    print("\n=== Testing Gemini API ===")
    try:
        from app.firebase_llm import VertexAIClient
        
        client = VertexAIClient()
        print("✓ Gemini API client created")
        
        # Test a simple generation
        response = client.generate_text("Say 'Hello, Firebase + Pinecone!' in one word.")
        if response.startswith("[Gemini API error]"):
            print(f"✗ Gemini API error: {response}")
            print("\nTroubleshooting:")
            print("  1. Make sure GEMINI_API_KEY is correct")
            print("  2. Get a free API key from: https://aistudio.google.com/app/apikey")
            print("  3. Check that the API key has proper permissions")
            return False
        
        print(f"✓ Gemini API response: {response[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Gemini API test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure GEMINI_API_KEY is set in .env")
        print("  2. Get a free API key from: https://aistudio.google.com/app/apikey")
        print("  3. Check that google-generativeai is installed: pip install google-generativeai")
        return False


def test_gemini_embeddings():
    """Test Gemini Embeddings API"""
    print("\n=== Testing Gemini Embeddings API ===")
    try:
        import requests
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("✗ GEMINI_API_KEY not set")
            return False
        
        # Test embedding generation
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": "test embedding"}]},
            "taskType": "RETRIEVAL_DOCUMENT"
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "embedding" in result and "values" in result["embedding"]:
            emb_dim = len(result["embedding"]["values"])
            print(f"✓ Embedding generated successfully")
            print(f"  - Dimension: {emb_dim} (expected: 768)")
            if emb_dim == 768:
                print("  ✓ Dimension matches Gemini text-embedding-004")
            return True
        else:
            print("✗ Unexpected embedding response format")
            return False
            
    except ImportError:
        print("✗ requests library not installed")
        print("  Install with: pip install requests")
        return False
    except Exception as e:
        print(f"✗ Embeddings API test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure GEMINI_API_KEY is correct")
        print("  2. Check that the API key has access to embeddings")
        print("  3. Verify internet connection")
        return False


def test_rag_pipeline():
    """Test RAG pipeline initialization"""
    print("\n=== Testing RAG Pipeline ===")
    try:
        from app.firebase_rag import FirebaseRagPipeline
        
        pipeline = FirebaseRagPipeline()
        print("✓ RAG pipeline created")
        print(f"  - Using Pinecone for vector storage")
        print(f"  - Using Firestore for metadata")
        print(f"  - Using Gemini Embeddings API")
        return True
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure all dependencies are installed")
        print("  2. Check that PINECONE_API_KEY is set")
        print("  3. Verify Firebase is initialized correctly")
        return False


def test_realtime_db():
    """Test Firebase Realtime Database connection"""
    print("\n=== Testing Realtime Database ===")
    try:
        from app.firebase_config import get_realtime_db, get_user_stress_data
        
        rt_db = get_realtime_db()
        if rt_db is None:
            print("⚠️  Realtime Database not configured (FIREBASE_DATABASE_URL not set)")
            print("  This is optional - only needed for real-time stress data from iOS app")
            return True  # Not a failure, just optional
        
        print("✓ Realtime Database client created")
        
        # Test reading (should not fail even if path doesn't exist)
        try:
            test_data = get_user_stress_data("test_user")
            print("✓ Realtime Database connection working")
            if test_data.get("stress_level") or test_data.get("heart_rate"):
                print(f"  - Found test data: {test_data}")
            else:
                print("  - No test data found (this is normal)")
        except Exception as e:
            print(f"⚠️  Realtime DB read test: {e}")
        
        return True
    except Exception as e:
        print(f"⚠️  Realtime Database test: {e}")
        print("  This is optional - only needed for iOS app integration")
        return True  # Not a critical failure


def main():
    """Run all tests"""
    print("=" * 60)
    print("Firebase + Pinecone + Gemini Setup Test")
    print("=" * 60)
    
    results = {
        "Environment Variables": test_environment_variables(),
        "Firebase": test_firebase(),
        "Pinecone": test_pinecone(),
        "Gemini API": test_gemini_api(),
        "Gemini Embeddings": test_gemini_embeddings(),
        "RAG Pipeline": test_rag_pipeline(),
        "Realtime DB": test_realtime_db(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Your setup is ready to use.")
        print("\nNext steps:")
        print("  1. Upload documents via /upload endpoint")
        print("  2. Ingest documents via /ingest endpoint")
        print("  3. Start asking questions via /chat endpoint")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check .env file has all required variables")
        print("  - Verify API keys are correct")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
