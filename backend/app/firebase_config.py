"""
Firebase configuration and initialization
"""
import os
import firebase_admin
from firebase_admin import credentials, firestore, db
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.abspath(os.path.join(APP_DIR, "..", ".env"))
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

# Firebase app instance
_firebase_app: Optional[firebase_admin.App] = None
_firestore_client: Optional[firestore.Client] = None
_realtime_db: Optional[db.Reference] = None


def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global _firebase_app, _firestore_client, _realtime_db
    
    if _firebase_app is not None:
        return _firebase_app
    
    # Check for service account key file
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if service_account_path:
        # Handle relative paths - resolve relative to backend directory
        if not os.path.isabs(service_account_path):
            service_account_path = os.path.join(APP_DIR, "..", service_account_path)
            service_account_path = os.path.abspath(service_account_path)
        
        if os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            _firebase_app = firebase_admin.initialize_app(cred)
        else:
            raise FileNotFoundError(f"Service account file not found: {service_account_path}")
    else:
        # Try to use default credentials (for Google Cloud environments)
        try:
            _firebase_app = firebase_admin.initialize_app()
        except Exception as e:
            # If no credentials found, use application default credentials
            # This requires GOOGLE_APPLICATION_CREDENTIALS env var or gcloud auth
            _firebase_app = firebase_admin.initialize_app(
                options={"projectId": os.getenv("FIREBASE_PROJECT_ID")}
            )
    
    _firestore_client = firestore.client()
    
    # Initialize Realtime Database if project ID is set
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    database_url = os.getenv("FIREBASE_DATABASE_URL")
    if project_id:
        # Use custom database URL if provided (for regional databases)
        if database_url:
            _realtime_db = db.reference(url=database_url)
        else:
            _realtime_db = db.reference()
    
    return _firebase_app


def get_firestore() -> firestore.Client:
    """Get Firestore client"""
    if _firestore_client is None:
        initialize_firebase()
    return _firestore_client


def get_realtime_db() -> Optional[db.Reference]:
    """Get Realtime Database reference"""
    if _realtime_db is None:
        initialize_firebase()
    return _realtime_db


def get_user_stress_data(user_id: Optional[str] = None) -> dict:
    """
    Get current stress level and heart rate from Realtime Database
    
    Args:
        user_id: Optional user ID. If None, uses default path or env var
    
    Returns:
        dict with 'stress_level' (as string: 'low', 'medium', 'high'), 'heart_rate', and 'timestamp'
    """
    try:
        rt_db = get_realtime_db()
        if rt_db is None:
            return {"stress_level": None, "heart_rate": None}
        
        user_id = user_id or os.getenv("FIREBASE_USER_ID", "default")
        # Path structure: /users/{user_id}/realtime/current
        path = f"users/{user_id}/realtime/current"
        
        snapshot = rt_db.child(path).get()
        
        if snapshot:
            # Get numeric stress level (0.0 to 1.0)
            stress_numeric = snapshot.get("stressLevel")
            heart_rate = snapshot.get("heartRate")
            timestamp = snapshot.get("timestamp")
            
            # Convert numeric stress level to categorical
            # Thresholds: < 0.3 = low, 0.3-0.7 = medium, > 0.7 = high
            if stress_numeric is not None:
                if stress_numeric < 0.3:
                    stress_level = "low"
                elif stress_numeric < 0.7:
                    stress_level = "medium"
                else:
                    stress_level = "high"
            else:
                stress_level = None
            
            return {
                "stress_level": stress_level,
                "stress_level_numeric": stress_numeric,
                "heart_rate": heart_rate,
                "timestamp": timestamp
            }
        return {"stress_level": None, "heart_rate": None}
    except Exception as e:
        print(f"Error reading Realtime DB: {e}")
        return {"stress_level": None, "heart_rate": None}
