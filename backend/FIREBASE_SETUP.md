# Firebase Setup Guide

This guide will help you set up Firebase for the backend migration.

## Prerequisites

1. A Firebase project (you should already have one if you're using Realtime Database)
2. Google Cloud project (Firebase projects are also Google Cloud projects)
3. Service account credentials

## Setup Steps

### 1. Enable Required Services

In your Firebase/Google Cloud Console:

1. **Enable Vertex AI API:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to "APIs & Services" > "Library"
   - Search for "Vertex AI API" and enable it

2. **Enable Firestore:**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Select your project
   - Go to "Firestore Database" and create a database (if not already created)
   - Choose "Start in production mode" (we'll set up security rules later)

### 2. Create Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to "IAM & Admin" > "Service Accounts"
3. Click "Create Service Account"
4. Give it a name (e.g., "opiate-backend")
5. Grant these roles:
   - **Cloud Datastore User** (for Firestore)
   - **Vertex AI User** (for Vertex AI Gemini)
   - **Firebase Realtime Database Admin** (for reading stress/heart rate data)
6. Click "Done"
7. Click on the created service account
8. Go to "Keys" tab
9. Click "Add Key" > "Create new key" > "JSON"
10. Download the JSON file

### 3. Configure Environment Variables

Create or update `backend/.env`:

```bash
# Firebase/Google Cloud Configuration
FIREBASE_PROJECT_ID=your-project-id
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Service Account (choose one method)
# Method 1: Path to service account JSON file
FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/service-account-key.json

# Method 2: Use GOOGLE_APPLICATION_CREDENTIALS (alternative)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Vertex AI Model
VERTEX_AI_MODEL=gemini-1.5-flash

# Optional: User ID for Realtime DB (if you want default user)
FIREBASE_USER_ID=default
```

### 4. Set Up Firestore Vector Index (Optional but Recommended)

For better performance with vector search, you can set up a Firestore vector index:

1. Go to Firestore in Firebase Console
2. Create an index on the `document_chunks` collection
3. Fields: `embedding` (vector type)

**Note:** The current implementation uses approximate search by fetching documents and computing similarity. For production with large datasets, consider setting up a proper vector index.

### 5. Realtime Database Structure

Make sure your Realtime Database has this structure (adjust paths in `firebase_config.py` if different):

```
{
  "users": {
    "{user_id}": {
      "vitals": {
        "stress_level": 1,
        "heart_rate": 75,
        "timestamp": 1234567890
      }
    }
  }
}
```

## Testing the Setup

1. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Test Firebase connection:
   ```python
   from app.firebase_config import initialize_firebase, get_firestore
   initialize_firebase()
   db = get_firestore()
   print("✓ Firebase connected!")
   ```

3. Test Vertex AI:
   ```python
   from app.firebase_llm import VertexAIClient
   client = VertexAIClient()
   response = client.generate_text("Hello, test message")
   print(response)
   ```

## Migration from FAISS

If you have existing FAISS indexes, you'll need to re-ingest your documents:

1. Make sure your documents are in `data/uploads/`
2. Call the `/ingest` endpoint (this will process files and store in Firestore)
3. The old FAISS index files can be removed (they're no longer used)

## Troubleshooting

### "Permission denied" errors
- Make sure your service account has the required roles
- Check that the service account key file path is correct

### "Project not found" errors
- Verify `FIREBASE_PROJECT_ID` matches your Firebase project ID
- Check that Vertex AI API is enabled

### Vector search is slow
- Consider setting up a Firestore vector index
- Reduce the limit in `firebase_rag.py` `retrieve()` method (currently 1000)

### Realtime DB not accessible
- Check your Realtime Database rules allow read access
- Verify the path structure matches your iOS app's structure
- Update `get_user_stress_data()` path in `firebase_config.py` if needed

## Cost Considerations

- **Firestore:** Free tier includes 1GB storage, 50K reads/day, 20K writes/day
- **Vertex AI:** Pay per use, check [pricing](https://cloud.google.com/vertex-ai/pricing)
- **Realtime Database:** Free tier includes 1GB storage, 10GB/month transfer

For development, you should be well within free tiers. Monitor usage in Google Cloud Console.

