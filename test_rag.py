#!/usr/bin/env python3
"""
Test script for the RAG system with LM Studio
"""
import os
import sys
import requests
import time

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_health():
    """Test if the server is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start it with: python -m backend.app.main")
        return False

def test_ingest():
    """Test the ingest endpoint"""
    try:
        response = requests.post("http://localhost:8000/ingest")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ingested {data.get('ingested', 0)} documents")
            return True
        else:
            print(f"❌ Ingest failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ingest failed with error: {e}")
        return False

def test_chat(message, objective=None):
    """Test the chat endpoint"""
    try:
        data = {"message": message, "k": 3}
        if objective:
            data["objective"] = objective
        
        response = requests.post("http://localhost:8000/chat", data=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat response received")
            print(f"Answer: {result.get('answer', 'No answer')}")
            print(f"References: {len(result.get('references', []))} sources")
            return True
        else:
            print(f"❌ Chat failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat failed with error: {e}")
        return False

def main():
    print("🧪 Testing RAG System with LM Studio")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing server health...")
    if not test_health():
        return
    
    # Test 2: Ingest documents
    print("\n2. Testing document ingestion...")
    if not test_ingest():
        return
    
    # Test 3: Chat with the system
    print("\n3. Testing chat functionality...")
    test_queries = [
        ("What is machine learning?", "Learn about ML basics"),
        ("Explain Python functions", "Understand Python programming"),
        ("What are the types of machine learning?", None)
    ]
    
    for i, (query, objective) in enumerate(test_queries, 1):
        print(f"\n   Test 3.{i}: {query}")
        if objective:
            print(f"   Objective: {objective}")
        test_chat(query, objective)
        time.sleep(1)  # Small delay between requests
    
    print("\n🎉 RAG system testing completed!")

if __name__ == "__main__":
    main()
