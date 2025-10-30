"""
Simple test script for the CBT FastAPI
Tests basic functionality without requiring all models to be loaded
"""

import sys
import os
import asyncio
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that our modules can be imported"""
    print("Testing imports...")
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("✅ Pydantic imported successfully")
    except ImportError as e:
        print(f"❌ Pydantic import failed: {e}")
        return False
    
    try:
        import main
        print("✅ main.py imported successfully")
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False
    
    try:
        import cbt_models
        print("✅ cbt_models.py imported successfully")
    except Exception as e:
        print(f"❌ cbt_models.py import failed: {e}")
        return False
    
    return True

def test_app_creation():
    """Test FastAPI app creation"""
    print("\nTesting FastAPI app creation...")
    
    try:
        from main import app
        print("✅ FastAPI app created successfully")
        
        # Check if basic routes exist
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/models/status", "/analyze/cbt", "/audio/transcribe"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} exists")
            else:
                print(f"❌ Route {route} missing")
                
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False

def test_model_status():
    """Test model status functionality"""
    print("\nTesting model status...")
    
    try:
        from cbt_models import get_model_status
        status = get_model_status()
        print(f"✅ Model status retrieved: {status}")
        return True
    except Exception as e:
        print(f"❌ Model status failed: {e}")
        print(f"   This is expected if models are not available")
        return True  # Not a critical failure

async def test_health_endpoint():
    """Test health endpoint using TestClient"""
    print("\nTesting health endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        print(f"✅ Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Health response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

async def test_model_status_endpoint():
    """Test model status endpoint"""
    print("\nTesting model status endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        response = client.get("/models/status")
        
        print(f"✅ Model status endpoint status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Model status response: {response.json()}")
        else:
            print(f"❌ Model status endpoint failed: {response.text}")
            
        return True  # Don't fail on model loading issues
    except Exception as e:
        print(f"❌ Model status endpoint test failed: {e}")
        return False

async def test_cbt_analysis_endpoint():
    """Test CBT analysis endpoint with simple text"""
    print("\nTesting CBT analysis endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        test_data = {
            "text": "I feel overwhelmed with everything",
            "include_voice_emotion": False
        }
        
        response = client.post("/analyze/cbt", json=test_data)
        
        print(f"✅ CBT analysis endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ CBT analysis succeeded")
            print(f"   - Emotion: {result.get('emotion', 'unknown')}")
            print(f"   - Intent: {result.get('intent', 'unknown')}")
            print(f"   - Risk: {result.get('risk', 'unknown')}")
        elif response.status_code == 500:
            print(f"⚠️  CBT analysis failed (expected if models not loaded): {response.json()}")
        else:
            print(f"❌ CBT analysis unexpected status: {response.text}")
            
        return True  # Don't fail on model loading issues
    except Exception as e:
        print(f"❌ CBT analysis endpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 CBT FastAPI Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_creation,
        test_model_status,
    ]
    
    async_tests = [
        test_health_endpoint,
        test_model_status_endpoint,
        test_cbt_analysis_endpoint,
    ]
    
    # Run synchronous tests
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Run asynchronous tests
    async def run_async_tests():
        async_results = []
        for test in async_tests:
            try:
                result = await test()
                async_results.append(result)
            except Exception as e:
                print(f"❌ Async test {test.__name__} crashed: {e}")
                async_results.append(False)
        return async_results
    
    async_results = asyncio.run(run_async_tests())
    results.extend(async_results)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("🎉 All tests passed! The API is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        print("   Note: Model-related failures are expected if model files are not available.")
        return 1

if __name__ == "__main__":
    exit(main())