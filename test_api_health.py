#!/usr/bin/env python3
"""
Test FastAPI health and model status
"""

import requests
import json

def check_api_health():
    """Check if FastAPI is running and healthy"""
    
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        print(f"📡 Health Check Status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy!")
            print(f"📊 Health Data: {json.dumps(health_data, indent=2)}")
        else:
            print(f"❌ API Health Issue: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI server not running on http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
        
    return True

def check_model_status():
    """Check model loading status"""
    
    try:
        response = requests.get("http://127.0.0.1:8000/models/status")
        print(f"🔧 Model Status Check: {response.status_code}")
        
        if response.status_code == 200:
            status_data = response.json()
            print("✅ Model Status Retrieved!")
            print(f"📊 Model Status: {json.dumps(status_data, indent=2)}")
            
            # Check individual models
            models = ["intent", "emotion", "cognitive", "risk"]
            for model in models:
                status = status_data.get(model, False)
                icon = "✅" if status else "❌"
                print(f"  {icon} {model.capitalize()} Model: {'Loaded' if status else 'Not Loaded'}")
                
        else:
            print(f"❌ Model Status Issue: {response.text}")
            
    except Exception as e:
        print(f"❌ Model status error: {e}")

def test_simple_cbt_analysis():
    """Test basic CBT analysis endpoint"""
    
    try:
        test_data = {"text": "I'm feeling anxious and worried"}
        response = requests.post(
            "http://127.0.0.1:8000/analyze/cbt",
            json=test_data
        )
        
        print(f"🧠 CBT Analysis Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ CBT Analysis Working!")
            print(f"  - Emotion: {result.get('emotion')}")
            print(f"  - Intent: {result.get('intent')}")
            print(f"  - Risk: {result.get('risk')}")
        else:
            print(f"❌ CBT Analysis Error: {response.text}")
            
    except Exception as e:
        print(f"❌ CBT Analysis test error: {e}")

if __name__ == "__main__":
    print("🔍 Diagnosing FastAPI and CBT Models...")
    print("=" * 50)
    
    if check_api_health():
        check_model_status()
        test_simple_cbt_analysis()
    else:
        print("⚠️ FastAPI server needs to be started first")
        print("💡 Run: python main.py")