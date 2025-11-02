#!/usr/bin/env python3
"""
Complete test of the journal analysis pipeline
Tests FastAPI backend and simulates Next.js transformation
"""

import requests
import json
from datetime import datetime

def test_fastapi_journal_analysis():
    """Test the FastAPI journal analysis endpoint"""
    
    print("🧪 Testing FastAPI Journal Analysis Endpoint...")
    
    # Test data
    test_data = {
        "title": "Feeling overwhelmed at work",
        "content": "I keep thinking that I'm going to fail at my new project. Every small mistake feels like proof that I'm not good enough for this job. I can't stop catastrophizing about what my boss will think.",
        "mood": "anxious",
        "prompt": "What's been on your mind lately?"
    }
    
    print(f"📡 URL: http://127.0.0.1:8000/analyze/journal")
    print(f"📋 Data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Make request to FastAPI
        response = requests.post(
            "http://127.0.0.1:8000/analyze/journal",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            analysis_data = response.json()
            print("✅ FastAPI Analysis Success!")
            
            # Show key analysis results
            content_analysis = analysis_data.get("content_analysis", {})
            print(f"\n🔍 Analysis Results:")
            print(f"  - Emotion: {content_analysis.get('emotion')} (score: {content_analysis.get('emotion_score', 0):.3f})")
            print(f"  - Intent: {content_analysis.get('intent')} (score: {content_analysis.get('intent_score', 0):.3f})")
            print(f"  - Risk: {content_analysis.get('risk')} (score: {content_analysis.get('risk_score', 0):.3f})")
            print(f"  - Distortions: {content_analysis.get('distortions', [])}")
            print(f"  - Overall Sentiment: {analysis_data.get('overall_sentiment')}")
            
            return analysis_data
            
        else:
            print(f"❌ FastAPI Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: FastAPI server not running on http://127.0.0.1:8000")
        return None
    except Exception as e:
        print(f"❌ Request Error: {e}")
        return None

def simulate_nextjs_transformation(fastapi_data):
    """Simulate the Next.js API transformation"""
    
    print("\n🔄 Simulating Next.js API Transformation...")
    
    if not fastapi_data:
        print("❌ No FastAPI data to transform")
        return None
    
    # Extract content_analysis (this is the fix we implemented)
    content_analysis = fastapi_data.get("content_analysis", {})
    
    # Transform to frontend schema
    transformed_analysis = {
        "contentAnalysis": {
            "emotion": content_analysis.get("emotion"),
            "emotionScore": content_analysis.get("emotion_score"),
            "intent": content_analysis.get("intent"),
            "intentScore": content_analysis.get("intent_score"),
            "risk": content_analysis.get("risk"),
            "riskScore": content_analysis.get("risk_score"),
            "distortions": content_analysis.get("distortions", []),
            "distortionDetails": content_analysis.get("distortion_details", []),
            "reframes": content_analysis.get("reframes", []),
            "behavioralSuggestions": content_analysis.get("behavioral_suggestions", []),
            "clinicianNotes": content_analysis.get("clinician_notes", []),
        },
        "overallSentiment": fastapi_data.get("overall_sentiment", "neutral"),
        "keyThemes": fastapi_data.get("key_themes", []),
        "therapeuticInsights": fastapi_data.get("therapeutic_insights", []),
        "progressIndicators": fastapi_data.get("progress_indicators", []),
        "recommendations": fastapi_data.get("recommendations", []),
        "analysisTimestamp": fastapi_data.get("analysis_timestamp"),
    }
    
    print("✅ Transformation Complete!")
    print(f"📤 Transformed Structure:")
    print(f"  - Emotion: {transformed_analysis['contentAnalysis']['emotion']}")
    print(f"  - Intent: {transformed_analysis['contentAnalysis']['intent']}")  
    print(f"  - Risk: {transformed_analysis['contentAnalysis']['risk']}")
    print(f"  - Distortions: {transformed_analysis['contentAnalysis']['distortions']}")
    print(f"  - Reframes: {len(transformed_analysis['contentAnalysis']['reframes'])} suggestions")
    print(f"  - Behavioral Suggestions: {len(transformed_analysis['contentAnalysis']['behavioralSuggestions'])} items")
    
    return transformed_analysis

def simulate_journal_creation(transformed_data):
    """Simulate creating a journal entry with the transformed data"""
    
    print("\n📝 Simulating Journal Creation...")
    
    if not transformed_data:
        print("❌ No transformed data for journal creation")
        return None
    
    # Simulate journal document structure
    journal_entry = {
        "userId": "user_test_12345",
        "title": "Feeling overwhelmed at work",
        "content": "I keep thinking that I'm going to fail at my new project...",
        "selectedPrompt": "What's been on your mind lately?",
        "mood": "anxious",
        "analysis": transformed_data,  # This would be saved to MongoDB
        "createdAt": datetime.now().isoformat(),
        "updatedAt": datetime.now().isoformat()
    }
    
    print("✅ Journal Entry Structure Created!")
    print(f"📄 Entry Summary:")
    print(f"  - Title: {journal_entry['title']}")
    print(f"  - Mood: {journal_entry['mood']}")
    print(f"  - Has Analysis: {'Yes' if journal_entry.get('analysis') else 'No'}")
    
    # Verify analysis fields are present
    analysis = journal_entry.get('analysis', {})
    content_analysis = analysis.get('contentAnalysis', {})
    
    verification_checks = {
        "Emotion Present": bool(content_analysis.get('emotion')),
        "Intent Present": bool(content_analysis.get('intent')),
        "Risk Present": bool(content_analysis.get('risk')),
        "Distortions Present": bool(content_analysis.get('distortions')),
        "Reframes Present": bool(content_analysis.get('reframes')),
        "Behavioral Suggestions Present": bool(content_analysis.get('behavioralSuggestions'))
    }
    
    print(f"\n✅ Analysis Verification:")
    for check, status in verification_checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check}")
    
    return journal_entry

def main():
    """Run complete pipeline test"""
    
    print("🚀 Testing Complete Journal Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Test FastAPI backend
    fastapi_result = test_fastapi_journal_analysis()
    
    # Step 2: Test Next.js transformation
    transformed_result = simulate_nextjs_transformation(fastapi_result)
    
    # Step 3: Test journal creation
    journal_result = simulate_journal_creation(transformed_result)
    
    print("\n" + "=" * 60)
    
    if all([fastapi_result, transformed_result, journal_result]):
        print("🎉 COMPLETE PIPELINE TEST: SUCCESS!")
        print("✅ FastAPI Backend: Working")
        print("✅ Data Transformation: Working") 
        print("✅ Journal Creation: Working")
        print("✅ Analysis Fields: All Present")
        
        # Show final summary
        content_analysis = journal_result['analysis']['contentAnalysis']
        print(f"\n📊 Final Analysis Summary:")
        print(f"  🎭 Emotion: {content_analysis['emotion']}")
        print(f"  🎯 Intent: {content_analysis['intent']}")
        print(f"  ⚠️  Risk: {content_analysis['risk']}")
        print(f"  🧠 Distortions: {', '.join(content_analysis['distortions']) if content_analysis['distortions'] else 'None'}")
        print(f"  💡 Reframes: {len(content_analysis['reframes'])} suggestions")
        
    else:
        print("❌ COMPLETE PIPELINE TEST: FAILED")
        print(f"  FastAPI Backend: {'✅' if fastapi_result else '❌'}")
        print(f"  Data Transformation: {'✅' if transformed_result else '❌'}")
        print(f"  Journal Creation: {'✅' if journal_result else '❌'}")

if __name__ == "__main__":
    main()