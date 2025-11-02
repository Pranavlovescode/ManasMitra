#!/usr/bin/env python3
"""
Test Next.js Journal API endpoints to verify they work with the transformed data
"""
import requests
import json

# Test data
journal_data = {
    "title": "Test Journal Entry",
    "content": "I'm feeling really anxious about my upcoming presentation. I keep thinking that everyone will judge me harshly and I'll embarrass myself. I can't stop worrying about all the things that could go wrong.",
    "mood": "sad",
    "selectedPrompt": "What emotions are you experiencing?"
}

print("🧪 Testing Next.js Journal API...")
print("=" * 50)

# Test POST - Create journal entry
print("📝 Testing POST /api/journal (Create Entry)...")
try:
    response = requests.post(
        "http://localhost:3000/api/journal",
        json=journal_data,
        headers={
            "Content-Type": "application/json",
            # Note: In real app, this would include Clerk auth headers
        }
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 201:
        result = response.json()
        print("✅ Journal created successfully!")
        journal_id = result.get('_id')
        
        if result.get('analysis'):
            print("🧠 Analysis data found:")
            analysis = result['analysis']
            
            if analysis.get('contentAnalysis'):
                ca = analysis['contentAnalysis']
                print(f"  - Emotion: {ca.get('emotion')} (score: {ca.get('emotionScore')})")
                print(f"  - Intent: {ca.get('intent')} (score: {ca.get('intentScore')})")
                print(f"  - Risk: {ca.get('risk')} (score: {ca.get('riskScore')})")
                print(f"  - Distortions: {ca.get('distortions')}")
                print(f"  - Reframes: {len(ca.get('reframes', []))} suggestions")
                print(f"  - Behavioral Suggestions: {len(ca.get('behavioralSuggestions', []))} items")
            
            print(f"  - Overall Sentiment: {analysis.get('overallSentiment')}")
            print(f"  - Key Themes: {analysis.get('keyThemes')}")
            print(f"  - Therapeutic Insights: {len(analysis.get('therapeuticInsights', []))} insights")
            print(f"  - Recommendations: {len(analysis.get('recommendations', []))} recommendations")
        else:
            print("❌ No analysis data found in response")
        
        print(f"📄 Journal ID: {journal_id}")
    
    elif response.status_code == 401:
        print("🔒 Authentication required (expected in test environment)")
        print("This is normal - the API requires Clerk authentication")
    
    else:
        print(f"❌ Error: {response.text}")

except requests.exceptions.ConnectionError:
    print("❌ Next.js server not running on localhost:3000")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("💡 Note: To test with authentication, run from Next.js frontend")
print("   The API requires Clerk authentication headers")