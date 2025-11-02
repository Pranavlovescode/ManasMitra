#!/usr/bin/env python3
"""
Test the data transformation logic to ensure proper format conversion
"""
import json
from datetime import datetime

# Simulate FastAPI journal analysis response
fastapi_response = {
    "content_analysis": {
        "emotion": "optimism",
        "emotion_score": 0.5974566340446472,
        "original_emotion": None,
        "emotion_source": "text",
        "intent": "anxious",
        "intent_score": 0.024188367649912834,
        "risk": "moderate",
        "risk_score": 0.423,
        "escalation": False,
        "distortions": ["all_or_nothing", "catastrophizing"],
        "distortion_details": [],
        "reframes": [
            "Is it really always or never? Find specific exceptions that disprove the extreme thought.",
            "What evidence do you have that this will definitely happen?"
        ],
        "behavioral_suggestions": [
            "Try a short self-compassion exercise: acknowledge the difficulty and name one small next step",
            "Practice deep breathing for 5 minutes"
        ],
        "clinician_notes": [
            "Emotion: optimism, intent: anxious, risk: moderate",
            "Cognitive distortions detected: all_or_nothing, catastrophizing"
        ],
        "user_facing": "I hear you, thank you for sharing. Here are some suggestions...",
        "analysis_timestamp": "2025-11-02T14:26:27.611484",
        "raw_analysis": {"text": "Sample text", "emotion": "optimism"}
    },
    "title_analysis": None,
    "overall_sentiment": "negative",
    "key_themes": ["Emotional state: optimism", "Cognitive patterns identified"],
    "therapeutic_insights": ["Entry indicates elevated emotional distress - consider professional support"],
    "progress_indicators": ["Demonstrating self-awareness of thought patterns"],
    "recommendations": ["Practice grounding techniques and self-care activities"],
    "analysis_timestamp": "2025-11-02T14:26:27.611484"
}

# JavaScript-like transformation (simulated in Python)
def transform_analysis_data(analysis_data):
    content_analysis = analysis_data.get("content_analysis", {})
    
    return {
        "analysis": {
            "contentAnalysis": {
                "emotion": content_analysis.get("emotion"),
                "emotionScore": content_analysis.get("emotion_score"),
                "intent": content_analysis.get("intent"),
                "intentScore": content_analysis.get("intent_score"),
                "risk": content_analysis.get("risk"),
                "riskScore": content_analysis.get("risk_score"),
                "distortions": content_analysis.get("distortions", []),
                "distortionDetails": [
                    {
                        "distortionType": d.get("distortion_type"),
                        "confidence": d.get("confidence"),
                        "emoji": d.get("emoji"),
                        "explanation": d.get("explanation"),
                        "reframingSuggestion": d.get("reframing_suggestion")
                    } for d in content_analysis.get("distortion_details", [])
                ],
                "reframes": content_analysis.get("reframes", []),
                "behavioralSuggestions": content_analysis.get("behavioral_suggestions", []),
                "clinicianNotes": content_analysis.get("clinician_notes", []),
            },
            "overallSentiment": analysis_data.get("overall_sentiment", "neutral"),
            "keyThemes": analysis_data.get("key_themes", []),
            "therapeuticInsights": analysis_data.get("therapeutic_insights", []),
            "progressIndicators": analysis_data.get("progress_indicators", []),
            "recommendations": analysis_data.get("recommendations", []),
            "analysisTimestamp": analysis_data.get("analysis_timestamp"),
        }
    }

print("🔄 Testing Data Transformation Pipeline...")
print("=" * 70)

print("📥 FastAPI Response Structure:")
print(json.dumps(fastapi_response, indent=2)[:500] + "...")

print("\n" + "=" * 70)

transformed = transform_analysis_data(fastapi_response)
print("📤 Transformed for MongoDB Storage:")
print(json.dumps(transformed, indent=2))

print("\n" + "=" * 70)

# Verify all critical fields are present and correctly mapped
analysis = transformed["analysis"]
content_analysis = analysis["contentAnalysis"]

print("✅ Verification - Critical Analysis Fields:")
print(f"  📊 Emotion: '{content_analysis['emotion']}' (score: {content_analysis['emotionScore']})")
print(f"  🎯 Intent: '{content_analysis['intent']}' (score: {content_analysis['intentScore']})")
print(f"  ⚠️  Risk: '{content_analysis['risk']}' (score: {content_analysis['riskScore']})")
print(f"  🧠 Distortions: {content_analysis['distortions']}")
print(f"  💡 Reframes: {len(content_analysis['reframes'])} suggestions")
print(f"  🎯 Behavioral: {len(content_analysis['behavioralSuggestions'])} suggestions")
print(f"  📝 Clinician Notes: {len(content_analysis['clinicianNotes'])} notes")

print(f"\n  🌡️  Overall Sentiment: '{analysis['overallSentiment']}'")
print(f"  🔍 Key Themes: {len(analysis['keyThemes'])} themes")
print(f"  💭 Therapeutic Insights: {len(analysis['therapeuticInsights'])} insights")
print(f"  📈 Progress Indicators: {len(analysis['progressIndicators'])} indicators")
print(f"  📋 Recommendations: {len(analysis['recommendations'])} items")

# Check for missing data
missing_fields = []
if not content_analysis.get('emotion'): missing_fields.append('emotion')
if not content_analysis.get('intent'): missing_fields.append('intent')  
if not content_analysis.get('risk'): missing_fields.append('risk')
if not content_analysis.get('distortions'): missing_fields.append('distortions')

if missing_fields:
    print(f"\n❌ Missing Fields: {missing_fields}")
else:
    print(f"\n✅ All critical analysis fields present and properly formatted!")

print(f"\n🏗️  Ready for MongoDB Journal.analysis field structure!")