# CBT Mental Health Analysis API

A FastAPI-based REST API for Cognitive Behavioral Therapy (CBT) analysis using AI models. This API exposes the functionality from the integrated CBT Streamlit application as web services for integration with websites and applications.

## Features

- **Text Analysis**: Comprehensive CBT analysis including emotion detection, intent classification, risk assessment, and cognitive distortion detection
- **Audio Processing**: Speech-to-text transcription and voice emotion detection
- **Multiple AI Models**: Intent classification, emotion analysis, suicide risk detection, and cognitive distortion identification
- **RESTful API**: Clean, documented endpoints with automatic OpenAPI schema generation
- **Audio Support**: Upload audio files for transcription and voice-based emotion detection
- **Error Handling**: Comprehensive error handling with appropriate HTTP status codes

## Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r fastapi_requirements.txt
```

2. **Ensure model files are available:**
   - Copy all model directories from the original Streamlit app
   - Ensure `cbt.py` and model modules are in the same directory
   - Models should be in subdirectories: `intent_classification/`, `emotion_classifier/`, `risk-detection/`, `cognitive_distortion/`, `voicebasedemotion/`

3. **Run the API:**
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### Health & Status

#### GET `/health`
Check API health and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-30T12:00:00Z",
  "models_loaded": {
    "intent": true,
    "emotion": true,
    "cognitive": true,
    "risk": true
  }
}
```

#### GET `/models/status`
Get detailed model status information.

**Response:**
```json
{
  "intent": true,
  "emotion": true,
  "cognitive": true,
  "risk": true,
  "asr_available": true,
  "voice_emotion_available": true,
  "status_message": "4/4 core models loaded, ASR available, voice emotion available"
}
```

### Text Analysis

#### POST `/analyze/cbt`
Comprehensive CBT analysis of text input.

**Request:**
```json
{
  "text": "I always mess everything up and I'm never going to succeed at anything",
  "include_voice_emotion": false
}
```

**Response:**
```json
{
  "emotion": "sadness",
  "emotion_score": 0.85,
  "original_emotion": "sadness",
  "emotion_source": "text",
  "intent": "seek_help",
  "intent_score": 0.72,
  "risk": "moderate",
  "risk_score": 0.45,
  "escalation": false,
  "distortions": ["all_or_nothing", "overgeneralization"],
  "distortion_details": [
    {
      "distortion_type": "all-or-nothing thinking",
      "confidence": 0.89,
      "emoji": "⚫⚪",
      "explanation": "Seeing things in absolute, black and white categories...",
      "reframing_suggestion": "Look for the grey areas and nuances..."
    }
  ],
  "reframes": ["Consider specific examples where you have succeeded..."],
  "behavioral_suggestions": ["Keep a success journal..."],
  "clinician_notes": ["High confidence all-or-nothing thinking detected"],
  "user_facing": "It sounds like you're being really hard on yourself...",
  "analysis_timestamp": "2025-10-30T12:00:00Z"
}
```

#### POST `/analyze/emotion`
Emotion analysis only.

**Request:**
```json
{
  "text": "I feel overwhelmed and anxious about everything"
}
```

**Response:**
```json
{
  "emotion": "anxiety",
  "score": 0.78,
  "original_emotion": "anxiety"
}
```

#### POST `/analyze/intent`
Intent classification only.

**Request:**
```json
{
  "text": "Can you help me understand why I feel this way?"
}
```

**Response:**
```json
{
  "intent": "seek_help",
  "confidence": 0.85
}
```

#### POST `/analyze/risk`
Risk assessment only.

**Request:**
```json
{
  "text": "Sometimes I wonder if life is worth living"
}
```

**Response:**
```json
{
  "level": "moderate",
  "score": 0.55,
  "category": "🟠 Moderate Risk"
}
```

#### POST `/analyze/distortions`
Cognitive distortion detection only.

**Request:**
```json
{
  "text": "Everyone thinks I'm stupid and they're probably right"
}
```

**Response:**
```json
{
  "detected_distortions": ["mind_reading", "overgeneralization"],
  "distortion_details": [
    {
      "distortion_type": "mind reading",
      "confidence": 0.92,
      "emoji": "👁️",
      "explanation": "Assuming you know what others are thinking...",
      "reframing_suggestion": "Check your assumptions by asking for clarification..."
    }
  ]
}
```

### Audio Processing

#### POST `/audio/transcribe`
Transcribe audio to text using Whisper.

**Request:**
- Upload audio file (WAV, MP3, M4A, OGG, FLAC)

**Response:**
```json
{
  "transcription": "I feel really overwhelmed with everything right now",
  "success": true,
  "error": null
}
```

#### POST `/audio/emotion`
Detect emotion from voice audio.

**Request:**
- Upload audio file

**Response:**
```json
{
  "emotion": "sad",
  "success": true,
  "error": null
}
```

#### POST `/audio/analyze`
Complete audio analysis (transcription + voice emotion + CBT analysis).

**Request:**
- Upload audio file

**Response:**
```json
{
  "transcription": "I always mess everything up",
  "voice_emotion": "sad",
  "cbt_analysis": {
    "emotion": "sad",
    "emotion_source": "voice",
    // ... full CBT analysis response
  }
}
```

## Integration Examples

### JavaScript/Web Integration

```javascript
// Text analysis
async function analyzeCBT(text) {
  const response = await fetch('http://localhost:8000/analyze/cbt', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: text })
  });
  
  if (!response.ok) {
    throw new Error('Analysis failed');
  }
  
  return await response.json();
}

// Audio analysis
async function analyzeAudio(audioFile) {
  const formData = new FormData();
  formData.append('file', audioFile);
  
  const response = await fetch('http://localhost:8000/audio/analyze', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error('Audio analysis failed');
  }
  
  return await response.json();
}

// Usage examples
analyzeCBT("I feel like nothing I do matters")
  .then(result => {
    console.log('Emotion:', result.emotion);
    console.log('Risk Level:', result.risk);
    console.log('Distortions:', result.distortions);
    console.log('Suggestions:', result.reframes);
  })
  .catch(error => console.error('Error:', error));
```

### Python Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def analyze_text(text):
    """Analyze text using CBT API"""
    response = requests.post(
        f"{BASE_URL}/analyze/cbt",
        json={"text": text}
    )
    response.raise_for_status()
    return response.json()

def analyze_audio(audio_file_path):
    """Analyze audio file"""
    with open(audio_file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{BASE_URL}/audio/analyze",
            files=files
        )
    response.raise_for_status()
    return response.json()

# Usage
result = analyze_text("I always fail at everything I try")
print(f"Emotion: {result['emotion']}")
print(f"Risk Level: {result['risk']}")
print(f"Distortions: {result['distortions']}")
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Text analysis
curl -X POST "http://localhost:8000/analyze/cbt" \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel overwhelmed and anxious"}'

# Audio transcription
curl -X POST "http://localhost:8000/audio/transcribe" \
     -F "file=@audio.wav"

# Complete audio analysis
curl -X POST "http://localhost:8000/audio/analyze" \
     -F "file=@audio.wav"
```

## Configuration

### Environment Variables

```bash
# Optional: Configure logging level
export LOG_LEVEL=INFO

# Optional: Configure CORS origins for production
export CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
```

### Production Deployment

For production deployment, consider:

1. **Security**: Update CORS settings, add authentication if needed
2. **Performance**: Use multiple workers with Gunicorn
3. **Monitoring**: Add logging and monitoring
4. **Resources**: Ensure adequate memory/CPU for model loading

```bash
# Production deployment with Gunicorn
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Model Requirements

Ensure these directories and files exist:

```
project_root/
├── main.py
├── cbt_models.py
├── cbt.py
├── intent_classification/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── emotion_classifier/
│   ├── emotion_model.pth
│   ├── vocab.txt
│   └── emotion_model.py
├── risk-detection/
│   └── suicide_model.pth
├── cognitive_distortion/
│   └── cognitive_distortion_model.py
└── voicebasedemotion/
    └── speechemotion.py
```

## Error Handling

The API provides structured error responses:

```json
{
  "error": "Model not available",
  "detail": "Cognitive distortion model not loaded",
  "timestamp": "2025-10-30T12:00:00Z"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `422`: Validation error
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

## Limitations

- Models run on CPU by default (can be configured for GPU)
- Audio files should be under 30 seconds for optimal performance
- Text input limited to 5000 characters
- Some models may not be available if files are missing

## Support

For issues or questions:
1. Check the model loading status at `/models/status`
2. Review the API documentation at `/docs`
3. Check server logs for error details
4. Ensure all model files are properly installed