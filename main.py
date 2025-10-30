"""
FastAPI CBT Mental Health Analysis API
Exposes CBT models and analysis capabilities as REST endpoints
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import traceback
from datetime import datetime
import os

# Import our CBT models module
from cbt_models import (
    initialize_cbt_engine,
    get_model_status,
    transcribe_audio_locally,
    detect_voice_emotion_from_bytes,
    load_asr_pipeline,
    load_voice_emotion_components
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CBT Mental Health Analysis API",
    description="REST API for Cognitive Behavioral Therapy analysis using AI models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    include_voice_emotion: Optional[bool] = Field(False, description="Whether this text came from voice input")

class EmotionPrediction(BaseModel):
    emotion: str
    score: float
    original_emotion: Optional[str] = None

class IntentPrediction(BaseModel):
    intent: str
    confidence: float

class RiskAssessment(BaseModel):
    level: str
    score: float
    category: str

class CognitiveDistortion(BaseModel):
    distortion_type: str
    confidence: float
    emoji: str
    explanation: Optional[str] = None
    reframing_suggestion: Optional[str] = None

class CBTAnalysisResponse(BaseModel):
    emotion: Optional[str] = None
    emotion_score: Optional[float] = None
    original_emotion: Optional[str] = None
    emotion_source: str = "text"
    intent: Optional[str] = None
    intent_score: Optional[float] = None
    risk: Optional[str] = None
    risk_score: Optional[float] = None
    escalation: Optional[bool] = None
    distortions: List[str] = []
    distortion_details: List[CognitiveDistortion] = []
    reframes: List[str] = []
    behavioral_suggestions: List[str] = []
    clinician_notes: List[str] = []
    user_facing: Optional[str] = None
    analysis_timestamp: datetime
    raw_analysis: Optional[Dict[str, Any]] = None

class AudioTranscriptionResponse(BaseModel):
    transcription: Optional[str]
    success: bool
    error: Optional[str] = None

class VoiceEmotionResponse(BaseModel):
    emotion: Optional[str]
    success: bool
    error: Optional[str] = None

class ModelStatusResponse(BaseModel):
    intent: bool
    emotion: bool
    cognitive: bool
    risk: bool
    asr_available: bool
    voice_emotion_available: bool
    status_message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime

# Dependency to get CBT engine
def get_cbt_engine():
    try:
        return initialize_cbt_engine()
    except Exception as e:
        logger.error(f"Failed to initialize CBT engine: {e}")
        raise HTTPException(status_code=500, detail="CBT engine initialization failed")

# Root endpoint with HTML response
@app.get("/", response_class=HTMLResponse)
async def root():
    """API root endpoint with user-friendly HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CBT Mental Health Analysis API</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .header { text-align: center; color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { display: inline-block; padding: 3px 8px; border-radius: 3px; font-weight: bold; }
            .get { background: #28a745; color: white; }
            .post { background: #007bff; color: white; }
            .status { margin: 20px 0; }
            .success { color: #28a745; }
            .warning { color: #ffc107; }
            .error { color: #dc3545; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 CBT Mental Health Analysis API</h1>
            <p>REST API for Cognitive Behavioral Therapy analysis using AI models</p>
        </div>
        
        <div class="status">
            <h2>📊 Quick Links</h2>
            <p>📖 <a href="/docs" target="_blank">Interactive API Documentation (Swagger UI)</a></p>
            <p>📝 <a href="/redoc" target="_blank">Alternative Documentation (ReDoc)</a></p>
            <p>💚 <a href="/health" target="_blank">Health Check</a></p>
            <p>🔧 <a href="/models/status" target="_blank">Model Status</a></p>
        </div>

        <h2>🚀 Available Endpoints</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/analyze/cbt</strong>
            <p>Comprehensive CBT analysis including emotion, intent, risk assessment, and cognitive distortion detection</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/analyze/emotion</strong>
            <p>Emotion detection from text input</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/analyze/intent</strong>
            <p>Intent classification from text input</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/analyze/risk</strong>
            <p>Suicide risk assessment from text input</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/analyze/distortions</strong>
            <p>Cognitive distortion detection from text input</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/audio/transcribe</strong>
            <p>Audio transcription using Whisper ASR</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/audio/emotion</strong>
            <p>Voice emotion detection from audio files</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/audio/analyze</strong>
            <p>Complete audio analysis: transcription + voice emotion + CBT analysis</p>
        </div>

        <h2>💡 Example Usage</h2>
        <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
# Text Analysis
curl -X POST "http://localhost:8000/analyze/cbt" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "I feel overwhelmed and anxious about everything"}'

# Audio Analysis  
curl -X POST "http://localhost:8000/audio/analyze" \\
     -F "file=@audio.wav"
        </pre>

        <div style="text-align: center; margin-top: 50px; color: #6c757d;">
            <p>🔬 CBT Mental Health Analysis API v1.0.0</p>
            <p>Built with FastAPI • Powered by AI Models</p>
        </div>
    </body>
    </html>
    """
    return html_content

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    try:
        models_status = get_model_status()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            models_loaded=models_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Model status endpoint
@app.get("/models/status", response_model=ModelStatusResponse)
async def get_models_status():
    """Get detailed status of all loaded models"""
    try:
        models_status = get_model_status()
        
        # Check additional components
        asr_available = load_asr_pipeline() is not None
        voice_components = load_voice_emotion_components()
        voice_emotion_available = all(comp is not None for comp in voice_components)
        
        loaded_count = sum(models_status.values())
        total_count = len(models_status)
        
        status_message = f"{loaded_count}/{total_count} core models loaded"
        if asr_available:
            status_message += ", ASR available"
        if voice_emotion_available:
            status_message += ", voice emotion available"
            
        return ModelStatusResponse(
            **models_status,
            asr_available=asr_available,
            voice_emotion_available=voice_emotion_available,
            status_message=status_message
        )
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
        raise HTTPException(status_code=500, detail="Model status check failed")

# CBT Analysis endpoint
@app.post("/analyze/cbt", response_model=CBTAnalysisResponse)
async def analyze_cbt(request: TextAnalysisRequest, cbt_engine = Depends(get_cbt_engine)):
    """
    Perform comprehensive CBT analysis on text input
    """
    try:
        # Perform CBT analysis
        result = cbt_engine.analyze(request.text)
        
        # Process cognitive distortions with detailed information
        distortion_details = []
        cognitive_wrapper = cbt_engine.config.get("cognitive_wrapper")
        
        if cognitive_wrapper and "distortion_details" in result:
            for dist in result.get("distortion_details", []):
                distortion_details.append(CognitiveDistortion(
                    distortion_type=dist["distortion_type"],
                    confidence=dist["confidence"],
                    emoji=dist["emoji"],
                    explanation=cognitive_wrapper.get_distortion_explanation(dist["distortion_type"]),
                    reframing_suggestion=cognitive_wrapper.get_reframing_suggestion(dist["distortion_type"])
                ))
        
        return CBTAnalysisResponse(
            emotion=result.get('emotion'),
            emotion_score=result.get('emotion_score'),
            original_emotion=result.get('original_emotion'),
            emotion_source="voice" if request.include_voice_emotion else "text",
            intent=result.get('intent'),
            intent_score=result.get('intent_score'),
            risk=result.get('risk'),
            risk_score=result.get('risk_score'),
            escalation=result.get('escalation'),
            distortions=result.get('distortions', []),
            distortion_details=distortion_details,
            reframes=result.get('reframes', []),
            behavioral_suggestions=result.get('behavioral_suggestions', []),
            clinician_notes=result.get('clinician_notes', []),
            user_facing=result.get('user_facing'),
            analysis_timestamp=datetime.now(),
            raw_analysis=result
        )
        
    except Exception as e:
        logger.error(f"CBT analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"CBT analysis failed: {str(e)}")

# Emotion analysis endpoint
@app.post("/analyze/emotion")
async def analyze_emotion(request: TextAnalysisRequest, cbt_engine = Depends(get_cbt_engine)):
    """
    Analyze emotion from text input
    """
    try:
        emotion_wrapper = cbt_engine.emotion_model
        result = emotion_wrapper.predict(request.text)
        
        return EmotionPrediction(
            emotion=result.get('emotion', 'unknown'),
            score=result.get('score', 0.0),
            original_emotion=result.get('original_emotion')
        )
        
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")

# Intent classification endpoint
@app.post("/analyze/intent")
async def analyze_intent(request: TextAnalysisRequest, cbt_engine = Depends(get_cbt_engine)):
    """
    Classify intent from text input
    """
    try:
        intent_wrapper = cbt_engine.intent_model
        result = intent_wrapper.predict(request.text)
        
        return IntentPrediction(
            intent=result.get('intent', 'unknown'),
            confidence=result.get('confidence', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Intent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Intent analysis failed: {str(e)}")

# Risk assessment endpoint
@app.post("/analyze/risk")
async def analyze_risk(request: TextAnalysisRequest, cbt_engine = Depends(get_cbt_engine)):
    """
    Assess suicide risk from text input
    """
    try:
        risk_wrapper = cbt_engine.risk_model
        result = risk_wrapper.predict(request.text)
        
        # Get risk category for display
        risk_score = result.get('score', 0.0)
        if risk_score < 0.20:
            category = "🟢 Minimal Risk"
        elif risk_score < 0.40:
            category = "🟡 Low Risk"
        elif risk_score < 0.60:
            category = "🟠 Moderate Risk"
        elif risk_score < 0.80:
            category = "🔴 High Risk"
        else:
            category = "⚠️ Severe Risk"
        
        return RiskAssessment(
            level=result.get('level', 'unknown'),
            score=risk_score,
            category=category
        )
        
    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")

# Cognitive distortion analysis endpoint
@app.post("/analyze/distortions")
async def analyze_distortions(request: TextAnalysisRequest, cbt_engine = Depends(get_cbt_engine)):
    """
    Detect cognitive distortions in text input
    """
    try:
        cognitive_wrapper = cbt_engine.config.get("cognitive_wrapper")
        if not cognitive_wrapper or not cognitive_wrapper.model:
            raise HTTPException(status_code=503, detail="Cognitive distortion model not available")
        
        result = cognitive_wrapper.analyze(request.text)
        
        distortion_details = []
        for dist in result.get("distortion_details", []):
            distortion_details.append(CognitiveDistortion(
                distortion_type=dist["distortion_type"],
                confidence=dist["confidence"],
                emoji=dist["emoji"],
                explanation=cognitive_wrapper.get_distortion_explanation(dist["distortion_type"]),
                reframing_suggestion=cognitive_wrapper.get_reframing_suggestion(dist["distortion_type"])
            ))
        
        return {
            "detected_distortions": result.get("detected_distortions", []),
            "distortion_details": distortion_details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Distortion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Distortion analysis failed: {str(e)}")

# Audio transcription endpoint
@app.post("/audio/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper
    """
    try:
        # Check file type
        if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3", "audio/m4a", "audio/ogg", "audio/flac"]:
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Read file content
        content = await file.read()
        
        # Transcribe
        transcription = transcribe_audio_locally(content, file.filename)
        
        if transcription is None:
            return AudioTranscriptionResponse(
                transcription=None,
                success=False,
                error="Transcription failed - ASR model not available or audio processing error"
            )
        
        return AudioTranscriptionResponse(
            transcription=transcription,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return AudioTranscriptionResponse(
            transcription=None,
            success=False,
            error=f"Transcription failed: {str(e)}"
        )

# Voice emotion detection endpoint
@app.post("/audio/emotion", response_model=VoiceEmotionResponse)
async def detect_voice_emotion(file: UploadFile = File(...)):
    """
    Detect emotion from voice audio
    """
    try:
        # Check file type
        if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3", "audio/m4a", "audio/ogg", "audio/flac"]:
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Read file content
        content = await file.read()
        
        # Detect emotion
        emotion = detect_voice_emotion_from_bytes(content, file.filename)
        
        if emotion is None:
            return VoiceEmotionResponse(
                emotion=None,
                success=False,
                error="Voice emotion detection failed - model not available or audio processing error"
            )
        
        return VoiceEmotionResponse(
            emotion=emotion,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice emotion detection failed: {e}")
        return VoiceEmotionResponse(
            emotion=None,
            success=False,
            error=f"Voice emotion detection failed: {str(e)}"
        )

# Combined audio analysis endpoint
@app.post("/audio/analyze")
async def analyze_audio_complete(file: UploadFile = File(...), cbt_engine = Depends(get_cbt_engine)):
    """
    Complete audio analysis: transcription + voice emotion + CBT analysis
    """
    try:
        # Check file type
        if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3", "audio/m4a", "audio/ogg", "audio/flac"]:
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Read file content
        content = await file.read()
        
        # Transcribe audio
        transcription = transcribe_audio_locally(content, file.filename)
        if not transcription:
            raise HTTPException(status_code=422, detail="Could not transcribe audio")
        
        # Detect voice emotion
        voice_emotion = detect_voice_emotion_from_bytes(content, file.filename)
        
        # Perform CBT analysis on transcribed text
        cbt_result = cbt_engine.analyze(transcription)
        
        # Override emotion with voice-based result if available
        if voice_emotion:
            cbt_result['emotion'] = voice_emotion.lower()
            cbt_result['original_emotion'] = voice_emotion
            cbt_result['emotion_source'] = 'voice'
        
        # Process cognitive distortions
        distortion_details = []
        cognitive_wrapper = cbt_engine.config.get("cognitive_wrapper")
        
        if cognitive_wrapper and "distortion_details" in cbt_result:
            for dist in cbt_result.get("distortion_details", []):
                distortion_details.append(CognitiveDistortion(
                    distortion_type=dist["distortion_type"],
                    confidence=dist["confidence"],
                    emoji=dist["emoji"],
                    explanation=cognitive_wrapper.get_distortion_explanation(dist["distortion_type"]),
                    reframing_suggestion=cognitive_wrapper.get_reframing_suggestion(dist["distortion_type"])
                ))
        
        response = CBTAnalysisResponse(
            emotion=cbt_result.get('emotion'),
            emotion_score=cbt_result.get('emotion_score'),
            original_emotion=cbt_result.get('original_emotion'),
            emotion_source=cbt_result.get('emotion_source', 'voice'),
            intent=cbt_result.get('intent'),
            intent_score=cbt_result.get('intent_score'),
            risk=cbt_result.get('risk'),
            risk_score=cbt_result.get('risk_score'),
            escalation=cbt_result.get('escalation'),
            distortions=cbt_result.get('distortions', []),
            distortion_details=distortion_details,
            reframes=cbt_result.get('reframes', []),
            behavioral_suggestions=cbt_result.get('behavioral_suggestions', []),
            clinician_notes=cbt_result.get('clinician_notes', []),
            user_facing=cbt_result.get('user_facing'),
            analysis_timestamp=datetime.now(),
            raw_analysis=cbt_result
        )
        
        return {
            "transcription": transcription,
            "voice_emotion": voice_emotion,
            "cbt_analysis": response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete audio analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    print("🧠 Starting CBT Mental Health Analysis API...")
    print("📍 Server will be available at:")
    print("   - Main API: http://localhost:8000/")
    print("   - Documentation: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    print("🚀 Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")