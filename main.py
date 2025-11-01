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

class JournalAnalysisRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Journal entry title")
    content: str = Field(..., min_length=1, max_length=10000, description="Journal entry content")
    mood: Optional[str] = Field(None, description="Self-reported mood")
    prompt: Optional[str] = Field(None, description="Writing prompt used")

class JournalAnalysisResponse(BaseModel):
    content_analysis: CBTAnalysisResponse
    title_analysis: Optional[CBTAnalysisResponse] = None
    overall_sentiment: str
    key_themes: List[str] = []
    therapeutic_insights: List[str] = []
    progress_indicators: List[str] = []
    recommendations: List[str] = []
    analysis_timestamp: datetime

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
            <strong>/analyze/journal</strong>
            <p>Comprehensive journal analysis with therapeutic insights and recommendations</p>
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

# Journal Analysis Endpoint
@app.post("/analyze/journal", response_model=JournalAnalysisResponse)
async def analyze_journal(request: JournalAnalysisRequest, cbt_engine = Depends(get_cbt_engine)):
    """
    Comprehensive journal analysis for therapeutic insights
    """
    try:
        # Analyze the main content
        content_result = cbt_engine.analyze(request.content)
        
        # Analyze title if substantial enough
        title_result = None
        if len(request.title.split()) > 3:  # Only analyze titles with more than 3 words
            title_result = cbt_engine.analyze(request.title)
        
        # Process cognitive distortions for content
        content_distortion_details = []
        cognitive_wrapper = cbt_engine.config.get("cognitive_wrapper")
        
        if cognitive_wrapper and "distortion_details" in content_result:
            for dist in content_result.get("distortion_details", []):
                content_distortion_details.append(CognitiveDistortion(
                    distortion_type=dist["distortion_type"],
                    confidence=dist["confidence"],
                    emoji=dist["emoji"],
                    explanation=cognitive_wrapper.get_distortion_explanation(dist["distortion_type"]),
                    reframing_suggestion=cognitive_wrapper.get_reframing_suggestion(dist["distortion_type"])
                ))
        
        # Build content analysis response
        content_analysis = CBTAnalysisResponse(
            emotion=content_result.get('emotion'),
            emotion_score=content_result.get('emotion_score'),
            original_emotion=content_result.get('original_emotion'),
            emotion_source="text",
            intent=content_result.get('intent'),
            intent_score=content_result.get('intent_score'),
            risk=content_result.get('risk'),
            risk_score=content_result.get('risk_score'),
            escalation=content_result.get('escalation'),
            distortions=content_result.get('distortions', []),
            distortion_details=content_distortion_details,
            reframes=content_result.get('reframes', []),
            behavioral_suggestions=content_result.get('behavioral_suggestions', []),
            clinician_notes=content_result.get('clinician_notes', []),
            user_facing=content_result.get('user_facing'),
            analysis_timestamp=datetime.now(),
            raw_analysis=content_result
        )
        
        # Build title analysis if available
        title_analysis = None
        if title_result:
            title_distortion_details = []
            if cognitive_wrapper and "distortion_details" in title_result:
                for dist in title_result.get("distortion_details", []):
                    title_distortion_details.append(CognitiveDistortion(
                        distortion_type=dist["distortion_type"],
                        confidence=dist["confidence"],
                        emoji=dist["emoji"],
                        explanation=cognitive_wrapper.get_distortion_explanation(dist["distortion_type"]),
                        reframing_suggestion=cognitive_wrapper.get_reframing_suggestion(dist["distortion_type"])
                    ))
            
            title_analysis = CBTAnalysisResponse(
                emotion=title_result.get('emotion'),
                emotion_score=title_result.get('emotion_score'),
                original_emotion=title_result.get('original_emotion'),
                emotion_source="text",
                intent=title_result.get('intent'),
                intent_score=title_result.get('intent_score'),
                risk=title_result.get('risk'),
                risk_score=title_result.get('risk_score'),
                escalation=title_result.get('escalation'),
                distortions=title_result.get('distortions', []),
                distortion_details=title_distortion_details,
                reframes=title_result.get('reframes', []),
                behavioral_suggestions=title_result.get('behavioral_suggestions', []),
                clinician_notes=title_result.get('clinician_notes', []),
                user_facing=title_result.get('user_facing'),
                analysis_timestamp=datetime.now(),
                raw_analysis=title_result
            )
        
        # Generate overall insights
        overall_sentiment = _determine_overall_sentiment(content_result, request.mood)
        key_themes = _extract_key_themes(request.content, content_result)
        therapeutic_insights = _generate_therapeutic_insights(content_result, title_result)
        progress_indicators = _identify_progress_indicators(content_result, request.mood)
        recommendations = _generate_journal_recommendations(content_result, title_result)
        
        return JournalAnalysisResponse(
            content_analysis=content_analysis,
            title_analysis=title_analysis,
            overall_sentiment=overall_sentiment,
            key_themes=key_themes,
            therapeutic_insights=therapeutic_insights,
            progress_indicators=progress_indicators,
            recommendations=recommendations,
            analysis_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Journal analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Journal analysis failed: {str(e)}")

def _determine_overall_sentiment(content_result: Dict, mood: Optional[str]) -> str:
    """Determine overall sentiment from analysis and self-reported mood"""
    emotion = content_result.get('emotion', '').lower()
    risk_score = content_result.get('risk_score', 0.0)
    
    if risk_score > 0.6:
        return "concerning"
    elif emotion in ['sad', 'angry', 'fearful', 'disgusted']:
        return "negative"
    elif emotion in ['happy', 'joyful', 'excited']:
        return "positive"
    elif mood and mood.lower() in ['happy', 'excited', 'loved']:
        return "positive"
    elif mood and mood.lower() in ['sad']:
        return "negative"
    else:
        return "neutral"

def _extract_key_themes(content: str, analysis: Dict) -> List[str]:
    """Extract key themes from journal content"""
    themes = []
    
    # Theme detection based on content analysis
    emotion = analysis.get('emotion', '').lower()
    intent = analysis.get('intent', '').lower()
    distortions = analysis.get('distortions', [])
    
    if emotion:
        themes.append(f"Emotional state: {emotion}")
    
    if intent and 'help' in intent:
        themes.append("Seeking support")
    
    if distortions:
        themes.append("Cognitive patterns identified")
    
    # Simple keyword-based themes
    content_lower = content.lower()
    if any(word in content_lower for word in ['work', 'job', 'career', 'boss']):
        themes.append("Work/Career")
    if any(word in content_lower for word in ['relationship', 'family', 'friend', 'partner']):
        themes.append("Relationships")
    if any(word in content_lower for word in ['anxiety', 'anxious', 'worry', 'nervous']):
        themes.append("Anxiety")
    if any(word in content_lower for word in ['sad', 'depression', 'depressed', 'down']):
        themes.append("Mood concerns")
    if any(word in content_lower for word in ['goal', 'achievement', 'success', 'progress']):
        themes.append("Personal growth")
        
    return themes[:5]  # Limit to 5 themes

def _generate_therapeutic_insights(content_result: Dict, title_result: Optional[Dict]) -> List[str]:
    """Generate therapeutic insights from analysis"""
    insights = []
    
    # Risk-based insights
    risk_score = content_result.get('risk_score', 0.0)
    if risk_score > 0.4:
        insights.append("Entry indicates elevated emotional distress - consider professional support")
    
    # Distortion-based insights
    distortions = content_result.get('distortions', [])
    if distortions:
        insights.append(f"Cognitive patterns detected: {', '.join(distortions[:3])}")
    
    # Emotional insights
    emotion = content_result.get('emotion', '')
    emotion_score = content_result.get('emotion_score', 0.0)
    if emotion and emotion_score > 0.7:
        insights.append(f"Strong {emotion} emotion detected - good emotional awareness")
    
    # Intent insights
    intent = content_result.get('intent', '')
    if 'help' in intent.lower():
        insights.append("Demonstrates healthy help-seeking behavior")
    
    return insights[:4]  # Limit to 4 insights

def _identify_progress_indicators(analysis: Dict, mood: Optional[str]) -> List[str]:
    """Identify positive progress indicators"""
    indicators = []
    
    # Positive emotional expressions
    emotion = analysis.get('emotion', '').lower()
    if emotion in ['happy', 'joyful', 'grateful']:
        indicators.append("Positive emotional expression")
    
    # Self-awareness indicators
    if analysis.get('distortions'):
        indicators.append("Demonstrating self-awareness of thought patterns")
    
    # Mood alignment
    if mood and mood in ['happy', 'excited', 'loved']:
        indicators.append("Self-reported positive mood")
    
    # Behavioral insights from reframes
    reframes = analysis.get('reframes', [])
    if reframes:
        indicators.append("Showing capacity for cognitive reframing")
    
    return indicators[:3]  # Limit to 3 indicators

def _generate_journal_recommendations(content_result: Dict, title_result: Optional[Dict]) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []
    
    # Risk-based recommendations
    risk_score = content_result.get('risk_score', 0.0)
    if risk_score > 0.6:
        recommendations.append("Consider reaching out to a mental health professional")
    elif risk_score > 0.3:
        recommendations.append("Practice grounding techniques and self-care activities")
    
    # Distortion-based recommendations
    distortions = content_result.get('distortions', [])
    if 'catastrophizing' in distortions:
        recommendations.append("Try the 'best case, worst case, most likely case' exercise")
    if 'all_or_nothing' in distortions:
        recommendations.append("Practice identifying the gray areas in situations")
    
    # Emotion-based recommendations
    emotion = content_result.get('emotion', '').lower()
    if emotion in ['sad', 'angry']:
        recommendations.append("Consider engaging in mood-lifting activities or exercise")
    elif emotion == 'anxious':
        recommendations.append("Try deep breathing or mindfulness exercises")
    
    # General recommendations
    recommendations.append("Continue regular journaling to track patterns over time")
    
    return recommendations[:4]  # Limit to 4 recommendations

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