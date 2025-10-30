"""
CBT Models Module - Extracted and adapted from integrated_cbt_streamlit.py
Contains all model loading and prediction functions for FastAPI use
"""

import torch
import os
import sys
import json
import numpy as np
from transformers import BertModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import tempfile
from functools import lru_cache

# Add the project folders to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'intent_classification'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'risk-detection'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'emotion_classifier'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_distortion'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'voicebasedemotion'))

# Import the CBT engine
from cbt import CBTEngine

# Import model dependencies
try:
    from emotion_classifier.emotion_model import LSTMEmotionClassifier, load_vocab, predict_emotion as lstm_predict_emotion, emotion_labels
except ImportError:
    print("Warning: Emotion classifier not available")
    LSTMEmotionClassifier = None
    emotion_labels = []

try:
    from cognitive_distortion.cognitive_distortion_model import CognitiveDistortionModel
except ImportError:
    print("Warning: Cognitive distortion model not available")
    CognitiveDistortionModel = None

# Optional voice emotion deps
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception as _e:
    LIBROSA_AVAILABLE = False

# Global variables to store loaded models
_models = {
    "intent_classifier": None,
    "suicide_model": None,
    "suicide_tokenizer": None,
    "emotion_model_data": None,
    "cognitive_model": None,
    "cbt_engine": None,
    "asr_pipeline": None,
    "voice_components": None
}

# Suicide Risk Model Architecture
class SuicideRiskModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(SuicideRiskModel, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(768, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output[0][:, 0, :]
        x = self.dropout(pooled_output)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Model wrapper classes
class IntentModelWrapper:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def predict(self, text):
        if self.classifier is None:
            return {"intent": "unknown", "confidence": 0.0}
        
        try:
            results = predict_intent(text, self.classifier)
            if isinstance(results, list) and len(results) > 0:
                top_result = results[0]
                return {
                    "intent": top_result['label'],
                    "confidence": top_result['score']
                }
        except Exception as e:
            print(f"Error in intent prediction: {e}")
        
        return {"intent": "unknown", "confidence": 0.0}

class SuicideRiskModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(self, text):
        if self.model is None or self.tokenizer is None:
            return {"level": "low", "score": 0.0}
        
        try:
            result = predict_suicide_risk(text, self.model, self.tokenizer)
            risk_score = result["risk_score"]
            
            if "Minimal" in result["risk_category"]:
                risk_level = "low"
            elif "Low" in result["risk_category"]:
                risk_level = "low"
            elif "Moderate" in result["risk_category"]:
                risk_level = "moderate"
            elif "High" in result["risk_category"]:
                risk_level = "high"
            else:
                risk_level = "urgent"
            
            return {
                "level": risk_level,
                "score": risk_score
            }
        except Exception as e:
            print(f"Error in risk prediction: {e}")
            return {"level": "low", "score": 0.0}

class EmotionModelWrapper:
    def __init__(self, model_data):
        self.model_data = model_data
    
    def predict(self, text):
        if self.model_data is None:
            return {"emotion": "neutral", "score": 0.0}
        
        try:
            emotion_results = predict_emotion(text, self.model_data)
            
            if isinstance(emotion_results, list) and len(emotion_results) > 0:
                top_emotion = emotion_results[0]
                return {
                    "emotion": top_emotion["label"].lower(),
                    "score": top_emotion["score"],
                    "original_emotion": top_emotion["label"]
                }
            
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
        
        return {"emotion": "neutral", "score": 0.0}

class CognitiveDistortionModelWrapper:
    def __init__(self, model):
        self.model = model
        self.distortion_map = {
            "all-or-nothing thinking": "all_or_nothing",
            "catastrophizing": "catastrophizing",
            "fortune-telling": "catastrophizing",
            "mind reading": "mind_reading",
            "overgeneralization": "overgeneralization",
            "personalization": "personalization",
            "labeling": "overgeneralization",
            "magnification or minimization": "catastrophizing",
            "mental filtering": "all_or_nothing",
            "should statements": "overgeneralization"
        }
        
        self.emoji_map = {
            "all-or-nothing thinking": "⚫⚪",
            "catastrophizing": "💥",
            "fortune-telling": "🔮",
            "labeling": "🏷️",
            "magnification or minimization": "🔍",
            "mental filtering": "🧠",
            "mind reading": "👁️",
            "overgeneralization": "🌐",
            "personalization": "👤",
            "should statements": "📜"
        }
    
    def analyze(self, text):
        if self.model is None:
            return {}
        
        try:
            result = predict_cognitive_distortions(text, self.model)
            
            if "distortions" in result and result["distortions"]:
                distortions = []
                distortion_details = []
                
                for dist in result["distortions"]:
                    dist_type = dist["distortion_type"]
                    confidence = dist["confidence"]
                    
                    distortion_details.append({
                        "distortion_type": dist_type,
                        "confidence": confidence,
                        "emoji": self.emoji_map.get(dist_type, "🧩")
                    })
                    
                    if confidence > 0.3:
                        mapped_distortion = self.distortion_map.get(dist_type)
                        if mapped_distortion:
                            distortions.append(mapped_distortion)
                
                return {
                    "detected_distortions": distortions,
                    "distortion_details": distortion_details
                }
            
        except Exception as e:
            print(f"Error in cognitive distortion detection: {e}")
        
        return {}
        
    def get_distortion_explanation(self, distortion_type):
        explanations = {
            "all-or-nothing thinking": "Seeing things in absolute, black and white categories, with no middle ground.",
            "catastrophizing": "Expecting the worst possible outcome to happen, often with limited evidence.",
            "fortune-telling": "Predicting negative future events or outcomes without considering other possibilities.",
            "mind reading": "Assuming you know what others are thinking without sufficient evidence.",
            "overgeneralization": "Taking one negative event and applying it as a never-ending pattern of defeat.",
            "personalization": "Taking excessive responsibility for external events or other people's behaviors.",
            "labeling": "Assigning global negative traits to yourself or others based on specific behaviors.",
            "magnification or minimization": "Exaggerating negatives and minimizing positives in your evaluation of events.",
            "mental filtering": "Focusing exclusively on negative aspects while ignoring positive elements.",
            "should statements": "Having rigid rules about how you or others should behave, leading to guilt and disappointment."
        }
        return explanations.get(distortion_type, "No explanation available for this distortion type.")
    
    def get_reframing_suggestion(self, distortion_type):
        suggestions = {
            "all-or-nothing thinking": "Look for the grey areas and nuances in the situation. Consider that most situations fall somewhere between extremes.",
            "catastrophizing": "Ask yourself what's most likely to happen instead of focusing on the worst scenario. Consider past similar situations and their actual outcomes.",
            "fortune-telling": "Remind yourself that you cannot predict the future with certainty. Consider multiple possible outcomes, including positive ones.",
            "mind reading": "Check your assumptions by asking for clarification or considering alternative interpretations of others' behaviors.",
            "overgeneralization": "Look for specific counter-examples that contradict your generalization. Focus on this specific situation rather than applying it broadly.",
            "personalization": "Consider all factors that might have contributed to the situation, not just your actions. Remember that many outcomes are influenced by multiple factors.",
            "labeling": "Describe the specific behavior rather than applying a global label. Remind yourself that one action doesn't define a person's entire character.",
            "magnification or minimization": "Try to evaluate both positive and negative aspects with equal weight. Ask if you would judge someone else this harshly.",
            "mental filtering": "Deliberately look for positive aspects you might be filtering out. Ask what evidence contradicts your negative focus.",
            "should statements": "Replace 'should' with more flexible language like 'I would prefer' or 'It would be helpful if.' Accept that people (including yourself) are imperfect."
        }
        return suggestions.get(distortion_type, "No reframing suggestion available for this distortion type.")

# Model loading functions
def load_intent_classifier():
    """Load the intent classification model"""
    if _models["intent_classifier"] is not None:
        return _models["intent_classifier"]
        
    model_dir = os.path.join(os.path.dirname(__file__), 'intent_classification')
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # Use CPU
            return_all_scores=True,
            truncation=True
        )
        _models["intent_classifier"] = classifier
        return classifier
    except Exception as e:
        print(f"Error loading intent classification model: {e}")
        return None

def predict_intent(text, classifier):
    """Predict intent classification"""
    if classifier is None:
        return "Model not loaded"
    
    try:
        results = classifier(text)
        top_results = sorted(results[0], key=lambda x: x['score'], reverse=True)[:3]
        return top_results
    except Exception as e:
        return f"Error: {e}"

def load_suicide_risk_model():
    """Load the suicide risk detection model"""
    if _models["suicide_model"] is not None:
        return _models["suicide_model"], _models["suicide_tokenizer"]
        
    try:
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        model = SuicideRiskModel(bert_model)
        
        model_path = os.path.join(os.path.dirname(__file__), 'risk-detection', 'suicide_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        _models["suicide_model"] = model
        _models["suicide_tokenizer"] = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"Error loading suicide risk model: {e}")
        return None, None

def predict_suicide_risk(text, model, tokenizer):
    """Predict suicide risk level"""
    if model is None or tokenizer is None:
        return {"risk_score": 0.0, "risk_category": "Model not loaded"}
    
    try:
        tokenized = tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        with torch.no_grad():
            score = model(input_ids, attention_mask).item()
        
        if score < 0.20:
            risk_level = "🟢 Minimal Risk"
        elif score < 0.40:
            risk_level = "🟡 Low Risk"
        elif score < 0.60:
            risk_level = "🟠 Moderate Risk"
        elif score < 0.80:
            risk_level = "🔴 High Risk"
        else:
            risk_level = "⚠️ Severe Risk"
        
        return {
            "risk_score": round(score, 4),
            "risk_category": risk_level
        }
    except Exception as e:
        return {"risk_score": 0.0, "risk_category": f"Error: {e}"}

def load_emotion_classifier():
    """Load the custom LSTM emotion classification model"""
    if _models["emotion_model_data"] is not None:
        return _models["emotion_model_data"]
        
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'emotion_classifier', 'emotion_model.pth')
        vocab_path = os.path.join(os.path.dirname(__file__), 'emotion_classifier', 'vocab.txt')
        device = torch.device('cpu')
        
        if not LSTMEmotionClassifier:
            print("LSTM Emotion Classifier not available")
            return None
            
        vocab = load_vocab(vocab_path)
        model_data = torch.load(model_path, map_location=device)
        
        embed_dim = model_data.get('embed_dim', 300)
        hidden_dim = model_data.get('hidden_dim', 256)
        
        actual_vocab_size = model_data['model_state_dict']['embedding.weight'].shape[0]
        num_classes = len(emotion_labels)
        
        model = LSTMEmotionClassifier(actual_vocab_size, embed_dim, hidden_dim, num_classes)
        
        try:
            if 'model_state_dict' in model_data:
                model.load_state_dict(model_data['model_state_dict'])
            else:
                model.load_state_dict(model_data)
        except Exception as e:
            print(f"Could not load model weights: {e}. Using default model.")
            
        model.eval()
        model.to(device)
        
        result = {
            "model": model,
            "vocab": vocab,
            "device": device,
            "emotion_labels": emotion_labels
        }
        _models["emotion_model_data"] = result
        return result
    except Exception as e:
        print(f"Error loading emotion classification model: {e}")
        return None

def predict_emotion(text, model_data):
    """Predict emotions from text using the LSTM model"""
    if model_data is None:
        return "Model not loaded"
    
    try:
        model = model_data["model"]
        vocab = model_data["vocab"]
        device = model_data["device"]
        
        emotion_predictions = lstm_predict_emotion(text, model, vocab, device)
        
        if not emotion_predictions:
            return [{"label": "neutral", "score": 1.0}]
        
        return emotion_predictions
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return [{"label": "neutral", "score": 1.0}]

def load_cognitive_distortion_model():
    """Load the cognitive distortion detection model"""
    if _models["cognitive_model"] is not None:
        return _models["cognitive_model"]
        
    try:
        cognitive_model_path = os.path.join(os.path.dirname(__file__), 'cognitive_distortion')
        sys.path.append(cognitive_model_path)
        
        import importlib.util
        module_path = os.path.join(cognitive_model_path, 'cognitive_distortion_model.py')
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location("cognitive_distortion_model", module_path)
            cognitive_distortion_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cognitive_distortion_module)
            CognitiveDistortionModel = cognitive_distortion_module.CognitiveDistortionModel
            
            model = CognitiveDistortionModel()
            model.load_model()
            _models["cognitive_model"] = model
            return model
        else:
            print(f"Cognitive distortion model file not found at {module_path}")
            return None
    except Exception as e:
        print(f"Error loading cognitive distortion model: {e}")
        return None

def predict_cognitive_distortions(text, model):
    """Predict cognitive distortions in text"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        result = model.predict(text)
        return result
    except Exception as e:
        return {"error": str(e)}

# Audio processing functions
def load_voice_emotion_components():
    """Load components from voicebasedemotion/speechemotion.py"""
    if _models["voice_components"] is not None:
        return _models["voice_components"]
        
    try:
        from speechemotion import (
            predict_emotion as predict_voice_emotion,
            model as VOICE_MODEL,
            feature_extractor as VOICE_FE,
            id2label as VOICE_ID2LABEL,
        )
        components = (predict_voice_emotion, VOICE_MODEL, VOICE_FE, VOICE_ID2LABEL)
        _models["voice_components"] = components
        return components
    except Exception as e:
        print(f"Voice emotion module not available: {e}")
        return None, None, None, None

def detect_voice_emotion_from_bytes(raw_bytes: bytes, filename: str = "audio.wav"):
    """Detect emotion from audio bytes"""
    predict_fn, v_model, v_fe, id2label = load_voice_emotion_components()
    if not predict_fn or not v_model or not v_fe or not id2label:
        return None
    if not LIBROSA_AVAILABLE:
        print("librosa is required for voice emotion detection.")
        return None
        
    suffix = os.path.splitext(filename)[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        label = predict_fn(tmp_path, v_model, v_fe, id2label)
        return label
    except Exception as e:
        print(f"Voice emotion detection failed: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def load_asr_pipeline():
    """Load ASR pipeline for speech-to-text"""
    if _models["asr_pipeline"] is not None:
        return _models["asr_pipeline"]
        
    try:
        asr = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-small",
            device=-1,
            chunk_length_s=30,
        )
        _models["asr_pipeline"] = asr
        return asr
    except Exception as e:
        print(f"ASR pipeline not available: {e}")
        return None

def transcribe_audio_locally(raw_bytes: bytes, filename: str = "audio.wav"):
    """Transcribe audio to text"""
    asr = load_asr_pipeline()
    if asr is None:
        return None
        
    suffix = os.path.splitext(filename)[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
            
        if LIBROSA_AVAILABLE:
            import librosa
            target_sr = 16000
            audio_array, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
            out = asr({"array": audio_array, "sampling_rate": target_sr})
        else:
            out = asr(tmp_path)
            
        if isinstance(out, dict) and "text" in out:
            return out["text"]
        if isinstance(out, str):
            return out
        return None
    except Exception as e:
        print(f"Transcription failed: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def initialize_cbt_engine():
    """Initialize the CBT engine with loaded models"""
    if _models["cbt_engine"] is not None:
        return _models["cbt_engine"]
        
    model_status = {}
    
    # Load models
    try:
        intent_classifier = load_intent_classifier()
        intent_wrapper = IntentModelWrapper(intent_classifier)
        model_status["intent"] = intent_classifier is not None
    except Exception as e:
        print(f"⚠️ Intent model not loaded: {str(e)}")
        intent_wrapper = IntentModelWrapper(None)
        model_status["intent"] = False
    
    try:
        suicide_model, suicide_tokenizer = load_suicide_risk_model()
        risk_wrapper = SuicideRiskModelWrapper(suicide_model, suicide_tokenizer)
        model_status["risk"] = suicide_model is not None
    except Exception as e:
        print(f"⚠️ Risk model not loaded: {str(e)}")
        risk_wrapper = SuicideRiskModelWrapper(None, None)
        model_status["risk"] = False
    
    try:
        emotion_model_data = load_emotion_classifier()
        emotion_wrapper = EmotionModelWrapper(emotion_model_data)
        model_status["emotion"] = emotion_model_data is not None
    except Exception as e:
        print(f"⚠️ Emotion model not loaded: {str(e)}")
        emotion_wrapper = EmotionModelWrapper(None)
        model_status["emotion"] = False
    
    try:
        cognitive_model = load_cognitive_distortion_model()
        cognitive_wrapper = CognitiveDistortionModelWrapper(cognitive_model)
        model_status["cognitive"] = cognitive_model is not None
    except Exception as e:
        print(f"⚠️ Cognitive distortion model not loaded: {str(e)}")
        cognitive_wrapper = CognitiveDistortionModelWrapper(None)
        model_status["cognitive"] = False
    
    config = {
        "escalation_levels": ["high", "urgent"],
        "models_loaded": model_status,
        "cognitive_wrapper": cognitive_wrapper
    }
    
    cbt_engine = CBTEngine(
        emotion_model=emotion_wrapper,
        intent_model=intent_wrapper,
        risk_model=risk_wrapper,
        config=config
    )
    
    # Enhanced detect_distortions if cognitive model is available
    if cognitive_wrapper.model is not None:
        original_detect_distortions = cbt_engine.detect_distortions
        
        def enhanced_detect_distortions(text):
            result = cognitive_wrapper.analyze(text)
            if result and "detected_distortions" in result and result["detected_distortions"]:
                return result["detected_distortions"]
            return original_detect_distortions(text)
        
        cbt_engine.detect_distortions = enhanced_detect_distortions
    
    _models["cbt_engine"] = cbt_engine
    return cbt_engine

def get_model_status():
    """Get status of all loaded models"""
    cbt_engine = initialize_cbt_engine()
    return cbt_engine.config.get("models_loaded", {})