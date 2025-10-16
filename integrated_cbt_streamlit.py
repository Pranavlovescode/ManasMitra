import streamlit as st
import torch
import os
import sys
import json
import random
import numpy as np
from transformers import BertModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from emotion_classifier.emotion_model import LSTMEmotionClassifier, load_vocab, predict_emotion as lstm_predict_emotion, emotion_labels
from cognitive_distortion.cognitive_distortion_model import CognitiveDistortionModel



# Debug imports and paths
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in cognitive_distortion directory:")
cognitive_distortion_path = os.path.join(os.path.dirname(__file__), 'cognitive_distortion')
try:
    print(os.listdir(cognitive_distortion_path))
except Exception as e:
    print(f"Error listing cognitive_distortion directory: {e}")

# Import the CBT engine
from cbt import CBTEngine

# Add the project folders to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'intent_classification'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'risk-detection'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'emotion_classifier'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_distortion'))

# Import models (using try/except to handle potential import errors)
try:
    from cognitive_distortion_model import CognitiveDistortionModel
except ImportError:
    pass

# Model wrapper classes to integrate with CBT engine
class IntentModelWrapper:
    """Wrapper for intent classification model to work with CBT engine"""
    
    def __init__(self, classifier):
        self.classifier = classifier
    
    def predict(self, text):
        """CBT engine expects .predict() method"""
        if self.classifier is None:
            return {"intent": "unknown", "confidence": 0.0}
        
        try:
            results = predict_intent(text, self.classifier)
            # Get the top prediction
            if isinstance(results, list) and len(results) > 0:
                top_result = results[0]  # Top result is already sorted
                return {
                    "intent": top_result['label'],
                    "confidence": top_result['score']
                }
        except Exception as e:
            st.error(f"Error in intent prediction: {e}")
        
        return {"intent": "unknown", "confidence": 0.0}

class SuicideRiskModelWrapper:
    """Wrapper for suicide risk model to work with CBT engine"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(self, text):
        """CBT engine expects .predict() method"""
        if self.model is None or self.tokenizer is None:
            return {"level": "low", "score": 0.0}
        
        try:
            result = predict_suicide_risk(text, self.model, self.tokenizer)
            risk_score = result["risk_score"]
            
            # Map risk category from predict_suicide_risk to CBT engine expected format
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
            st.error(f"Error in risk prediction: {e}")
            return {"level": "low", "score": 0.0}

class EmotionModelWrapper:
    """Wrapper for emotion classifier model to work with CBT engine"""
    
    def __init__(self, model_data):
        self.model_data = model_data
    
    def predict(self, text):
        """CBT engine expects .predict() method"""
        if self.model_data is None:
            return {"emotion": "neutral", "score": 0.0}
        
        try:
            emotion_results = predict_emotion(text, self.model_data)
            
            if isinstance(emotion_results, list) and len(emotion_results) > 0:
                # Get the top emotion
                top_emotion = emotion_results[0]
                
                # # Map to simpler emotion categories for CBT engine
                # emotion_mapping = {
                #     "sadness": "sad",
                #     "grief": "sad",
                #     "disappointment": "sad",
                #     "remorse": "sad",
                #     "fear": "anxious",
                #     "nervousness": "anxious",
                #     "confusion": "anxious",
                #     "anger": "angry",
                #     "annoyance": "angry",
                #     "disapproval": "angry",
                #     "disgust": "angry",
                #     "joy": "happy",
                #     "amusement": "happy",
                #     "excitement": "happy",
                #     "gratitude": "happy",
                #     "love": "happy",
                #     "optimism": "happy",
                #     "relief": "happy",
                #     "pride": "happy",
                #     "admiration": "happy",
                #     "desire": "happy",
                #     "caring": "happy",
                #     "realization": "neutral",
                #     "surprise": "neutral",
                #     "neutral": "neutral",
                #     "embarrassment": "anxious"
                # }
                
                # mapped_emotion = emotion_mapping.get(top_emotion["label"].lower(), "neutral")
                
                return {
                    "emotion": top_emotion["label"].lower(),
                    "score": top_emotion["score"],
                    "original_emotion": top_emotion["label"]
                }
            
        except Exception as e:
            st.error(f"Error in emotion prediction: {e}")
        
        return {"emotion": "neutral", "score": 0.0}

class CognitiveDistortionModelWrapper:
    """Wrapper for cognitive distortion model to work with CBT engine"""
    
    def __init__(self, model):
        self.model = model
        self.distortion_map = {
            "all-or-nothing thinking": "all_or_nothing",
            "catastrophizing": "catastrophizing",
            "fortune-telling": "catastrophizing",  # Map to existing CBT engine category
            "mind reading": "mind_reading",
            "overgeneralization": "overgeneralization",
            "personalization": "personalization",
            "labeling": "overgeneralization",  # Map to existing CBT engine category
            "magnification or minimization": "catastrophizing",  # Map to existing CBT engine category
            "mental filtering": "all_or_nothing",  # Map to existing CBT engine category
            "should statements": "overgeneralization"  # Map to existing CBT engine category
        }
        
        # Emoji mappings for visual representation
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
        """CBT engine can call .analyze() method"""
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
                    
                    # Save the original distortion details for UI display
                    distortion_details.append({
                        "distortion_type": dist_type,
                        "confidence": confidence,
                        "emoji": self.emoji_map.get(dist_type, "🧩")
                    })
                    
                    # Only include distortions with confidence > 0.3
                    if confidence > 0.3:
                        # Map to CBT engine's expected distortion categories
                        mapped_distortion = self.distortion_map.get(dist_type)
                        if mapped_distortion:
                            distortions.append(mapped_distortion)
                
                return {
                    "detected_distortions": distortions,
                    "distortion_details": distortion_details
                }
            
        except Exception as e:
            st.error(f"Error in cognitive distortion detection: {e}")
        
        return {}
        
    def get_distortion_explanation(self, distortion_type):
        """Get an explanation for the given distortion type"""
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
        """Get a reframing suggestion for the given distortion type"""
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

@st.cache_resource
def load_intent_classifier():
    """Load the intent classification model"""
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
        return classifier
    except Exception as e:
        st.error(f"Error loading intent classification model: {e}")
        return None

def predict_intent(text, classifier):
    """Predict intent classification"""
    if classifier is None:
        return "Model not loaded"
    
    try:
        results = classifier(text)
        # Get top 3 predictions
        top_results = sorted(results[0], key=lambda x: x['score'], reverse=True)[:3]
        return top_results
    except Exception as e:
        return f"Error: {e}"

@st.cache_resource
def load_suicide_risk_model():
    """Load the suicide risk detection model"""
    try:
        # Load BERT model and tokenizer
        # use BertModel explicitly (some transformers builds don't expose AutoModel)
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Initialize the suicide risk model
        model = SuicideRiskModel(bert_model)
        
        # Load trained weights
        model_path = os.path.join(os.path.dirname(__file__), 'risk-detection', 'suicide_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading suicide risk model: {e}")
        return None, None
        
def predict_suicide_risk(text, model, tokenizer):
    """Predict suicide risk level"""
    if model is None or tokenizer is None:
        return {"risk_score": 0.0, "risk_category": "Model not loaded"}
    
    try:
        # Tokenize the text
        tokenized = tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Make prediction
        with torch.no_grad():
            score = model(input_ids, attention_mask).item()
        
        # Determine risk level
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

# Custom Emotion Classification Model
@st.cache_resource
def load_emotion_classifier():
    """Load the custom LSTM emotion classification model"""
    try:
        # Set up paths for model and vocabulary
        model_path = os.path.join(os.path.dirname(__file__), 'emotion_classifier', 'emotion_model.pth')
        vocab_path = os.path.join(os.path.dirname(__file__), 'emotion_classifier', 'vocab.txt')
        device = torch.device('cpu')
        
        # Load vocabulary
        vocab = load_vocab(vocab_path)
        
        # Load the saved model data (including state dict and parameters)
        model_data = torch.load(model_path, map_location=device)
        
        # Extract model parameters from saved data
        embed_dim = model_data.get('embed_dim', 300)
        hidden_dim = model_data.get('hidden_dim', 256)
        
        # Get the actual vocabulary size from the state dict
        actual_vocab_size = model_data['model_state_dict']['embedding.weight'].shape[0]
        num_classes = len(emotion_labels)
        
        # Create model with the correct parameters
        model = LSTMEmotionClassifier(actual_vocab_size, embed_dim, hidden_dim, num_classes)
        
        try:
            # Load the actual state dict from the nested structure
            if 'model_state_dict' in model_data:
                model.load_state_dict(model_data['model_state_dict'])
                st.success("Loaded model state_dict successfully!")
            else:
                # Try loading directly if not nested
                model.load_state_dict(model_data)
                st.success("Loaded model weights directly!")
        except Exception as e:
            st.warning(f"Could not load model weights: {e}. Using default model.")
            
        model.eval()
        model.to(device)
        
        st.success("LSTM emotion classification model loaded successfully!")
        return {
            "model": model,
            "vocab": vocab,
            "device": device,
            "emotion_labels": emotion_labels
        }
    except Exception as e:
        st.error(f"Error loading emotion classification model: {e}")
        return None

def predict_emotion(text, model_data):
    """Predict emotions from text using the LSTM model"""
    if model_data is None:
        return "Model not loaded"
    
    try:
        # Extract model components from the model_data
        model = model_data["model"]
        vocab = model_data["vocab"]
        device = model_data["device"]
        
        # Use the imported lstm_predict_emotion function from emotion_model.py
        emotion_predictions = lstm_predict_emotion(text, model, vocab, device)
        
        if not emotion_predictions:
            # If no emotions detected, add a neutral emotion
            return [{"label": "neutral", "score": 1.0}]
        
        # Return the predictions in the expected format
        return emotion_predictions
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return [{"label": "neutral", "score": 1.0}]

@st.cache_resource
def load_cognitive_distortion_model():
    """Load the cognitive distortion detection model"""
    try:
        # Use the absolute import path
        cognitive_model_path = os.path.join(os.path.dirname(__file__), 'cognitive_distortion')
        sys.path.append(cognitive_model_path)
        
        # Import using direct file loading to avoid path issues
        import importlib.util
        module_path = os.path.join(cognitive_model_path, 'cognitive_distortion_model.py')
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location("cognitive_distortion_model", module_path)
            cognitive_distortion_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cognitive_distortion_module)
            CognitiveDistortionModel = cognitive_distortion_module.CognitiveDistortionModel
            
            model = CognitiveDistortionModel()
            model.load_model()
            st.success("Cognitive distortion model loaded successfully!")
            return model
        else:
            st.error(f"Cognitive distortion model file not found at {module_path}")
            return None
    except Exception as e:
        st.error(f"Error loading cognitive distortion model: {e}")
        return None
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

@st.cache_resource
def initialize_cbt_engine():
    """Initialize the CBT engine with loaded models"""
    # Load models with error handling
    model_status = {}
    
    # Intent model
    try:
        intent_classifier = load_intent_classifier()
        intent_wrapper = IntentModelWrapper(intent_classifier)
        model_status["intent"] = intent_classifier is not None
    except Exception as e:
        st.warning(f"⚠️ Intent model not loaded: {str(e)}")
        intent_wrapper = IntentModelWrapper(None)
        model_status["intent"] = False
    
    # Risk model
    try:
        suicide_model, suicide_tokenizer = load_suicide_risk_model()
        risk_wrapper = SuicideRiskModelWrapper(suicide_model, suicide_tokenizer)
        model_status["risk"] = suicide_model is not None
    except Exception as e:
        st.warning(f"⚠️ Risk model not loaded: {str(e)}")
        risk_wrapper = SuicideRiskModelWrapper(None, None)
        model_status["risk"] = False
    
    # Emotion model
    try:
        emotion_model_data = load_emotion_classifier()
        emotion_wrapper = EmotionModelWrapper(emotion_model_data)
        model_status["emotion"] = emotion_model_data is not None
    except Exception as e:
        st.warning(f"⚠️ Emotion model not loaded: {str(e)}")
        emotion_wrapper = EmotionModelWrapper(None)
        model_status["emotion"] = False
    
    # Cognitive distortion model
    try:
        cognitive_model = load_cognitive_distortion_model()
        cognitive_wrapper = CognitiveDistortionModelWrapper(cognitive_model)
        model_status["cognitive"] = cognitive_model is not None
    except Exception as e:
        st.warning(f"⚠️ Cognitive distortion model not loaded: {str(e)}")
        cognitive_wrapper = CognitiveDistortionModelWrapper(None)
        model_status["cognitive"] = False
    
    # Initialize CBT engine with enhanced configuration
    config = {
        "escalation_levels": ["high", "urgent"],
        "models_loaded": model_status,
        "cognitive_wrapper": cognitive_wrapper  # Store the wrapper for later use
    }
    
    # Enhance CBT engine with cognitive distortion model if available
    cbt_engine = CBTEngine(
        emotion_model=emotion_wrapper,
        intent_model=intent_wrapper,
        risk_model=risk_wrapper,
        config=config
    )
    
    # Add cognitive distortion model to detect_distortions method if available
    if cognitive_wrapper.model is not None:
        original_detect_distortions = cbt_engine.detect_distortions
        
        def enhanced_detect_distortions(text):
            # First try with the cognitive model
            result = cognitive_wrapper.analyze(text)
            if result and "detected_distortions" in result and result["detected_distortions"]:
                detected = result["detected_distortions"]
                # Store distortion details for UI display
                if "distortion_details" in result:
                    if 'distortion_details' not in st.session_state:
                        st.session_state['distortion_details'] = []
                    st.session_state["distortion_details"] = result["distortion_details"]
                return detected
            
            # Fall back to the original method if cognitive model didn't find anything
            return original_detect_distortions(text)
        
        # Replace the original method with the enhanced one
        cbt_engine.detect_distortions = enhanced_detect_distortions
    
    return cbt_engine

# Streamlit UI
st.set_page_config(
    page_title="CBT Mental Health Analysis",
    page_icon="🧠",
    layout="wide"
)

st.title('🧠 CBT Mental Health Analysis')
st.write("Comprehensive Cognitive Behavioral Therapy analysis using AI models")

# Load CBT engine
with st.spinner("Loading CBT analysis engine..."):
    cbt_engine = initialize_cbt_engine()
    
# Display model status
with st.expander("Model Status"):
    models_loaded = cbt_engine.config.get("models_loaded", {})
    
    # Create columns for model status indicators
    cols = st.columns(4)
    
    with cols[0]:
        if models_loaded.get("intent", False):
            st.success("✅ Intent Model: Loaded")
        else:
            st.error("❌ Intent Model: Not Loaded")
    
    with cols[1]:
        if models_loaded.get("emotion", False):
            st.success("✅ Emotion Model: Loaded")
        else:
            st.error("❌ Emotion Model: Not Loaded")
    
    with cols[2]:
        if models_loaded.get("cognitive", False):
            st.success("✅ Cognitive Model: Loaded")
        else:
            st.error("❌ Cognitive Model: Not Loaded")
    
    with cols[3]:
        if models_loaded.get("risk", False):
            st.success("✅ Risk Model: Loaded")
        else:
            st.error("❌ Risk Model: Not Loaded")
            
    if not any(models_loaded.values()):
        st.warning("""
        No models were loaded successfully. The application will fall back to basic rule-based analysis.
        
        To load the models properly:
        1. Ensure all required model files are in the correct directories
        2. Install all dependencies with `pip install -r requirements.txt`
        3. Run `python setup_models.py` to prepare model files
        """)
    elif not all(models_loaded.values()):
        st.info("""
        Some models were not loaded successfully. The application will fall back to simpler analysis for those components.
        
        To load all models properly:
        1. Check that all model files are in the correct directories
        2. Ensure all dependencies are installed with `pip install -r requirements.txt`
        """)

# Input section
st.subheader("Share your thoughts and feelings:")
user_input = st.text_area(
    "What's on your mind?",
    placeholder="Describe how you're feeling or what you're thinking about...",
    height=150,
    help="Share your thoughts, feelings, or concerns. The CBT engine will analyze and provide structured feedback."
)

# Example texts for CBT analysis
example_texts = [
    "I always mess everything up and I'm never going to succeed at anything",
    "Everyone thinks I'm stupid and they're probably right",
    "This project is going to be a complete disaster and ruin my career",
    "I can't handle this stress anymore, it feels overwhelming",
    "I feel like there's no point in trying because I'll just fail again",
    "Nobody really cares about me and I'm just bothering people"
]

selected_example = st.selectbox(
    "Or try one of these examples:",
    ["Select an example..."] + example_texts
)

if selected_example != "Select an example...":
    user_input = selected_example

if st.button('🔍 Analyze with CBT Framework', type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Performing CBT analysis..."):
            # Run CBT analysis
            cbt_result = cbt_engine.analyze(user_input)
        
        # Display results in organized sections
        col1, col2 = st.columns(2)
        
        with col1:
            # Emotional and Intent Analysis
            st.subheader('🎯 Analysis Summary')
            
            emotion_info = st.container()
            with emotion_info:
                emotion = cbt_result.get('emotion', 'Unknown')
                original_emotion = cbt_result.get('original_emotion', None)
                
                if original_emotion:
                    st.write(f"**Detected Emotion:** {emotion} ({original_emotion})")
                else:
                    st.write("**Detected Emotion:**", emotion)
                    
                if cbt_result.get('emotion_score'):
                    st.progress(float(cbt_result['emotion_score']))
                
                st.write("**Identified Intent:**", cbt_result.get('intent', 'Unknown'))
                if cbt_result.get('intent_score'):
                    st.progress(float(cbt_result['intent_score']))
                
                risk_level = cbt_result.get('risk', 'Unknown')
                risk_score = cbt_result.get('risk_score', 0)
                
                if cbt_result.get('escalation'):
                    st.error(f"⚠️ **Risk Level**: {risk_level} (Score: {risk_score:.3f})")
                    st.error("🚨 **SAFETY ALERT**: High risk detected. Please seek immediate professional help.")
                elif risk_level in ['moderate', 'high']:
                    st.warning(f"⚠️ **Risk Level**: {risk_level} (Score: {risk_score:.3f})")
                else:
                    st.success(f"✅ **Risk Level**: {risk_level} (Score: {risk_score:.3f})")
        
        with col2:
            # Cognitive Distortions
            st.subheader('🧩 Cognitive Patterns')
            
            distortions = cbt_result.get('distortions', [])
            if distortions:
                st.write("**Cognitive Distortions Detected:**")
                
                # Check if we have detailed distortion info from the model
                detailed_distortions = st.session_state.get("distortion_details", [])
                
                # If we have detailed distortion info, use that for visualization
                if detailed_distortions:
                    for i, dist in enumerate(detailed_distortions[:3]):
                        distortion_type = dist["distortion_type"]
                        confidence = dist["confidence"] * 100
                        
                        # Use appropriate emoji for each distortion type
                        emoji_map = {
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
                        
                        emoji = emoji_map.get(distortion_type, "🧩")
                        
                        # Display distortion with confidence
                        st.write(f"**{i+1}. {emoji} {distortion_type.title()}**: {confidence:.2f}%")
                        st.progress(dist["confidence"])
                        
                        # Display collapsible explanation and reframing
                        cognitive_wrapper = cbt_engine.config.get("cognitive_wrapper")
                        if cognitive_wrapper:
                            with st.expander(f"Learn about {distortion_type.title()}"):
                                explanation = cognitive_wrapper.get_distortion_explanation(distortion_type)
                                reframing = cognitive_wrapper.get_reframing_suggestion(distortion_type)
                                st.write(f"**What it is**: {explanation}")
                                st.write(f"**How to reframe it**: {reframing}")
                else:
                    # Just list the distortions simply if we don't have detailed info
                    for distortion in distortions:
                        st.write(f"• {distortion.replace('_', ' ').title()}")
            else:
                st.write("✅ No major cognitive distortions detected")
        
        # CBT Interventions Section
        st.subheader('💡 CBT Interventions & Suggestions')
        
        # Create tabs for different intervention types
        tab1, tab2, tab3 = st.tabs(["🔄 Cognitive Reframing", "🎯 Behavioral Experiments", "📝 Clinical Notes"])
        
        with tab1:
            st.write("**Reframing Suggestions:**")
            reframes = cbt_result.get('reframes', [])
            for i, reframe in enumerate(reframes, 1):
                st.write(f"{i}. {reframe}")
        
        with tab2:
            st.write("**Behavioral Suggestions:**")
            behaviors = cbt_result.get('behavioral_suggestions', [])
            for i, behavior in enumerate(behaviors, 1):
                st.write(f"{i}. {behavior}")
        
        with tab3:
            st.write("**Clinical Summary:**")
            notes = cbt_result.get('clinician_notes', [])
            for note in notes:
                st.write(f"• {note}")
        
        # User-facing message
        st.subheader('💬 Supportive Response')
        user_message = cbt_result.get('user_facing', '')
        
        if cbt_result.get('escalation'):
            st.error(user_message)
        else:
            st.info(user_message)
        
        # Raw analysis data (expandable)
        with st.expander("🔧 Technical Analysis Details"):
            st.json(cbt_result)

# Information sections
col1, col2 = st.columns(2)

with col1:
    with st.expander("ℹ️ About CBT Analysis"):
        st.write("""
        **Cognitive Behavioral Therapy (CBT)** is an evidence-based approach that focuses on:
        
        - **Cognitive Patterns**: Identifying unhelpful thinking patterns
        - **Behavioral Interventions**: Suggesting practical actions to improve mood
        - **Reframing**: Developing more balanced, realistic thoughts
        - **Safety Assessment**: Monitoring for risk factors
        
        This tool provides CBT-style analysis using AI models trained on mental health data.
        """)

with col2:
    with st.expander("🛡️ Safety & Disclaimers"):
        st.write("""
        **Important Safety Information:**
        
        - This tool is for educational and supportive purposes only
        - It does not replace professional mental health care
        - If you're experiencing a mental health crisis, please:
          - Call emergency services (911)
          - Contact a crisis helpline
          - Reach out to a mental health professional
        
        **Crisis Resources:**
        - National Suicide Prevention Lifeline: 988
        - Crisis Text Line: Text HOME to 741741
        """)

# Footer
st.markdown("---")
st.caption("CBT Mental Health Analysis Tool | Based on evidence-based Cognitive Behavioral Therapy principles")