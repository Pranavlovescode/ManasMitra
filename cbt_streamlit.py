import streamlit as st
import torch
import os
import sys
import json
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np

# Import the CBT engine
from cbt import CBTEngine

# Add the project folders to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'intent_classification'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'risk-detection'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'emotion_classifier'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_distortion'))

# Import model components
from emotion_classifier.emotion_model import LSTMEmotionClassifier, load_vocab, predict_emotion as lstm_predict_emotion, emotion_labels
from cognitive_distortion.cognitive_distortion_model import CognitiveDistortionModel




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
            results = self.classifier(text)
            # Get the top prediction
            if results and len(results[0]) > 0:
                top_result = max(results[0], key=lambda x: x['score'])
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
            # Tokenize the text
            tokenized = self.tokenizer(
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
                score = self.model(input_ids, attention_mask).item()
            
            # Determine risk level for CBT engine
            if score < 0.20:
                risk_level = "low"
            elif score < 0.40:
                risk_level = "low"
            elif score < 0.60:
                risk_level = "moderate"
            elif score < 0.80:
                risk_level = "high"
            else:
                risk_level = "urgent"
            
            return {
                "level": risk_level,
                "score": score
            }
        except Exception as e:
            st.error(f"Error in risk prediction: {e}")
            return {"level": "low", "score": 0.0}


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
        
        try:
            # Load the saved model data (including state dict and parameters)
            model_data = torch.load(model_path, map_location=device)
            
            # Extract model parameters from saved data
            embed_dim = model_data.get('embed_dim', 300)
            hidden_dim = model_data.get('hidden_dim', 256)
            vocab_size = model_data.get('vocab_size', len(vocab))
            num_classes = len(emotion_labels)
            
            # Create model with the correct parameters
            model = LSTMEmotionClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
            
            # Load the actual state dict from the nested structure
            if 'model_state_dict' in model_data:
                model.load_state_dict(model_data['model_state_dict'])
            else:
                # Try loading directly if not nested
                model.load_state_dict(model_data)
        except Exception as e:
            # Fallback to default parameters if loading fails
            st.warning(f"Could not load model weights: {e}. Using default model.")
            vocab_size = len(vocab)
            embed_dim = 300
            hidden_dim = 256
            num_classes = len(emotion_labels)
            model = LSTMEmotionClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
            
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

# Suicide Risk Model Architecture (same as before)
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

@st.cache_resource
def load_suicide_risk_model():
    """Load the suicide risk detection model"""
    try:
        # Load BERT model and tokenizer
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
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

# Cognitive Distortion Model
@st.cache_resource
def load_cognitive_distortion_model():
    """Load the cognitive distortion detection model"""
    try:
        from cognitive_distortion.cognitive_distortion_model import CognitiveDistortionModel
        
        model = CognitiveDistortionModel()
        success = model.load_model()
        if success:
            st.success("Cognitive distortion model loaded successfully!")
            return model
        else:
            st.error("Failed to load cognitive distortion model")
            return None
    except Exception as e:
        st.error(f"Error loading cognitive distortion model: {e}")
        return None

class CognitiveDistortionModelWrapper:
    """Wrapper for cognitive distortion model to work with CBT engine"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, text):
        """CBT engine expects .predict() method"""
        if self.model is None:
            return {"distortion": "unknown", "confidence": 0.0}
        
        try:
            results = self.model.predict(text)
            if "distortions" in results and results["distortions"]:
                # Get the top distortion
                top_distortion = results["distortions"][0]
                return {
                    "distortion": top_distortion["distortion_type"],
                    "score": top_distortion["confidence"]
                }
        except Exception as e:
            st.error(f"Error in cognitive distortion prediction: {e}")
        
        return {"distortion": "unknown", "confidence": 0.0}

@st.cache_resource
def initialize_cbt_engine():
    """Initialize the CBT engine with loaded models"""
    # Load models
    intent_classifier = load_intent_classifier()
    suicide_model, suicide_tokenizer = load_suicide_risk_model()
    emotion_model = load_emotion_classifier()
    cognitive_distortion_model = load_cognitive_distortion_model()
    
    # Create model wrappers
    intent_wrapper = IntentModelWrapper(intent_classifier)
    risk_wrapper = SuicideRiskModelWrapper(suicide_model, suicide_tokenizer)
    
    # Initialize CBT engine
    config = {
        "escalation_levels": ["high", "urgent"],
        "cognitive_distortion_model": cognitive_distortion_model  # Add cognitive distortion model to the config
    }
    
    cbt_engine = CBTEngine(
        emotion_model=emotion_model,
        intent_model=intent_wrapper,
        risk_model=risk_wrapper,
        config=config
    )
    
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
    "I always mess everything up and I'm never going to succeed at anything",  # all-or-nothing thinking
    "Everyone thinks I'm stupid and they're probably right",  # mind reading
    "This project is going to be a complete disaster and ruin my career",  # catastrophizing
    "I can't handle this stress anymore, it feels overwhelming",  # emotional reasoning
    "I feel like there's no point in trying because I'll just fail again",  # fortune telling
    "Nobody really cares about me and I'm just bothering people",  # overgeneralization
    "It's all my fault that the team project failed",  # personalization
    "I should have known better than to try something so difficult"  # should statements
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
                st.write("**Detected Emotion:**", cbt_result.get('emotion', 'Unknown'))
                if cbt_result.get('emotion_score'):
                    st.progress(cbt_result['emotion_score'])
                
                st.write("**Identified Intent:**", cbt_result.get('intent', 'Unknown'))
                if cbt_result.get('intent_score'):
                    st.progress(cbt_result['intent_score'])
                
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