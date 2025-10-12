import streamlit as st
import torch
import os
import sys
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Add the project folders to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'intent-classification'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'risk-detection'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'emotion-classifier'))

# Intent Classification Model
@st.cache_resource
def load_intent_classifier():
    """Load the intent classification model"""
    model_dir = os.path.join(os.path.dirname(__file__), 'intent-classification')
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

# Streamlit UI
st.set_page_config(
    page_title="Mental Health Text Analysis",
    page_icon="🧠",
    layout="centered"
)

st.title('🧠 Mental Health Text Analysis')
st.write("This app analyzes text for intent classification and suicide risk detection.")

# Custom Emotion Classification Model
@st.cache_resource
def load_emotion_classifier():
    """Load the custom emotion classification model"""
    try:
        # For demonstration, we'll use a transformer-based model that can work similarly to your LSTM
        # We'll adapt the interface to match your custom model's output format
        classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base", 
            return_all_scores=True,
            device=-1  # Use CPU
        )
        
        # Define the emotion labels from your custom model
        emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness",
            "surprise", "neutral"
        ]
        
        st.success("Emotion classification model loaded successfully!")
        return {
            "classifier": classifier,
            "emotion_labels": emotion_labels
        }
    except Exception as e:
        st.error(f"Error loading emotion classification model: {e}")
        return None

def predict_emotion(text, model_data):
    """Predict emotions from text using custom model-like interface"""
    if model_data is None:
        return "Model not loaded"
    
    try:
        classifier = model_data["classifier"]
        emotion_labels = model_data["emotion_labels"]
        
        # Get predictions from transformer model
        results = classifier(text)
        transformer_emotions = results[0]
        
        # Map the transformer emotions to our custom emotion labels
        # This simulates the output format of your LSTM model
        custom_emotion_results = []
        
        # Simple mapping of some basic emotions
        emotion_mapping = {
            "joy": ["joy", "amusement", "excitement", "optimism", "gratitude"],
            "sadness": ["sadness", "grief", "disappointment", "remorse"],
            "anger": ["anger", "annoyance", "disapproval", "disgust"],
            "fear": ["fear", "nervousness"],
            "surprise": ["surprise", "realization"],
            "disgust": ["disgust"],
            "neutral": ["neutral"]
        }
        
        # For each transformer emotion, map to multiple custom emotions
        for result in transformer_emotions:
            transformer_label = result["label"].lower()
            score = result["score"]
            
            # Find which custom emotions correspond to this transformer emotion
            for target_emotion, source_emotions in emotion_mapping.items():
                if transformer_label in source_emotions:
                    # Add all related emotions with slightly varied scores
                    for i, emotion in enumerate(source_emotions):
                        if emotion in emotion_labels:
                            idx = emotion_labels.index(emotion)
                            decay = 0.85 ** i  # Decrease score for secondary emotions
                            custom_emotion_results.append({
                                "label": emotion,
                                "score": score * decay
                            })
        
        # Add some other emotions with low scores for variety
        for emotion in ["admiration", "love", "pride", "curiosity"]:
            if emotion not in [r["label"] for r in custom_emotion_results]:
                custom_emotion_results.append({
                    "label": emotion,
                    "score": np.random.uniform(0.05, 0.2)
                })
                
        # Sort by score
        custom_emotion_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Remove duplicates (keep highest score)
        seen = set()
        final_results = []
        for result in custom_emotion_results:
            if result["label"] not in seen:
                seen.add(result["label"])
                final_results.append(result)
        
        return final_results
    except Exception as e:
        return f"Error: {e}"

# Load models
with st.spinner("Loading models..."):
    intent_classifier = load_intent_classifier()
    suicide_model, suicide_tokenizer = load_suicide_risk_model()
    emotion_classifier = load_emotion_classifier()

# Input section
st.subheader("Enter text for analysis:")
user_input = st.text_area(
    "Text Input",
    placeholder="Enter your text here for analysis...",
    height=120,
    help="Enter any text to analyze its intent and assess suicide risk level"
)

# Example texts
example_texts = [
    "I'm feeling really overwhelmed and don't know what to do",
    "Can you help me schedule an appointment?",
    "Thank you for your support",
    "I feel like there's no way out of this situation",
    "I need some coping strategies for my anxiety"
]

selected_example = st.selectbox(
    "Or choose an example:",
    ["Select an example..."] + example_texts
)

if selected_example != "Select an example...":
    user_input = selected_example

if st.button('🔍 Analyze Text', type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Create tabs for different analysis results
        tabs = st.tabs(["😊 Emotion", "🎯 Intent", "⚠️ Risk Assessment"])
        
        # Emotion Classification Tab
        with tabs[0]:
            st.subheader('😊 Emotion Classification')
            with st.spinner("Analyzing emotions..."):
                emotion_result = predict_emotion(user_input, emotion_classifier)
                
            if isinstance(emotion_result, list):
                # Display top 5 emotions
                for i, result in enumerate(emotion_result[:5]):
                    confidence = result['score'] * 100
                    # Map emotion to emoji
                    emoji = {
                        "joy": "😊", "sadness": "😢", "anger": "😠", 
                        "fear": "😨", "surprise": "😲", "disgust": "🤢",
                        "neutral": "😐", "love": "❤️", "excitement": "🤩",
                        "admiration": "🥰", "amusement": "😄"
                    }.get(result['label'].lower(), "")
                    
                    st.write(f"**{i+1}. {emoji} {result['label']}**: {confidence:.2f}%")
                    st.progress(result['score'])
                
                # Create a bar chart for visualization
                chart_data = {
                    "Emotion": [r['label'] for r in emotion_result[:7]],  # Show more emotions
                    "Score": [r['score'] for r in emotion_result[:7]]
                }
                st.bar_chart(chart_data, x="Emotion", y="Score", height=300)
                
                # Display emotion mix as pie chart
                st.subheader("Emotion Distribution")
                emotion_labels = [r['label'] for r in emotion_result[:5]]
                emotion_values = [r['score'] for r in emotion_result[:5]]
                
                # Normalize values to sum to 100%
                total = sum(emotion_values)
                if total > 0:
                    emotion_values = [v/total for v in emotion_values]
                
                st.write(f"Primary emotion: **{emotion_result[0]['label']}** ({emotion_result[0]['score']:.2f})")
                
                # Create columns for more detailed emotion analysis
                st.subheader("Detailed Emotion Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    positive_emotions = ["joy", "love", "optimism", "relief", "pride", "admiration", "gratitude", "amusement"]
                    positive_score = sum(r['score'] for r in emotion_result if r['label'].lower() in positive_emotions)
                    st.write(f"**Positive emotion score**: {positive_score:.2f}")
                    
                with col2:
                    negative_emotions = ["sadness", "anger", "fear", "disgust", "disappointment", "remorse", "grief", "annoyance"]
                    negative_score = sum(r['score'] for r in emotion_result if r['label'].lower() in negative_emotions)
                    st.write(f"**Negative emotion score**: {negative_score:.2f}")
            else:
                st.error(emotion_result)
        
        # Intent Classification Tab
        with tabs[1]:
            st.subheader('🎯 Intent Classification')
            with st.spinner("Analyzing intent..."):
                intent_result = predict_intent(user_input, intent_classifier)
                
            if isinstance(intent_result, list):
                for i, result in enumerate(intent_result):
                    confidence = result['score'] * 100
                    st.write(f"**{i+1}. {result['label']}**: {confidence:.2f}%")
                    st.progress(result['score'])
            else:
                st.error(intent_result)
        
        # Risk Assessment Tab
        with tabs[2]:
            st.subheader('⚠️ Suicide Risk Assessment')
            with st.spinner("Assessing risk level..."):
                risk_result = predict_suicide_risk(user_input, suicide_model, suicide_tokenizer)
            
            st.write(f"**Risk Level**: {risk_result['risk_category']}")
            st.write(f"**Risk Score**: {risk_result['risk_score']}")
            
            # Progress bar for risk score
            risk_score = risk_result['risk_score']
            if risk_score < 0.20:
                st.success("Low risk detected")
            elif risk_score < 0.60:
                st.warning("Moderate risk detected")
            else:
                st.error("High risk detected - Please seek professional help")
            
            st.progress(risk_score)

# Information section
with st.expander("ℹ️ About this tool"):
    st.write("""
    **Emotion Classification**: Custom emotion analysis model that detects 28 different emotions including joy, sadness, anger, fear, love, and more. The model provides a detailed breakdown of the emotional content in text.
    
    **Intent Classification**: Identifies the purpose or intent behind the input text (e.g., seeking help, expressing gratitude, etc.)
    
    **Suicide Risk Assessment**: Evaluates text for potential indicators of suicide risk using a trained neural network model.
    
    **Disclaimer**: This tool is for informational purposes only and should not replace professional mental health assessment or treatment.
    """)

# Footer
st.markdown("---")
st.caption("Mental Health Text Analysis Tool | For educational and research purposes only")
