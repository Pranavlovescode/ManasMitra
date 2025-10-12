import streamlit as st
import torch
import os
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Add the project folders to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'intent-classification'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'risk-detection'))

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

# Load models
with st.spinner("Loading models..."):
    intent_classifier = load_intent_classifier()
    suicide_model, suicide_tokenizer = load_suicide_risk_model()

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
        # Create two columns for results
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
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
    **Intent Classification**: Identifies the purpose or intent behind the input text (e.g., seeking help, expressing gratitude, etc.)
    
    **Suicide Risk Assessment**: Evaluates text for potential indicators of suicide risk using a trained neural network model.
    
    **Disclaimer**: This tool is for informational purposes only and should not replace professional mental health assessment or treatment.
    """)

# Footer
st.markdown("---")
st.caption("Mental Health Text Analysis Tool | For educational and research purposes only")
