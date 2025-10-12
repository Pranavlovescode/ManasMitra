# CBT Mental Health Analysis - Streamlit App

This advanced Streamlit application integrates **Cognitive Behavioral Therapy (CBT) principles** with AI models to provide comprehensive mental health text analysis. It combines intent classification, suicide risk detection, and CBT-based interventions in a single, user-friendly interface.

## 🌟 Features

### Core Analysis Components

- **🎯 Intent Classification**: Identifies the purpose behind user input
- **⚠️ Suicide Risk Assessment**: Evaluates text for potential risk indicators
- **😊 Emotion Detection**: Identifies emotional states (currently rule-based)
- **🧩 Cognitive Distortion Detection**: Identifies unhelpful thinking patterns

### CBT Interventions

- **🔄 Cognitive Reframing**: Structured suggestions for challenging negative thoughts
- **🎯 Behavioral Experiments**: Practical actions to improve mood and functioning
- **📝 Clinical Notes**: Professional-style summaries for clinicians
- **💬 Supportive Responses**: User-friendly therapeutic feedback

### Safety Features

- **🚨 Crisis Detection**: Automatic escalation for high-risk content
- **🛡️ Safety Resources**: Built-in crisis helpline information
- **⚠️ Professional Disclaimers**: Clear guidance on tool limitations

## 🚀 Quick Start

### Method 1: One-Click Setup (Windows)

```bash
# Double-click this file to install and run
run_cbt_app.bat
```

### Method 2: PowerShell (Windows)

```powershell
.\run_cbt_app.ps1
```

### Method 3: Manual Installation

```bash
# Install dependencies
pip install streamlit>=1.33.0 torch>=2.0.0 transformers>=4.49.0
pip install -r intent-classification/requirements.txt
pip install -r risk-detection/requirements.txt

# Run the app
streamlit run cbt_streamlit.py
```

## 🧠 CBT Framework Integration

This app implements core CBT principles:

### Cognitive Component

- **Thought Identification**: Analyzes text for cognitive patterns
- **Distortion Detection**: Identifies common cognitive distortions:
  - All-or-nothing thinking
  - Catastrophizing
  - Mind reading
  - Overgeneralization
  - Personalization

### Behavioral Component

- **Activity Suggestions**: Recommends specific behavioral interventions
- **Grounding Techniques**: Provides coping strategies based on detected emotions
- **Exposure Exercises**: Suggests gradual exposure for avoidance patterns

### Safety Integration

- **Risk Assessment**: Continuous monitoring for safety concerns
- **Crisis Protocols**: Automatic escalation and resource provision
- **Professional Guidance**: Clear recommendations for professional help

## 📊 How It Works

1. **Text Input**: User enters thoughts, feelings, or concerns
2. **AI Analysis**: Three models analyze the input:
   - Intent classification model (Hugging Face transformer)
   - Suicide risk model (Custom BERT-based neural network)
   - Emotion detection (Rule-based with patterns)
3. **CBT Processing**: The CBT engine processes results to:
   - Identify cognitive distortions
   - Generate reframing suggestions
   - Recommend behavioral interventions
   - Assess safety and escalation needs
4. **Structured Output**: Results displayed in organized, therapeutic format

## 🔧 Technical Architecture

### Model Integration

```python
# CBT Engine coordinates all components
CBTEngine(
    emotion_model=EmotionModelWrapper(),
    intent_model=IntentModelWrapper(),
    risk_model=SuicideRiskModelWrapper()
)
```

### Model Wrappers

- **IntentModelWrapper**: Adapts Hugging Face pipeline for CBT engine
- **SuicideRiskModelWrapper**: Adapts PyTorch model for CBT engine
- **EmotionModelStub**: Rule-based emotion detection (expandable)

### Key Files

- **`cbt_streamlit.py`**: Main Streamlit application
- **`cbt.py`**: Core CBT engine with therapeutic logic
- **`intent-classification/`**: Pre-trained intent classification model
- **`risk-detection/`**: Suicide risk detection model and architecture

## 🎨 User Interface

### Main Analysis View

- **Wide Layout**: Organized in columns and tabs
- **Real-time Feedback**: Progress bars and status indicators
- **Safety Alerts**: Prominent display of high-risk content
- **Example Texts**: Pre-loaded examples for testing

### Results Organization

- **Analysis Summary**: Emotion, intent, and risk overview
- **Cognitive Patterns**: Detected distortions with explanations
- **CBT Interventions**: Tabbed view of reframing, behavioral, and clinical notes
- **Supportive Response**: User-friendly therapeutic message

## ⚠️ Safety & Ethics

### Built-in Safeguards

- **Crisis Detection**: Automatic identification of high-risk content
- **Professional Disclaimers**: Clear limitations and recommendations
- **Resource Provision**: Crisis helplines and emergency contacts
- **Escalation Protocols**: Structured response to safety concerns

### Ethical Considerations

- **Educational Purpose**: Designed for learning and support, not diagnosis
- **Professional Boundaries**: Clear guidance on seeking professional help
- **Privacy Notice**: Information about data handling and privacy
- **Cultural Sensitivity**: Acknowledges diverse backgrounds and experiences

## 🔮 Future Enhancements

### Model Improvements

- **Advanced Emotion Model**: Replace rule-based with trained classifier
- **Multilingual Support**: Text translation and analysis
- **Personalization**: User-specific CBT intervention preferences
- **Progress Tracking**: Session-based progress monitoring

### CBT Features

- **Thought Records**: Digital CBT worksheet integration
- **Mood Tracking**: Visual mood and thought pattern tracking
- **Homework Assignments**: Personalized CBT exercises
- **Therapist Integration**: Tools for mental health professionals

## 📝 Example Usage

### Input:

> "I always mess everything up and I'm never going to succeed at anything"

### Output:

- **Emotion**: Sad (0.8 confidence)
- **Intent**: Self-criticism (0.9 confidence)
- **Risk**: Low (0.1 score)
- **Distortions**: All-or-nothing thinking, Overgeneralization
- **Reframing**: "Is it really always or never? Find specific exceptions..."
- **Behavioral**: "Schedule a small activity you can finish in 10-20 minutes..."

## 🆘 Crisis Resources

If you're experiencing a mental health crisis:

- **Emergency**: Call 911 or local emergency services
- **National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741
- **International**: Visit findahelpline.com for local resources

## 📄 License & Disclaimer

This tool is for educational and research purposes only. It does not replace professional mental health assessment, diagnosis, or treatment. Always consult qualified mental health professionals for clinical needs.

---

**Mental Health CBT Analysis Tool** | Powered by AI and evidence-based CBT principles
