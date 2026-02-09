# Mental Health Text Analysis - Streamlit App

This Streamlit application combines **Intent Classification** and **Suicide Risk Detection** models to analyze text input from users.

## Features

- **Intent Classification**: Identifies the purpose or intent behind input text (e.g., seeking help, expressing gratitude, etc.)
- **Suicide Risk Assessment**: Evaluates text for potential indicators of suicide risk using a trained neural network model
- **User-friendly Interface**: Clean Streamlit interface with example texts and visual feedback

## Installation & Setup

### Method 1: Using Batch File (Windows)

1. Simply double-click `run_app.bat` - it will install all dependencies and start the app

### Method 2: Using PowerShell (Windows)

1. Open PowerShell in the project directory
2. Run: `.\run_app.ps1`

### Method 3: Manual Installation

1. Install Python requirements:

```bash
pip install -r streamlit_requirements.txt
pip install -r intent-classification/requirements.txt
pip install -r risk-detection/requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app_streamlit.py
```

## Usage

1. **Start the App**: The app will load both models (may take a moment on first run)
2. **Enter Text**: Type or paste text in the text area, or select from example texts
3. **Analyze**: Click "🔍 Analyze Text" to get results
4. **View Results**:
   - **Left Column**: Intent classification with confidence scores
   - **Right Column**: Suicide risk assessment with risk level and score

## Model Information

### Intent Classification Model

- Located in: `intent-classification/`
- Uses Hugging Face transformers
- Provides multi-class intent prediction with confidence scores

### Suicide Risk Detection Model

- Located in: `risk-detection/`
- Custom neural network built on BERT embeddings
- Model file: `suicide_model.pth`
- Outputs risk scores from 0-1 with categorical levels:
  - 🟢 Minimal Risk (< 0.20)
  - 🟡 Low Risk (0.20-0.39)
  - 🟠 Moderate Risk (0.40-0.59)
  - 🔴 High Risk (0.60-0.79)
  - ⚠️ Severe Risk (≥ 0.80)

## Files Structure

```
ManasMitra/
├── app_streamlit.py              # Main Streamlit application
├── streamlit_requirements.txt    # Core dependencies
├── run_app.bat                  # Windows batch installer/runner
├── run_app.ps1                  # PowerShell installer/runner
├── intent-classification/       # Intent classification model
│   ├── app.py
│   ├── handler.py
│   ├── requirements.txt
│   └── [model files...]
└── risk-detection/             # Suicide risk detection model
    ├── neural_network_suicide.py
    ├── suicide_model.pth
    ├── requirements.txt
    └── [other model files...]
```

## Requirements

- Python 3.8+
- Dependencies listed in:
  - `streamlit_requirements.txt`
  - `intent-classification/requirements.txt`
  - `risk-detection/requirements.txt`

## Disclaimer

⚠️ **Important**: This tool is for educational and research purposes only. It should not replace professional mental health assessment, diagnosis, or treatment. If you or someone you know is experiencing mental health concerns or suicidal thoughts, please seek help from qualified mental health professionals or contact emergency services.

## Troubleshooting

1. **Model Loading Errors**: Ensure all model files are present in their respective directories
2. **Package Conflicts**: Try creating a fresh virtual environment
3. **Memory Issues**: The models require sufficient RAM; close other applications if needed
4. **Port Issues**: If port 8501 is in use, Streamlit will automatically use the next available port
