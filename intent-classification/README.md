---
license: mit
language:
  - en
tags:
  - intent-classification
  - mental-health
  - transformer
  - conversational-ai
pipeline_tag: text-classification
base_model: distilbert-base-uncased
---

# 🧠 Intent Classifier (MindPadi)

The `intent_classifier` is a transformer-based text classification model trained to detect **user intents** in a mental health support setting. It powers the MindPadi assistant's ability to route conversations to the appropriate modules—like emotional support, scheduling, reflection, or journal analysis—based on the user’s message.

## 📝 Model Overview

- **Model Architecture:** DistilBERT (uncased) + classification head
- **Task:** Intent Classification
- **Classes:** Over 20 intent categories (e.g., `vent`, `gratitude`, `help_request`, `journal_analysis`)
- **Model Size:** ~66M parameters
- **Files:**
  - `config.json`
  - `pytorch_model.bin` or `model.safetensors`
  - `tokenizer_config.json`, `vocab.txt`, `tokenizer.json`
  - `checkpoint-*/` (optional training checkpoints)

## ✅ Intended Use

### ✔️ Use Cases

- Detecting user intent in MindPadi mental health conversations
- Enabling context-specific dialogue flows
- Assisting with journal entry triage and tagging
- Triggering therapy-related tools (e.g., emotion check-ins, PubMed summaries)

### 🚫 Not Intended For

- Multilingual intent classification (English-only)
- Legal or medical diagnosis tasks
- Multi-label classification (currently single-label per input)

## 💡 Example Intents Detected

| Intent             | Description                                   |
| ------------------ | --------------------------------------------- |
| `vent`             | User expressing frustration or emotion freely |
| `help_request`     | Seeking mental health support                 |
| `schedule_session` | Booking a therapy check-in                    |
| `gratitude`        | Showing appreciation for support              |
| `journal_analysis` | Submitting a journal entry for AI feedback    |
| `reflection`       | Talking about personal growth or setbacks     |
| `not_sure`         | Unsure or unclear message from user           |

## 🛠️ Training Details

- **Base Model:** `distilbert-base-uncased`
- **Dataset:** Curated and annotated conversations (`training/datasets/finetuned/intents/`)
- **Script:** `training/train_intent_classifier.py`
- **Preprocessing:**
  - Text normalization (lowercasing, punctuation removal)
  - Label encoding
- **Loss:** CrossEntropyLoss
- **Metrics:** Accuracy, F1-score
- **Tokenizer:** WordPiece (DistilBERT tokenizer)

## 📊 Evaluation

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 91.3% |
| F1-score  | 89.8% |
| Recall@3  | 97.1% |
| Precision | 88.4% |

Evaluation performed on a held-out validation split of MindPadi intent dataset.

## 🔍 Example Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("mindpadi/intent_classifier")
tokenizer = AutoTokenizer.from_pretrained("mindpadi/intent_classifier")

text = "I’m struggling with my emotions today"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

predicted_class = torch.argmax(outputs.logits, dim=1).item()
print("Predicted intent ID:", predicted_class)
```

To map `intent ID → label`, load your label encoder from:

```python
from joblib import load
label_encoder = load("intent_encoder/label_encoder.joblib")
print("Predicted intent:", label_encoder.inverse_transform([predicted_class])[0])
```

## 🔌 Inference Endpoint Example

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/mindpadi/intent_classifier"
headers = {"Authorization": f"Bearer <your-api-token>"}
payload = {"inputs": "Can I book a mental health session?"}

response = requests.post(API_URL, headers=headers, json=payload)
print(response.json())
```

## ⚠️ Limitations

- Not robust to long-form texts (>256 tokens); truncate or summarize input.
- May confuse overlapping intents like `vent` and `help_request`
- False positives possible in vague or sarcastic inputs
- Requires pairing with fallback model (`intent_fallback`) for reliability

## 🔐 Ethical Considerations

- This model is for **supportive routing**, not clinical diagnosis
- Use with user consent and proper data privacy safeguards
- Intent predictions should not override human judgment in sensitive contexts

## 📂 Integration Points

| Location                           | Functionality                                 |
| ---------------------------------- | --------------------------------------------- |
| `app/chatbot/intent_classifier.py` | Main classifier logic                         |
| `app/chatbot/intent_router.py`     | Routes based on predicted intent              |
| `app/utils/embedding_search.py`    | Uses `intent_encoder` for similarity fallback |
| `data/processed_intents.json`      | Annotated intent samples                      |

## 📜 License

MIT License – freely available for commercial and non-commercial use.

## 📬 Contact

- **Team:** MindPadi AI Developers
- **Profile:** [https://huggingface.co/mindpadi](https://huggingface.co/mindpadi)
- **Email:** \[[you@example.com](mailto:you@example.com)]

_Last updated: May 2025_

## 🚀 Run the Streamlit UI

Run a simple local web UI to try the classifier in this folder.

1. Create/activate a virtual environment (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Start the app:

```powershell
streamlit run app.py
```

3. Open the link shown in the console (usually http://localhost:8501). Type a message or pick an example and click "Classify" to see the predicted intent and top-k probabilities.

Notes:

- The app loads the model from the current folder, using the local `config.json`, `model.safetensors`, and tokenizer files.
- If you have a GPU, PyTorch will automatically use it; otherwise CPU is used.
