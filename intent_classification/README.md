

# 🧠 Intent Classifier 

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
