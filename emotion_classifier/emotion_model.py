import torch
import torch.nn as nn
import json
import numpy as np

# Define the model architecture
class LSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix=None):
        super().__init__()
        # Initialize embedding layer
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32), 
                freeze=False
            )
        else:
            # If no pretrained embeddings, initialize randomly
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        # Concatenate final states from both directions
        out = self.fc(torch.cat((h_n[-2], h_n[-1]), dim=1))
        return self.sigmoid(out)

# List of emotion labels
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

# Function to load the vocabulary
def load_vocab(vocab_file='vocab.txt'):
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return None

# Function to load the model
def load_model(model_path='emotion_model.pth', vocab_file='vocab.txt'):
    try:
        # Load vocabulary
        vocab = load_vocab(vocab_file)
        if vocab is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return None, None, device

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        embedding_dim = 300
        hidden_dim = 256
        num_classes = len(emotion_labels)

        model = LSTMEmotionClassifier(
            vocab_size=len(vocab),
            embed_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ).to(device)

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        return model, vocab, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, torch.device("cpu")

# Function to predict emotions from text
def predict_emotion(text, model=None, vocab=None, device=None):
    if model is None or vocab is None:
        try:
            model, vocab, device = load_model()
        except ValueError as e:
            print(f"Error unpacking load_model result: {e}")
            return [{"label": "neutral", "score": 1.0}]

    if model is None or vocab is None:
        return [{"label": "neutral", "score": 1.0}]

    # Tokenize and encode text
    tokens = [vocab.get(word.lower(), vocab['<UNK>']) for word in text.split()]

    # Pad/truncate
    max_len = 50
    if len(tokens) < max_len:
        tokens += [vocab['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]

    # Convert to tensor and predict
    try:
        x = torch.tensor(tokens).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            probs = model(x).cpu().numpy()[0]

        # Format results in the style expected by integrated_cbt
        results = []
        threshold = 0.5
        for i, label in enumerate(emotion_labels):
            if probs[i] >= threshold:
                results.append({"label": label, "score": float(probs[i])})

        # If no emotions above threshold, return neutral
        if not results:
            results.append({"label": "neutral", "score": 1.0})

        # Sort by score (descending)
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results
    except Exception as e:
        print(f"Error in prediction: {e}")
        return [{"label": "neutral", "score": 1.0}]
