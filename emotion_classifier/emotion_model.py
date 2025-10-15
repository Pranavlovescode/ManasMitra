import torch
import torch.nn as nn
import numpy as np

# Define the emotion labels
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

class LSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix=None):
        super().__init__()
        # Create embedding layer
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32), 
                freeze=False
            )
        else:
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

def load_vocab(file_path):
    """Load vocabulary from file"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    # If file exists, load it
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word = line.strip()
                if word:
                    vocab[word] = i + 2
    except FileNotFoundError:
        # Create a minimal vocab if file not found
        common_words = ["i", "you", "the", "a", "and", "to", "is", "in", "it", "that"]
        for i, word in enumerate(common_words):
            vocab[word] = i + 2
    return vocab

def encode_text(tokens, vocab, max_len=40):
    """Convert tokens to IDs using vocabulary"""
    ids = [vocab.get(t.lower(), 1) for t in tokens[:max_len]]  # 1 is <UNK>
    ids += [0] * (max_len - len(ids))  # 0 is <PAD>
    return ids

def predict_emotion(text, model, vocab, device):
    """Predict emotions from text using the LSTM model"""
    model.eval()
    
    # Tokenize (simple splitting)
    tokens = text.lower().split()
    
    # Encode tokens to IDs
    encoded = encode_text(tokens, vocab)
    
    # Convert to tensor
    x = torch.tensor(encoded).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        probs = output.squeeze(0).cpu().numpy()
    
    # Get predictions (apply threshold)
    threshold = 0.5
    predictions = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            predictions.append({"label": emotion_labels[i], "score": float(prob)})
    
    # Sort by probability (highest first)
    predictions.sort(key=lambda x: x["score"], reverse=True)
    
    return predictions