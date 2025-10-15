# Custom Emotion Classifier Implementation Guide

This guide explains how to implement your custom LSTM-based emotion classifier from the Jupyter notebook into the Streamlit app.

## Steps to Implement Your Custom Model

1. **Train and Save the Model**

   In your notebook, after training the model, add this code to save it:

   ```python
   # Save the trained model
   torch.save(model.state_dict(), "emotion_model.pth")
   
   # Save the vocabulary
   with open("vocab.txt", "w") as f:
       for word in vocab:
           if word != "<PAD>" and word != "<UNK>":
               f.write(f"{word}\n")
   ```

2. **Create a Python Module for the Model**

   Place the `emotion_model.py` file in the `emotion-classifier` directory with the following content:

   ```python
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
       with open(file_path, 'r', encoding='utf-8') as f:
           for i, line in enumerate(f):
               word = line.strip()
               if word:
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
       encoded = encode_text(tokens, vocab, max_len=40)
       
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
           if i < len(emotion_labels):
               if prob >= threshold:
                   predictions.append({"label": emotion_labels[i], "score": float(prob)})
       
       # Sort by probability (highest first)
       predictions.sort(key=lambda x: x["score"], reverse=True)
       
       return predictions
   ```

3. **Update `app_streamlit.py` to use your custom model**

   Replace the current emotion classifier implementation with this:

   ```python
   # Custom Emotion Classification Model
   @st.cache_resource
   def load_emotion_classifier():
       """Load the custom emotion classification model"""
       try:
           # Import the custom model
           from emotion_classifier.emotion_model import LSTMEmotionClassifier, load_vocab, predict_emotion as model_predict
           
           # Device configuration
           device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           
           # Model parameters
           vocab_path = os.path.join(os.path.dirname(__file__), 'emotion-classifier', 'vocab.txt')
           model_path = os.path.join(os.path.dirname(__file__), 'emotion-classifier', 'emotion_model.pth')
           
           # Load vocabulary
           vocab = load_vocab(vocab_path)
           
           # Create model with the right parameters
           model = LSTMEmotionClassifier(
               vocab_size=len(vocab),
               embed_dim=300,  # Make sure this matches your trained model
               hidden_dim=256, # Make sure this matches your trained model
               num_classes=28  # Number of emotions
           )
           
           # Load the trained weights
           model.load_state_dict(torch.load(model_path, map_location=device))
           model.to(device)
           model.eval()
           
           st.success("Emotion classification model loaded successfully!")
           return {
               "model": model,
               "vocab": vocab,
               "device": device,
               "predict_function": model_predict
           }
           
       except Exception as e:
           st.error(f"Error loading emotion classification model: {e}")
           return None
   
   def predict_emotion(text, model_data):
       """Predict emotions from text using custom LSTM model"""
       if model_data is None:
           return "Model not loaded"
       
       try:
           # Unpack model data
           model = model_data["model"]
           vocab = model_data["vocab"]
           device = model_data["device"]
           predict_function = model_data["predict_function"]
           
           # Use the model's prediction function
           predictions = predict_function(text, model, vocab, device)
           
           return predictions
       except Exception as e:
           return f"Error: {e}"
   ```

4. **Install Required Dependencies**

   Make sure PyTorch and any other dependencies are installed:

   ```bash
   pip install torch numpy transformers
   ```

## Important Notes

1. Make sure the model parameters (vocab_size, embed_dim, hidden_dim, num_classes) match those used in your trained model.
2. The embedding matrix is not loaded in the implementation above. If you want to use pre-trained embeddings, you'll need to save and load them separately.
3. This implementation uses a simple tokenization approach (just splitting on whitespace). For better results, consider using a more sophisticated tokenizer.

## Testing

After implementing these changes, test your model with various inputs to ensure it's working correctly. The model should identify multiple emotions in text and rank them by confidence score.