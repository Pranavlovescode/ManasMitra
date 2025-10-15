import torch
import numpy as np
from emotion_model import LSTMEmotionClassifier, emotion_labels

# Create and save a dummy model for testing
def create_and_save_model():
    vocab_size = 20000 + 2  # +2 for <PAD> and <UNK>
    embed_dim = 300
    hidden_dim = 256
    num_classes = len(emotion_labels)

    # Create model
    model = LSTMEmotionClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Save model
    # torch.save(model.state_dict(), "emotion_model.pth")
    

    # Save vocabulary (create a small example vocabulary)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    common_words = ["i", "you", "the", "a", "and", "to", "is", "in", "it", "that", "this", 
                    "have", "for", "not", "on", "with", "he", "she", "as", "do", "at", "but",
                    "his", "her", "by", "from", "they", "we", "say", "will", "happy", "sad",
                    "angry", "fear", "surprise", "disgust", "love", "hate", "good", "bad"]
    
    for i, word in enumerate(common_words):
        vocab[word] = i + 2

    torch.save({
    "model_state_dict": model.state_dict(),
    "embed_dim": 300,
    "hidden_dim": 256,
    "vocab_size": len(vocab),}, "emotion_model.pth")
    print("Model saved to emotion_model.pth")
        
    # Save vocabulary
    with open("vocab.txt", "w") as f:
        for word in vocab:
            if word != "<PAD>" and word != "<UNK>":
                f.write(f"{word}\n")
    
    print("Vocabulary saved to vocab.txt")

if __name__ == "__main__":
    create_and_save_model()