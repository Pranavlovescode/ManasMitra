import os
import torch
import shutil
from transformers import AutoTokenizer, AutoModel

def setup_model_files():
    print("Setting up model files for integrated CBT application...")
    
    # Create directories if they don't exist
    os.makedirs('emotion-classifier', exist_ok=True)
    
    # Check if emotion model vocabulary file exists
    if not os.path.exists('emotion-classifier/vocab.txt'):
        print("Creating placeholder vocab.txt for emotion classifier...")
        # Create a basic vocabulary file with common words
        common_words = ["<PAD>", "<UNK>", "i", "am", "feel", "sad", "happy", "anxious", "angry", 
                       "depressed", "worried", "scared", "excited", "hopeful", "hopeless",
                       "the", "and", "to", "a", "of", "is", "in", "it", "that", "you", "he",
                       "she", "they", "we", "not", "for", "on", "with", "as", "this", "but",
                       "have", "are", "at", "be", "by", "from", "was", "were", "will", "would"]
        
        with open('emotion-classifier/vocab.txt', 'w', encoding='utf-8') as f:
            for word in common_words:
                f.write(f"{word}\n")
        
        print("Created basic vocabulary file.")
    else:
        print("Emotion classifier vocabulary file already exists.")
    
    # Download BERT model if not already present for risk detection
    risk_model_dir = os.path.join('risk-detection', 'bert-base-uncased')
    if not os.path.exists(risk_model_dir):
        print("Downloading BERT model for risk detection...")
        try:
            # This will download the model to cache
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncached", cache_dir=risk_model_dir)
            model = AutoModel.from_pretrained("bert-base-uncached", cache_dir=risk_model_dir)
            print("BERT model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading BERT model: {e}")
            print("You'll need to manually download the BERT model or ensure internet connectivity.")
    
    print("\nSetup complete! You can now run the integrated CBT application with:")
    print("streamlit run integrated_cbt_streamlit.py")

if __name__ == "__main__":
    setup_model_files()