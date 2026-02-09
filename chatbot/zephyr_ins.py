# Run this in Python terminal
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="HuggingFaceH4/zephyr-7b-beta")
print("Model downloaded to:", model_path)
