# mistral_model.py

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

# Use an absolute local path and prefer local files to avoid HF repo id validation
MODEL_PATH = Path(__file__).parent.joinpath("mistral_local").resolve().as_posix()

print("🔹 Initializing Mistral model loader...")
print(f"📁 Model path: {MODEL_PATH}")

# Try GPU setup first
use_cuda = torch.cuda.is_available()
print(f"🧩 CUDA available: {use_cuda}")

bnb_config = None
if use_cuda:
    try:
        from bitsandbytes import __version__ as bnb_version
        print(f"⚙️ bitsandbytes version: {bnb_version}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    except Exception as e:
        print(f"⚠️ bitsandbytes not available or failed to load: {e}")
        bnb_config = None

# ---------------------- MODEL LOAD LOGIC ----------------------
# load tokenizer (prefer local files, fallback to normal from_pretrained)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def safe_load_model():
    """
    Automatically selects the best strategy for your hardware.
    """
    try:
        if use_cuda and bnb_config is not None:
            print("🚀 Attempting GPU 4-bit quantized load...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto"
            )
            print("✅ Loaded successfully with 4-bit quantization on GPU.")
            return model

        elif use_cuda:
            print("🚀 Loading model on GPU (float16, no quantization)...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map={"": "cuda"},
                torch_dtype=torch.float16
            )
            print("✅ Loaded model on GPU (float16).")
            return model

        else:
            print("⚙️ CUDA not available — loading model on CPU (slow mode).")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map={"": "cpu"},
                torch_dtype=torch.float32
            )
            print("✅ Model loaded on CPU successfully.")
            return model

    except RuntimeError as e:
        print(f"❌ GPU load failed due to: {e}")
        print("🔄 Retrying with CPU fallback...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map={"": "cpu"},
            torch_dtype=torch.float32
        )
        print("✅ Model loaded on CPU fallback.")
        return model


model = safe_load_model()
model.eval()

device = next(model.parameters()).device
print(f"✅ Mistral model loaded successfully on device: {device}")

# ---------------------- RESPONSE GENERATION ----------------------

def generate_response(prompt: str, max_new_tokens: int = 256):
    """
    Generate a natural language response from the Mistral model.
    """
    # working code fall back here if error occurs
    # try:
    #     inputs = tokenizer(prompt, return_tensors="pt").to(device)

    #     with torch.no_grad():
    #         outputs = model.generate(
    #             **inputs,
    #             max_new_tokens=max_new_tokens,
    #             temperature=0.7,
    #             top_p=0.9,
    #             pad_token_id=tokenizer.eos_token_id
    #         )

    #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return response

    # except Exception as e:
    #     return f"⚠️ Error generating response: {str(e)}"


    try:
        batch = tokenizer(prompt, return_tensors="pt")
        # move tensors to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        # Only decode the generated tokens (skip the input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response

    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"


# ---------------------- TEST ----------------------
if __name__ == "__main__":
    # test_prompt = "Hello, how are you feeling today?"
    test_prompt = "I have been feeling very anxious lately and I don't know why. Can you help me understand my feelings?"
    print("\n🧠 Testing model response...\n")
    response = generate_response(test_prompt)
    print("👤 User:", test_prompt)
    print("🤖 Therapist Bot:", response)
