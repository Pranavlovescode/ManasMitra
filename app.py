# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = FastAPI()
# # uncomment this when you want to use local model
# # MODEL_PATH = "./chatbot/zephyr_remote_model"  # path where you downloaded the model

# # print("Loading tokenizer and model...")
# # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# # model = AutoModelForCausalLM.from_pretrained(
# #     MODEL_PATH,
# #     device_map="auto",
# #     torch_dtype=torch.float16
# # )

# # Using HuggingFace Hub to load the model directly
# MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     device_map="auto",        # uses your GPU
#     torch_dtype=torch.float16
# )

# model.eval()
# print("Model loaded!")

# class PromptRequest(BaseModel):
#     prompt: str
#     max_new_tokens: int = 200
#     temperature: float = 0.7

# @app.post("/ask")
# async def ask(req: PromptRequest):
#     inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=req.max_new_tokens,
#         temperature=req.temperature
#     )
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return {"text": text}





# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="Therapist Chatbot")

# Model setup
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask(request: ChatRequest):
    input_text = request.prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": reply}
print("Model loaded and API is ready.")