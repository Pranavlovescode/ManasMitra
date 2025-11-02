"""Simple HTTP wrapper around the local Mistral model.

Expose a POST /generate endpoint that accepts JSON {"text": "..."}
and returns {"response": "..."} so the frontend can call it.

This file intentionally keeps the heavy model load in `mistral_model` so
the model is loaded once on import.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from mistral_model import generate_response as model_generate
import logging

app = FastAPI(title="Therapist Bot - Generation API")

# Allow requests from typical frontend origins during development.
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # change this to your frontend origin in production
	allow_credentials=True,
	allow_methods=["GET", "POST", "OPTIONS"],
	allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")


class GenerateRequest(BaseModel):
	text: str


class GenerateResponse(BaseModel):
	response: str


@app.get("/health")
def health():
	"""Health check endpoint."""
	return {"status": "ok"}


@app.post("/chatbot", response_model=GenerateResponse)
def generate(req: GenerateRequest):
	"""Generate a response from the model for the given text."""
	prompt = req.text
	if not prompt or not prompt.strip():
		raise HTTPException(status_code=400, detail="`text` must be a non-empty string")

	try:
		# model_generate is the function in mistral_model.py
		out = model_generate(prompt)
		return {"response": out}
	except Exception as e:
		logger.exception("Error while generating response")
		raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
	# Run with: python generate_response.py
	# Or use: uvicorn chatbot.generate_response:app --host 0.0.0.0 --port 8000 --reload
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)