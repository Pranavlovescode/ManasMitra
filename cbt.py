# CBT is evidence-based concept. CBT stands for Cognitive Behavioral Therapy.
# CBT is a type of psychotherapeutic treatment that helps individuals understand the thoughts and feelings that influence their behaviors.
# CBT is commonly used to treat a wide range of disorders, including phobias, addictions, depression, and anxiety.
# CBT is generally short-term and focused on helping clients deal with a very specific problem.
# CBT focuses on relationship between thoughts, feelings, beliefs and behaviors.
# negative thoughts and feelings are recognized and challenged and replaced with more positive and realistic ones.
# CBT uses problem solving techniques like face the fear, positive reinforcement, self-regulation, relaxation techniques etc.


"""CBT engine: analyze user text using emotion, intent and risk models and produce
CBT-style structured output: identified emotions/intents, cognitive distortions,
reframing suggestions, behavioral experiments, and safety escalation advice.

This module provides:
- CBTEngine: main class that coordinates models and heuristics
- small model-call wrapper to accept a variety of model interfaces
- lightweight cognitive-distortion heuristics and reframing templates
- demo stubs so you can run a quick local test

Model contract (expected): each model should accept a single text input and
return a dict-like result. CBTEngine will attempt `.predict(text)`, `model(text)`,
or `model.analyze(text)` in that order. Example expected partial outputs:
 - emotion_model -> {"emotion": "sad", "score": 0.9, ...}
 - intent_model  -> {"intent": "avoidance", "confidence": 0.8, ...}
 - risk_model    -> {"level": "low"} or {"score": 0.05}

This file intentionally uses simple, deterministic heuristics so it can run
locally without external dependencies. Replace the stub models with your
trained models by passing them into `CBTEngine(...)`.
"""

from typing import Any, Dict, Optional, Sequence
import re
import json
import torch

from intent_classification.app import load_classifier
from emotion_classifier.emotion_model import LSTMEmotionClassifier, load_vocab, predict_emotion, emotion_labels
from cognitive_distortion.cognitive_distortion_model import CognitiveDistortionModel


class CBTEngine:
	"""Core CBT engine coordinating the models and applying CBT heuristics.

	Inputs: user text string
	Outputs: dict with keys: text, emotion, intent, risk, distortions,
			 reframes, behavioral_suggestions, escalation, clinician_notes
	"""

	def __init__(self, emotion_model: Any, intent_model: Any, risk_model: Any, config: Optional[Dict] = None):
		self.emotion_model = emotion_model
		self.intent_model = intent_model
		self.risk_model = risk_model
		self.config = config or {}

		# thresholds / simple configuration
		self.risk_escalation_levels = set(self.config.get("escalation_levels", ["high", "urgent"]))

		# simple lists for heuristics
		self._distortion_patterns = {
			"all_or_nothing": [r"always", r"never", r"everyone", r"nobody"],
			"catastrophizing": [r"worst", r"disaster", r"ruined", r"can't handle"],
			"mind_reading": [r"they think", r"they'll think", r"they must think"],
			"overgeneralization": [r"always", r"every time", r"everybody"],
			"personalization": [r"it's my fault", r"my fault", r"I caused"],
		}

	def _call_model(self, model: Any, text: str) -> Dict:
		"""Try several common call patterns for injected models."""
		if model is None:
			return {}
		try:
			if hasattr(model, "predict"):
				return model.predict(text) or {}
		except Exception:
			pass
		try:
			if callable(model):
				return model(text) or {}
		except Exception:
			pass
		try:
			if hasattr(model, "analyze"):
				return model.analyze(text) or {}
		except Exception:
			pass
		# last resort: return empty dict
		return {}

	def preprocess(self, text: str) -> str:
		text = text.strip()
		text = re.sub(r"\s+", " ", text)
		return text

	def detect_distortions(self, text: str) -> Sequence[str]:
		text_l = text.lower()
		found = []
		for name, pats in self._distortion_patterns.items():
			for p in pats:
				if re.search(r"\b" + p + r"\b", text_l):
					found.append(name)
					break
		return found

	def generate_reframes(self, text: str, distortions: Sequence[str]) -> Sequence[str]:
		reframes = []
		for d in distortions:
			if d == "all_or_nothing":
				reframes.append("Is it really always or never? Find specific exceptions that disprove the extreme thought.")
			elif d == "catastrophizing":
				reframes.append("What's the realistic worst-case and how likely is it? Consider smaller, testable outcomes.")
			elif d == "mind_reading":
				reframes.append("Do you have direct evidence for what others think? What's an alternative, less-assuming view?")
			elif d == "overgeneralization":
				reframes.append("Can you find a specific example that contradicts the generalization?")
			elif d == "personalization":
				reframes.append("Are you overlooking external factors? What evidence supports other causes?")
			else:
				reframes.append("Try to test this thought: what would you tell a friend in the same situation?")
		# fall back if none found
		if not reframes:
			reframes.append("Try to label the thought, look for evidence for/against it, and generate a balanced alternative.")
		return reframes

	def behavioral_suggestions(self, emotion: Optional[str], intent: Optional[str]) -> Sequence[str]:
		suggestions = []
		if emotion:
			e = emotion.lower()
			if "sad" in e or "depress" in e:
				suggestions.append("Schedule a small activity you can finish in 10 - 20 minutes (walk, call a friend, tidy one corner)")
			if "anx" in e or "fear" in e or "panic" in e:
				suggestions.append("Try a grounding exercise: 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste")
			if "anger" in e:
				suggestions.append("Use a 5-minute cool-down routine: deep breathing and 2-minute walk before responding")
		if intent:
			i = intent.lower()
			if "avoidance" in i:
				suggestions.append("Break the feared task into a 2-minute step and gradually expand exposure")
			if "seek_help" in i or "support" in i:
				suggestions.append("Consider reaching out to a trusted person and describe one specific need (e.g., talk, company)")
		if not suggestions:
			suggestions.append("Try a short self-compassion exercise: acknowledge the difficulty and name one small next step")
		return suggestions

	def analyze(self, text: str) -> Dict[str, Any]:
		"""Run the pipeline: preprocess -> call models -> heuristics -> generate output."""
		out: Dict[str, Any] = {"text": text}
		text = self.preprocess(text)
		out["normalized_text"] = text

		# call models
		emotion_res = self._call_model(self.emotion_model, text)
		intent_res = self._call_model(self.intent_model, text)
		risk_res = self._call_model(self.risk_model, text)

		# normalize expected keys
		out["emotion"] = emotion_res.get("emotion") or emotion_res.get("label") or emotion_res.get("pred")
		out["emotion_score"] = emotion_res.get("score") or emotion_res.get("confidence")
		out["intent"] = intent_res.get("intent") or intent_res.get("label") or intent_res.get("pred")
		out["intent_score"] = intent_res.get("confidence") or intent_res.get("score")
		out["risk"] = risk_res.get("level") or risk_res.get("risk") or risk_res.get("label")
		out["risk_score"] = risk_res.get("score") or risk_res.get("severity")

		# safety escalation
		escalation = False
		if isinstance(out["risk"], str) and out["risk"].lower() in self.risk_escalation_levels:
			escalation = True
		if isinstance(out["risk_score"], (int, float)) and out["risk_score"] >= 0.7:
			escalation = True
		out["escalation"] = escalation

		# detection + reframing
		distortions = self.detect_distortions(text)
		out["distortions"] = distortions
		out["reframes"] = self.generate_reframes(text, distortions)

		# behavioral suggestions
		out["behavioral_suggestions"] = self.behavioral_suggestions(out.get("emotion") or "", out.get("intent") or "")

		# clinician-style summary and recommended next steps
		notes = []
		notes.append(f"Emotion: {out.get('emotion')}, intent: {out.get('intent')}, risk: {out.get('risk')}")
		if distortions:
			notes.append(f"Cognitive distortions detected: {', '.join(distortions)}")
		notes.append("Suggested CBT steps: 1) Identify thought, 2) Examine evidence, 3) Generate alternative, 4) Behavioral experiment")
		if escalation:
			notes.append("Safety escalation recommended , follow local crisis protocol and provide immediate resources.")
		out["clinician_notes"] = notes

		# friendly user-facing message
		user_msg = []
		user_msg.append("I hear you, thank you for sharing. Here are a few ideas you might try:")
		user_msg.extend(out["reframes"][:2])
		user_msg.extend(out["behavioral_suggestions"][:2])
		if escalation:
			user_msg.append("I'm detecting possible high-risk content. If you're in immediate danger, please call emergency services or a crisis line in your area.")
		out["user_facing"] = "\n".join(user_msg)

		return out


# --- Demo / stub models for quick local testing ---
class DummyEmotionModel:
	def predict(self, text: str) -> Dict:
		t = text.lower()
		if any(w in t for w in ["sad", "depress", "hopeless"]):
			return {"emotion": "sad", "score": 0.9}
		if any(w in t for w in ["anx", "scared", "worried", "panic"]):
			return {"emotion": "anxious", "score": 0.92}
		return {"emotion": "neutral", "score": 0.2}


class DummyIntentModel:
	def predict(self, text: str) -> Dict:
		t = text.lower()
		if any(w in t for w in ["avoid", "can't go", "stay home"]):
			return {"intent": "avoidance", "confidence": 0.8}
		if any(w in t for w in ["help", "talk to", "see someone"]):
			return {"intent": "seek_help", "confidence": 0.9}
		return {"intent": "none", "confidence": 0.3}


class DummyRiskModel:
	def predict(self, text: str) -> Dict:
		t = text.lower()
		if any(w in t for w in ["kill myself", "end my life", "suicide"]):
			return {"level": "high", "score": 0.99}
		if any(w in t for w in ["hurt myself", "self harm"]):
			return {"level": "moderate", "score": 0.6}
		return {"level": "low", "score": 0.1}
	
class LoadEmotionModel:
	# import sys
	# import os

	# Add the emotion-classifier folder to the path
	# sys.path.append(os.path.join(os.path.dirname(__file__), "emotion-classifier"))

	# from emotion_model import LSTMEmotionClassifier, load_vocab, predict_emotion, emotion_labels

	def __init__(self, model_path, vocab_path, device='cpu'):
		self.device = torch.device(device)
		self.vocab = load_vocab(vocab_path)
		# Load model checkpoint
		checkpoint = torch.load(model_path, map_location=self.device)
		# Reconstruct model architecture
		vocab_size = len(self.vocab)
		embed_dim = checkpoint.get("embed_dim", 100)
		hidden_dim = checkpoint.get("hidden_dim", 128)
		num_classes = len(emotion_labels)
		self.model = LSTMEmotionClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
		# self.model.load_state_dict(checkpoint["model_state_dict"])
		state_dict = torch.load(model_path, map_location=self.device)
		self.model.load_state_dict(state_dict, strict= False)
		self.model.to(self.device)
		print("Emotion model loaded.")
	# def predict(self, text):
	# 	return predict_emotion(text, self.model, self.vocab, self.device)
	def predict(self, text):
		preds = predict_emotion(text, self.model, self.vocab, self.device)
		if preds:
			top = sorted(preds, key=lambda x: x["score"], reverse=True)[0]
			return {"emotion": top["label"], "score": top["score"]}
		return {"emotion": "neutral", "score": 0.0}
	

class LoadCognitiveDistortionModel:
	def __init__(self):
		self.model     = CognitiveDistortionModel()
		if self.model.model is not None:
			print("Cognitive Distortion model loaded.")
		else:
			print("Cognitive Distortion model failed to load.")

	def predict(self, text):
		if self.model.model is None:
			return {}
		preds = self.model.predict(text)
		if preds and isinstance(preds, list):
			top = sorted(preds, key=lambda x: x["confidence"], reverse=True)[0]
			return {"distortion": top["distortion_type"], "score": top["confidence"]}
		return {}


if __name__ == "__main__":
	# quick demo that reads a line from stdin and prints JSON
	import sys

	demo_text = None
	if len(sys.argv) > 1:
		demo_text = " ".join(sys.argv[1:])
	else:
		print("Enter a short description of how you are feeling (single line):")
		try:
			demo_text = input().strip()
		except EOFError:
			demo_text = "I feel hopeless and I always mess up everything"

	emomodel=LoadEmotionModel("emotion_classifier/emotion_model.pth", "emotion_classifier/vocab.txt", device='cpu')
	# cogdismodel = LoadCognitiveDistortionModel("cognitive_distortion/cognitive_distortion_model.pkl")
	# intent_pipeline, intent_labels, intent_meta = load_classifier("intent_classification")

	engine = CBTEngine(emomodel, DummyIntentModel(), DummyRiskModel())
	result = engine.analyze(demo_text)
	print(json.dumps(result, indent=2))


