from typing import Dict, List, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EndpointHandler:
    def __init__(self, path: str = ""):
        self.path = path
        logger.info(f"Initializing intent classification model at path: {path}")

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            logger.error(f"Failed to initialize model/tokenizer from '{path}': {str(e)}")
            raise

    def _determine_task(self):
        raise NotImplementedError("_determine_task is deprecated; this handler only supports intent classification.")

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        inputs = data.get("inputs", "")
        parameters = data.get("parameters", None)
        if not inputs:
            logger.warning("No inputs provided")
            return [{"error": "No inputs provided"}]

        try:
            logger.info("Processing inputs for intent classification")
            result = self.pipeline(inputs, return_all_scores=True, **(parameters or {}))
            # Flatten per original behavior to return a list of label/score pairs
            return [{"label": item["label"], "score": item["score"]} for sublist in result for item in sublist]
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return [{"error": f"Inference failed: {str(e)}"}]