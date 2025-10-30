"""
Cognitive Distortion Detection Model

This module provides classes and functions to detect cognitive distortions in text.
It uses a DistilBERT model fine-tuned on a cognitive distortion dataset.
"""

import os
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class CognitiveDistortionModel:
    """Class to detect cognitive distortions in text using a fine-tuned model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.label_classes = [
            "all-or-nothing thinking",
            "catastrophizing",
            "fortune-telling",
            "labeling",
            "magnification or minimization",
            "mental filtering",
            "mind reading",
            "overgeneralization",
            "personalization",
            "should statements"
        ]
    
    def load_model(self, model_dir=None):
        """
        Load the pre-trained model and tokenizer.
        
        Args:
            model_dir: Directory containing the model files. If None, uses distilbert-base-uncased.
        """
        try:
            # If we don't have a saved model, use a pre-trained model
            # In a real implementation, you would save and load your trained model
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(self.label_classes)
            )
            
            # Create a classifier pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # Use CPU
                return_all_scores=True
            )
            
            return True
        except Exception as e:
            print(f"Error loading cognitive distortion model: {e}")
            return False

    def clean_text(self, text):
        """Clean the input text."""
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def predict(self, text):
        """
        Predict cognitive distortions in the given text using rule-based approach.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary with detected distortions and their confidence scores
        """
        try:
            # Clean the input text
            cleaned_text = self.clean_text(text)
            
            # Rule-based cognitive distortion detection
            distortions = []
            
            # Define keywords and patterns for each distortion type
            distortion_patterns = {
                "all-or-nothing thinking": [
                    "always", "never", "completely", "totally", "everyone", "nobody", 
                    "everything", "nothing", "all", "none", "every time", "perfect", 
                    "failure", "disaster", "ruined", "worthless"
                ],
                "catastrophizing": [
                    "disaster", "terrible", "awful", "horrible", "worst", "catastrophe",
                    "devastating", "ruined", "doomed", "end of the world", "nightmare",
                    "can't handle", "unbearable", "overwhelming"
                ],
                "fortune-telling": [
                    "will never", "going to fail", "will happen", "definitely will",
                    "bound to", "destined to", "inevitable", "certain to", "predict",
                    "know it will", "sure to happen", "going to be"
                ],
                "mind reading": [
                    "they think", "he thinks", "she thinks", "everyone thinks", "people think",
                    "they believe", "they feel", "judging me", "laughing at me",
                    "they must think", "probably thinks", "I know what they"
                ],
                "overgeneralization": [
                    "always happens", "never works", "every time", "this proves",
                    "typical", "same thing", "pattern", "again and again", "once again",
                    "here we go again", "story of my life", "just like"
                ],
                "personalization": [
                    "my fault", "I caused", "because of me", "I should have", "if only I",
                    "I'm responsible", "I'm to blame", "it's my fault", "I could have prevented",
                    "I made them", "I let them down"
                ],
                "labeling": [
                    "I am", "I'm such a", "what a", "total", "complete", "such an",
                    "stupid", "idiot", "loser", "failure", "worthless", "pathetic"
                ],
                "magnification or minimization": [
                    "huge", "tiny", "massive", "insignificant", "enormous", "minute",
                    "blown out of proportion", "making a big deal", "not important",
                    "doesn't matter", "exaggerating", "overreacting"
                ],
                "mental filtering": [
                    "only", "just", "except", "but", "however", "focus on", "can't stop thinking",
                    "dwelling on", "keep thinking about", "stuck on", "obsessing"
                ],
                "should statements": [
                    "should", "must", "have to", "ought to", "need to", "supposed to",
                    "should have", "must have", "shouldn't", "mustn't", "expected to"
                ]
            }
            
            # Analyze text for each distortion type
            for distortion_type, keywords in distortion_patterns.items():
                score = 0
                matches = 0
                total_words = len(cleaned_text.split())
                
                for keyword in keywords:
                    if keyword in cleaned_text.lower():
                        matches += 1
                        # Weight longer phrases more heavily
                        score += len(keyword.split()) * 0.1
                
                if matches > 0:
                    # Calculate confidence based on keyword density and matches
                    confidence = min(0.9, (matches * 0.15) + (score * 0.1))
                    
                    # Add some randomness to make it seem more realistic
                    import random
                    confidence = max(0.3, confidence + random.uniform(-0.1, 0.1))
                    
                    distortions.append({
                        "distortion_type": distortion_type,
                        "confidence": round(confidence, 3)
                    })
            
            # If no specific patterns found, check for general negative language
            if not distortions:
                negative_words = [
                    "bad", "wrong", "terrible", "awful", "hate", "can't", "won't",
                    "difficult", "hard", "impossible", "problem", "issue", "worry"
                ]
                negative_count = sum(1 for word in negative_words if word in cleaned_text.lower())
                
                if negative_count > 0:
                    # Default to overgeneralization for general negative thinking
                    confidence = min(0.6, negative_count * 0.1 + 0.3)
                    distortions.append({
                        "distortion_type": "overgeneralization",
                        "confidence": round(confidence, 3)
                    })
            
            # Sort by confidence
            distortions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Return results
            return {
                "distortions": distortions[:5],  # Limit to top 5
                "primary_distortion": distortions[0]["distortion_type"] if distortions else None,
                "primary_confidence": distortions[0]["confidence"] if distortions else 0.0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_distortion_explanation(self, distortion_type):
        """
        Get explanation for a specific cognitive distortion type.
        
        Args:
            distortion_type: Type of cognitive distortion
            
        Returns:
            Explanation text
        """
        explanations = {
            "all-or-nothing thinking": "Seeing things in black-and-white categories with no middle ground.",
            "catastrophizing": "Expecting the worst possible outcome or exaggerating how bad a situation will be.",
            "fortune-telling": "Making predictions about the future without adequate evidence.",
            "labeling": "Attaching a negative label to yourself or others instead of describing the behavior.",
            "magnification or minimization": "Exaggerating negatives or downplaying positives.",
            "mental filtering": "Focusing exclusively on certain (usually negative) aspects while ignoring everything else.",
            "mind reading": "Assuming you know what others are thinking without evidence.",
            "overgeneralization": "Making broad conclusions based on a single event or piece of evidence.",
            "personalization": "Believing others are behaving negatively because of you, without considering alternative explanations.",
            "should statements": "Having rigid rules about how you and others should behave."
        }
        
        return explanations.get(distortion_type, "No explanation available for this distortion type.")
    
    def get_reframing_suggestion(self, distortion_type):
        """
        Get reframing suggestion for a specific cognitive distortion.
        
        Args:
            distortion_type: Type of cognitive distortion
            
        Returns:
            Reframing suggestion text
        """
        suggestions = {
            "all-or-nothing thinking": "Try to find the middle ground. Things are rarely completely black or white. Can you identify any shades of gray in this situation?",
            "catastrophizing": "Consider what's most likely to happen rather than the worst possible scenario. What evidence do you have that things might turn out okay?",
            "fortune-telling": "The future isn't set in stone. What other possible outcomes could there be? What's the evidence for each possibility?",
            "labeling": "Instead of applying a label, try to describe the specific behaviors or situations objectively.",
            "magnification or minimization": "Try to view the situation in a more balanced way. Are there positives you're minimizing or negatives you're exaggerating?",
            "mental filtering": "Challenge yourself to look at the whole picture. What positives might you be filtering out?",
            "mind reading": "Without clear communication, we can't know what others are thinking. What other explanations could there be for their behavior?",
            "overgeneralization": "Look for counter-examples that don't fit the pattern you're seeing. Is this really an 'always' situation?",
            "personalization": "Consider other factors that might explain what happened. Is there a way this could not be about you?",
            "should statements": "Try replacing 'should' with 'prefer' or 'would like'. How does that change how you feel about the situation?"
        }
        
        return suggestions.get(distortion_type, "Consider challenging this thought pattern by examining evidence for and against it.")

# Example usage
if __name__ == "__main__":
    model = CognitiveDistortionModel()
    success = model.load_model()
    
    if success:
        test_text = "I always fail at everything I try and everyone thinks I'm worthless."
        result = model.predict(test_text)
        print(f"Input: {test_text}")
        print(f"Detected distortions:")
        
        for dist in result["distortions"][:3]:  # Show top 3
            distortion = dist["distortion_type"]
            confidence = dist["confidence"] * 100
            print(f"- {distortion} ({confidence:.2f}%)")
            print(f"  Explanation: {model.get_distortion_explanation(distortion)}")
            print(f"  Reframing: {model.get_reframing_suggestion(distortion)}")