import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

class JevDecisionAPI:
    def __init__(self):
        self.api_key = os.environ.get("TYPESAFE_API_KEY")
        self.base_url = os.environ.get("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
        self.model = "typesafe/jev-1.13"

    def route_intent(self, text: str) -> str:
        """
        Calls Jev Decision API to classify the intent.
        Returns: 'query', 'learn', 'contradiction', or 'clarification'
        """
        if not self.api_key:
            logger.warning("No TYPESAFE_API_KEY found, falling back to 'learn'")
            return "learn"

        payload = {
            "model": self.model,
            "state": text,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Classify the intent of the user's input.",
                    "criteria": {
                        "query": "The user is asking a question or requesting information.",
                        "learn": "The user is stating a new fact to be learned or asserting something.",
                        "contradiction": "The user is correcting an old fact.",
                        "clarification": "The input is ambiguous, confusing, or needs clarification."
                    }
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            resp = requests.post(f"{self.base_url}/alpha/decisions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data["answers"]["intent"]["choice"]
            logger.info(f"Jev routing: {choice} for input '{text}'")
            return choice
        except Exception as e:
            logger.error(f"Jev routing failed: {e}")
            return "learn" # fallback
