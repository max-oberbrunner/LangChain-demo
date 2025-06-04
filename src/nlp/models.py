import os

class NLPModelConfig:
    """Configuration class for NLP models and parameters."""
    def __init__(self):
        self.model_name = "gpt-3.5"  # Default model, can be adapted or fetched from environment
        self.api_key = os.getenv("OPENAI_API_KEY")

    def get_model_name(self):
        return self.model_name

    def get_api_key(self):
        return self.api_key