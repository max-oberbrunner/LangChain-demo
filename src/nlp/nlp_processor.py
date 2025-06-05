import os
from langchain_openai import OpenAI
from dotenv import load_dotenv
from models import NLPModelConfig  # Import the configuration class

# Load environment configurations
load_dotenv()

class NLPProcessor:
    def __init__(self):
        config = NLPModelConfig()  # Initialize the model configuration
        self.explanation_model = OpenAI(api_key=config.get_api_key(), model_name="gpt-3.5")  # GPT-3.5 for explanations
        self.suggestion_model = OpenAI(api_key=config.get_api_key(), model_name="gpt-4o-mini")  # GPT-4 for suggestions

    def get_code_explanation(self, code_snippet: str) -> str:
        """Generate an explanation for the given code snippet using GPT-3.5."""
        prompt = f"Explain the following code:\n{code_snippet}\n"
        response = self.explanation_model(prompt)
        return response

    def generate_code_suggestion(self, query: str) -> str:
        """Generate suggestions for code improvements using GPT-4."""
        prompt = f"Suggest improvements or changes for the following request:\n{query}\n"
        response = self.suggestion_model(prompt)
        return response