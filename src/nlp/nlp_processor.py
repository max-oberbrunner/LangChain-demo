import os
from langchain import OpenAI  # or any preferred model from LangChain
from dotenv import load_dotenv

# Load environment configurations
load_dotenv()

class NLPProcessor:
    def __init__(self):
        self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_code_explanation(self, code_snippet: str) -> str:
        """Generate an explanation for the given code snippet."""
        # Prepare the prompt for the model
        prompt = f"Explain the following code:\n{code_snippet}\n"
        
        # Call the model and get the response
        response = self.model(prompt)
        return response

    def generate_code_suggestion(self, query: str) -> str:
        """Generate suggestions based on the user's query."""
        # Prepare the prompt for the model
        prompt = f"Suggest improvements or changes for this request:\n{query}\n"
        
        # Call the model and get the response
        response = self.model(prompt)
        return response