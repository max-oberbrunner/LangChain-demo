import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from analysis.code_analyzer import vectorize_code_structure, query_code_suggestions

# Load environment variables
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    import getpass
    openai_api_key = getpass.getpass("Enter API key for OpenAI: ")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize the language model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

def main():
    # Define a code base with several functions
    code_base = {
        'function_a': "def function_a(): pass",
        'function_b': "def function_b(): return 'Hello, World!'",
        'function_c': "def function_c(x): return x * 2"
    }

    # Vectorize the codebase and create the Faiss index
    faiss_index = vectorize_code_structure(code_base)

    print("Welcome to the Code Analysis Tool!")
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        suggestions = query_code_suggestions(user_query, faiss_index, code_base)

        print("\nSuggestions:")
        for func_name, distance in suggestions.items():
            print(f" - {func_name}: {distance:.4f}")
        print()

if __name__ == "__main__":
    main()