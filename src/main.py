import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from analysis.code_analyzer import vectorize_code_structure, query_code_suggestions
from nlp.nlp_processor import NLPProcessor  # Import the NLPProcessor class

# Load environment variables
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    import getpass
    openai_api_key = getpass.getpass("Enter API key for OpenAI: ")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize the language model
model = init_chat_model("gpt-4o-mini", model_provider="openai")
nlp_processor = NLPProcessor()  # Initialize NLPProcessor

def provide_suggestions(suggestions, code_base):
    """Format and provide suggestions based on their relevance."""
    response_lines = []
    for func_name, info in suggestions.items():
        distance = info["distance"]
        description = info["description"]
        if distance < 0.3:
            response_lines.append(f" - {func_name}: Highly relevant (distance: {distance:.4f})\n   Description: {description}\n   Suggestion: You can use this function directly in your code.")
        elif 0.3 <= distance < 0.6:
            response_lines.append(f" - {func_name}: Somewhat relevant (distance: {distance:.4f})\n   Description: {description}\n   Suggestion: This function might be useful for your task.")
        else:
            response_lines.append(f" - {func_name}: Less relevant (distance: {distance:.4f})\n   Description: {description}\n   Suggestion: Consider exploring other functions.")
    
    return "\n".join(response_lines)

def main():
    # Define a code base with several functions
    code_base = {
        'function_a': """
def function_a():
    \"\"\"Returns a greeting message.\"\"\"
    return "Hello, World!"
""",

        'function_b': """
def function_b(name):
    \"\"\"Returns a personalized greeting message.
    
    Args:
        name (str): The name to greet.
    \"\"\"
    return f"Hello, {name}!"
""",

        'function_c': """
def function_c(x, y):
    \"\"\"Returns the sum of two numbers.

    Args:
        x (int or float): The first number.
        y (int or float): The second number.
    \"\"\"
    return x + y
""",

        'function_d': """
def function_d(n):
    \"\"\"Returns the factorial of a number.

    Args:
        n (int): The number to calculate the factorial for.
    \"\"\"
    if n == 0:
        return 1
    else:
        return n * function_d(n - 1)
""",

        'function_e': """
def function_e(data):
    \"\"\"Filters the even numbers from a list.

    Args:
        data (list): A list of integers.

    Returns:
        list: A list of even integers.
    \"\"\"
    return [x for x in data if x % 2 == 0]
"""
    }

    # Vectorize the codebase and create the Faiss index
    faiss_index = vectorize_code_structure(code_base)

    print("Welcome to the Code Analysis Tool!")
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        # Use the NLPProcessor to get code suggestions or explanations based on the user query
        if "explain" in user_query.lower():
            # Trigger explanation functionality
            explanation = nlp_processor.get_code_explanation(user_query)
            print(f"\nExplanation:\n{explanation}")
        else:
            # Query for code suggestions using the embedded vector index
            suggestions = query_code_suggestions(user_query, faiss_index, code_base)

            print("\nSuggestions:")
            response = provide_suggestions(suggestions, code_base)
            print(response)

if __name__ == "__main__":
    main()