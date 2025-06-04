import numpy as np
import faiss
import os
from langchain_openai import OpenAIEmbeddings

def analyze_code(code_snippet: str):
    """
    Analyzes a given code snippet.

    Args:
        code_snippet (str): The code snippet to analyze.

    Returns:
        dict: A dictionary containing the analysis results. 
              Currently returns a placeholder message indicating no issues found.
    """
    return {"analysis": "No issues found."}

def vectorize_code_structure(code_base):
    """
    Converts a code base into embeddings and creates a Faiss index for similarity search.

    Args:
        code_base (dict): A dictionary containing function names as keys and their
                          code snippets as values.

    Returns:
        faiss.Index: A Faiss index containing the vector embeddings of the code snippets.
    """
    # Initialize the embedding model using the OpenAI API key from the environment variables
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Generate embeddings for each code snippet in the code base
    vector_list = [embeddings.embed_query(code_snippet) for code_snippet in code_base.values()]
    vectors = np.array(vector_list).astype('float32')  # Convert the list of vectors to a NumPy array

    # Create a Faiss index for L2 distance (Euclidean distance) searches
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)  # Add the vector embeddings to the index
    
    return index

def query_code_suggestions(query: str, index, code_base, k: int = 5):
    """
    Queries the Faiss index to find suggestions based on a user query.

    Args:
        query (str): The input query string from the user.
        index (faiss.Index): The Faiss index containing function vectors for similarity search.
        code_base (dict): A dictionary containing function names and their associated code snippets.
        k (int, optional): The number of nearest neighbors to return. Defaults to 5.

    Returns:
        dict: A dictionary containing function names as keys and their corresponding distance
              and code snippet description as values.
    """
    # Generate the embedding for the user query
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    query_vector = embeddings.embed_query(query)  # Embed the user query
    query_vector = np.array(query_vector).astype('float32').reshape(1, -1)  # Reshape to make it two-dimensional

    # Search the Faiss index for the k nearest vectors to the query vector
    distances, indices = index.search(query_vector, k)
    
    # Prepare suggestions as a dictionary
    suggestions = {}
    for i in indices[0]:
        func_name = list(code_base.keys())[i]  # Get the name of the function
        suggestions[func_name] = {
            "distance": distances[0][i],  # Add the distance of the suggested function
            "description": code_base[func_name]  # Add the description (function code)
        }
    return suggestions