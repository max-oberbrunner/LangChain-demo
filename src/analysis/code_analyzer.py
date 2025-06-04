import numpy as np
import faiss
import os
from langchain_openai import OpenAIEmbeddings

def analyze_code(code_snippet: str):
    # Placeholder for actual code analysis logic
    return {"analysis": "No issues found."}  # Simple response for now

def vectorize_code_structure(code_base):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create embeddings for each code snippet
    vector_list = [embeddings.embed_query(code_snippet) for code_snippet in code_base.values()]
    vectors = np.array(vector_list).astype('float32')

    # Construct a Faiss index
    index = faiss.IndexFlatL2(vectors.shape[1])  # Use L2 distance
    index.add(vectors)  # Add vectors to the index
    
    return index

def query_code_suggestions(query: str, index, code_base, k: int = 5):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Embed the user query
    query_vector = embeddings.embed_query(query)  # This returns a list
    query_vector = np.array(query_vector).astype('float32').reshape(1, -1)  # Convert to NumPy array and reshape

    # Search for the k nearest vectors
    distances, indices = index.search(query_vector, k)
    
    # Prepare suggestions with their distances
    suggestions = {list(code_base.keys())[i]: distances[0][i] for i in indices[0]}
    return suggestions