import unittest
import numpy as np
import faiss
from dotenv import load_dotenv
from unittest.mock import patch
from analysis.code_analyzer import vectorize_code_structure, query_code_suggestions
import os
class TestCodeAnalysis(unittest.TestCase):

    def setUp(self):
        self.code_base = {
            'function_a': "def function_a(): pass",
            'function_b': "def function_b(): return 'Hello, World!'",
            'function_c': "def function_c(x): return x * 2"
        }

        load_dotenv()

        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            import getpass
            openai_api_key = getpass.getpass("Enter API key for OpenAI: ")

        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Mock the OpenAIEmbeddings to return a fixed embedding (as a NumPy array)
        with patch('langchain_openai.OpenAIEmbeddings') as MockEmbeddings:
            self.mock_embeddings = MockEmbeddings.return_value
            # Simulate an embedding with random values as a NumPy array
            self.mock_embeddings.embed_query.return_value = np.random.rand(1536).astype('float32')  # Simulate an embedding with 1536 dimensions

            # Create embeddings for the code base
            self.fuzzy_vectors = np.array([self.mock_embeddings.embed_query(snippet) for snippet in self.code_base.values()]).astype('float32')
            self.index = faiss.IndexFlatL2(self.fuzzy_vectors.shape[1])
            self.index.add(self.fuzzy_vectors)  # Prepare the index for querying

    def test_vectorization(self):
        """Test if the vectorization works correctly."""
        vectorized_index = vectorize_code_structure(self.code_base)
        self.assertIsInstance(vectorized_index, faiss.IndexFlatL2)
        self.assertEqual(vectorized_index.ntotal, len(self.code_base))

    def test_query_code_suggestions(self):
        """Test the code suggestions based on a query."""
        user_query = "How can I improve function_b?"
        suggestions = query_code_suggestions(user_query, self.index, self.code_base, k=2)
        
        # Check if suggestions are returned and are in the correct format
        self.assertIsInstance(suggestions, dict)
        self.assertGreater(len(suggestions), 0)  # Ensure there is at least one suggestion

        # Check that the returned suggestion key is one of the function names
        for key in suggestions.keys():
            self.assertIn(key, self.code_base.keys())

    def tearDown(self):
        """Clean up any resources if needed."""
        pass

if __name__ == '__main__':
    unittest.main()