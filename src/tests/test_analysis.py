import unittest
import numpy as np
import faiss
from analysis.code_analyzer import vectorize_code_structure, query_code_suggestions
from langchain.embeddings import OpenAIEmbeddings

class TestCodeAnalysis(unittest.TestCase):

    def setUp(self):
        self.code_base = {
            'function_a': "def function_a(): pass",
            'function_b': "def function_b(): return 'Hello, World!'",
            'function_c': "def function_c(x): return x * 2"
        }
        self.embeddings = OpenAIEmbeddings(api_key='test_key')  # Use a mock key for tests
        self.fuzzy_vectors = np.array([self.embeddings.embed(snippet) for snippet in self.code_base.values()]).astype('float32')
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
        
        # Check if suggestions are returned and are in correct format
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