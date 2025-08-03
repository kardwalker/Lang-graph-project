import unittest
from lesson1 import graph

class test_agent_node(unittest.TestCase):
    def test_welcome_node(self):
        # Test the initial state
        initial_state = {'greeting': 'Agent'}
        result = graph.invoke(initial_state)
        
        # Check if the greeting is updated correctly
        self.assertEqual(result['greeting'], "Welcome to LangGraph! Agent")

if __name__ == '__main__':
    unittest.main()        