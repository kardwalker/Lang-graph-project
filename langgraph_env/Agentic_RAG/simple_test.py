"""
One-shot RAG API client - ask a single question
Usage: python simple_test.py "Your question here"
"""
import requests
import sys
import json

def ask_question(question: str):
    """Send a question to the RAG API"""
    url = "http://localhost:8000/ask"
    payload = {"question": question}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print(f"❓ Question: {result['question']}")
        print(f"✅ Answer: {result['answer']}")
        print(f"📚 Context chunks retrieved: {len(result['context'])}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_test.py \"Your question here\"")
        sys.exit(1)
    
    question = sys.argv[1]
    ask_question(question)
