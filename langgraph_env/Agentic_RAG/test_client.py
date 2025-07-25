"""
Simple Python client to test the RAG API
"""
import requests
import json

def ask_question(question: str, api_url: str = "http://localhost:8000/ask"):
    """
    Send a question to the RAG API and get a response
    
    Args:
        question (str): The question to ask
        api_url (str): The API endpoint URL
    
    Returns:
        dict: The API response
    """
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making API request: {e}")
        return None

def main():
    """Interactive client for testing the RAG API"""
    print("🤖 RAG API Test Client")
    print("=" * 50)
    print("Enter your questions about Redis (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        question = input("\n🧠 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not question:
            print("⚠️  Please enter a valid question.")
            continue
        
        print("🔍 Processing your question...")
        
        result = ask_question(question)
        
        if result:
            print(f"\n✅ Answer: {result['answer']}")
            print(f"\n📚 Retrieved {len(result['context'])} context chunks")
        else:
            print("❌ Failed to get response from API")

if __name__ == "__main__":
    main()
