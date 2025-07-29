"""
Test script to check basic functionality
"""
import os
from dotenv import load_dotenv

print("Starting test...")
load_dotenv()

print(f"Azure API Key exists: {bool(os.getenv('AZURE_API_KEY'))}")
print(f"Azure endpoint exists: {bool(os.getenv('Azure_endpoint'))}")
print(f"Embedding API key exists: {bool(os.getenv('Embedding_api_key'))}")

try:
    from langchain_openai import AzureChatOpenAI
    print("AzureChatOpenAI imported successfully")
    
    model = AzureChatOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("Azure_endpoint"),
        deployment_name="gpt-4o-mini",
        api_version="2024-12-01-preview",
        temperature=0.7,
        max_tokens=512
    )
    print("Model initialized successfully")
    
    # Test a simple call
    response = model.invoke("Hello, can you respond?")
    print(f"Model response: {response.content}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")
