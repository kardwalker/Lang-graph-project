from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
import os
load_dotenv()


embedding = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/", 
    api_key=os.getenv("Embedding_api_key"),  # Replace with actual key
    deployment="text-embedding-ada-002",  # Deployment name must match what you named the embedding model in Azure is very important
    model="text-embedding-ada-002",       # Model name (optional, but   helps for clarity)
    api_version="2023-05-15",        # Use correct version (based on Azure docs or trial-and-error)
)
text1 = "LangGraph is a library for building stateful, multi-actor applications with LLMs."
text2 = "LangChain is a framework for building context-aware reasoning applications."
text3 = "The quick brown fox jumps over the lazy dog."

vectorstore = InMemoryVectorStore.from_texts([text1, text2 , text3],embedding=embedding)

# convert the vector store into a retriever
retriever = vectorstore.as_retriever()  

query = "What is LangGraph?"
results = retriever.invoke(query, k=2)  # Retrieve top 2 results
print("Results:", type(results))