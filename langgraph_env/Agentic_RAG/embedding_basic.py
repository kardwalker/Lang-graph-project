print("the following code is implementation of Embedding")

from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os   
load_dotenv()


from langchain_community.vectorstores import FAISS
embedding = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key"),  # Replace with actual key
    deployment="text-embedding-ada-002",  # Deployment name must match what you named the embedding model in Azure is very important
    model="text-embedding-ada-002",       # Model name (optional, but helps for clarity)
    api_version="2023-05-15",        # Use correct version (based on Azure docs or trial-and-error)
   
)



# Example usage:
#embed = embedding.embed_query("Azure OpenAI is awesome.")
#print("Embedding:", embed.__len__())  # This will print the embedding vector for the input text
# Embedding: 1536

# implemeting FAISS vectorstore
#from langchain.vectorstores import FAISS deprected
texts = ["LangChain is great", "Azure OpenAI integration is powerful", "Embeddings are numeric representations"]

# Step 3: Create FAISS vectorstore
faiss_store = FAISS.from_texts(texts, embedding)

# Step 4: Perform similarity search
results = faiss_store.similarity_search("What is LangChain?", k=2)
for r in results:
    print(r.page_content)

print(results)    