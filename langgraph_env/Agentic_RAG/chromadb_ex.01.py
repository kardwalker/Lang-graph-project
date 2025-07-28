from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_chroma import Chroma
from langchain.schema import Document

embeddding_model = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key"),  # Replace with actual key  
    deployment="text-embedding-ada-002",  # Deployment name must match what you
    # created in Azure
    model="text-embedding-ada-002",       # Model name (optional, but helps for clarity)
    api_version="2023-05-15",        # Use correct version (based on Azure docs or trial-and-error)
)
doc = [
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "doc1"}
    ),
    Document(
        page_content= "LangChain is an open-source framework designed to help developers build applications powered by Large Language Models (LLMs) such as OpenAI’s GPT, Anthropic's Claude, and others. It provides tools to connect LLMs with external data, reasoning chains, memory, tools, agents, and more."
        , metadata={"source": "doc2"}    

    ),
    Document(
        page_content="""If you're building an agent with tools + retrieval, use both together:
→ LlamaIndex handles document data
→ LangChain handles agents, tools, reasoning steps""",
        metadata={"source": "doc3"}

    )
]



# index the documents in chroma bvector store
vector_store = Chroma.from_documents(documents=doc, embedding=embeddding_model, collection_name="agentic_rag")
print("Vector store created with", len(vector_store.get()) , "documents.")

q = "What is LangGraph?"
results = vector_store.similarity_search(query=q, k=2)  # Retrieve top 2 results
print("Most similar documents for query:", q)
print(results[0])


q_embedding = embeddding_model.embed_query(q)

docs_by_vector = vector_store.similarity_search_by_vector(q_embedding, k=2)

# Print the results
print("Most similar documents by vector for query:", q)

print(docs_by_vector[0])