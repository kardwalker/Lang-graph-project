from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
import os
print("The following code is an example of using ChromaDB withpresistence with Azure s.")
load_dotenv()

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key"),  # Replace with actual key
    deployment="text-embedding-ada-002",  # Deployment name must match what you 
    # created in Azure
    model="text-embedding-ada-002",       # Model name (optional, but helps for clarity)
    api_version="2023-05-15", )

doc = [
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by Pregel and Apache Beam.",
        metadata={"source": "doc1"}
    ),
    Document(
        page_content= "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.",
        metadata={"source": "doc2"}
    ),
    Document(
        page_content= """Connect your Python application to a Redis database
redis-py is the Python client for Redis. The sections below explain how to install redis-py and connect your application to a Redis database.
redis-py requires a running Redis server. See here for Redis Open Source installation instructions.
You can also access Redis with an object-mapping client interface. See RedisOM for Python for more information.
For faster performance, install Redis with hiredis support. This provides a compiled response parser, and for most cases requires zero code changes. By default, if hiredis >= 1.0 is available, redis-py attempts to use it for response parsing.


Key use cases
Redis excels in various applications, including:

Caching: Supports multiple eviction policies, key expiration, and hash-field expiration.
Distributed Session Store: Offers flexible session data modeling (string, JSON, hash).
Data Structure Server: Provides low-level data structures (strings, lists, sets, hashes, sorted sets, JSON, etc.) with high-level semantics (counters, queues, leaderboards, rate limiters) and supports transactions & scripting.
NoSQL Data Store: Key-value, document, and time series data storage.
Search and Query Engine: Indexing for hash/JSON documents, supporting vector search, full-text search, geospatial queries, ranking, and aggregations via Redis Query Engine.
Event Store & Message Broker: Implements queues (lists), priority queues (sorted sets), event deduplication (sets), streams, and pub/sub with probabilistic stream processing capabilities.
Vector Store for GenAI: Integrates with AI applications (e.g. LangGraph, mem0) for short-term memory, long-term memory, LLM response caching (semantic caching), and retrieval augmented generation (RAG).
Real-Time Analytics: Powers personalization, recommendations, fraud detection, and risk assessment.
Why choose Redis?
Redis is a popular choice for developers worldwide due to its combination of speed, flexibility, and rich feature set. Here's why people choose Redis for:

Performance: Because Redis keeps data primarily in memory and uses efficient data structures, it achieves extremely low latency (often sub-millisecond) for both read and write operations. This makes it ideal for applications demanding real-time responsiveness.
Flexibility: Redis isn't just a key-value store, it provides native support for a wide range of data structures and capabilities listed in What is Redis?
Extensibility: Redis is not limited to the built-in data structures, it has a modules API that makes it possible to extend Redis functionality and rapidly implement new Redis commands
Simplicity: Redis has a simple, text-based protocol and well-documented command set
Ubiquity: Redis is battle tested in production workloads at a massive scale. There is a good chance you indirectly interact with Redis several times daily
Versatility: Redis is the de facto standard for use cases such as:
Caching: quickly access frequently used data without needing to query your primary database
Session management: read and write user session data without hurting user experience or slowing down every API call
Querying, sorting, and analytics: perform deduplication, full text search, and secondary indexing on in-memory data as fast as possible
Messaging and interservice communication: job queues, message brokering, pub/sub, and streams for communicating between services
Vector operations: Long-term and short-term LLM memory, RAG content retrieval, semantic caching, semantic routing, and vector similarity search"""
,
        metadata={"source": "doc3"}
    ),

]
# Create a Chroma vector store & persist it
# Note: Ensure the directory exists or Chroma will create it
vector_store = Chroma.from_documents(
    documents=doc,
    embedding=embedding_model,
    collection_name="agentic_rag",
    persist_directory="./chroma_store"
)

# Load existing vector store (persistence is automatic with persist_directory)
vector_store = Chroma(
    collection_name="agentic_rag",
    embedding_function=embedding_model,
    
    persist_directory="./chroma_store"
)
"""
Build a RAG Pipeline
In a RAG chatbot, you:

Get a query

Retrieve relevant documents using the vector store

Send those documents + query to the LLM for final answer

"""
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI

# Step 1: Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Step 2: Load the LLM

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",  # Ensure this matches your deployment name
)


# Step 3: RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True
)

# Step 4: Ask a question
query = "What is LangGraph?"
response = rag_chain.invoke(query)

print("Answer:", response["result"])
print("Sources:", response["source_documents"])

# 
# Expose It as an API with FastAPI

