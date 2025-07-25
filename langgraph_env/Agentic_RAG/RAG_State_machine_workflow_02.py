# RAG_State_machine_workflow_02.py
# ========================================
# RAG (Retrieval-Augmented Generation) Workflow Implementation
# Using LangGraph State Machine Architecture
# ========================================
print("✅ This is the improved implementation of a RAG workflow using LangGraph with a state machine architecture.")

# Core LangChain imports for RAG pipeline
from langchain_core.runnables import RunnablePassthrough  # For data flow in chains
from langchain_core.output_parsers import StrOutputParser  # Parse LLM output to string
from langchain_chroma import Chroma  # Vector database for document storage/retrieval
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI  # Azure OpenAI models
from langchain_core.prompts import ChatPromptTemplate  # Template for LLM prompts
from dotenv import load_dotenv  # Load environment variables from .env file
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split documents into chunks

# Python standard libraries
from typing import List, TypedDict  # Type hints for better code clarity
import os  # Operating system interface for environment variables
import asyncio  # Asynchronous programming support
import logging  # Logging functionality for debugging

# LangGraph imports for state machine workflow
from langchain_community.document_loaders import WebBaseLoader  # Load documents from web URLs
from langgraph.graph import StateGraph, START, END  # State machine graph components
from langgraph.checkpoint.memory import MemorySaver  # Memory management for state persistence

# ========================================
# CONFIGURATION & INITIALIZATION
# ========================================

# Load environment variables from .env file (API keys, endpoints, etc.)
load_dotenv()

# Configure logging to show INFO level messages with timestamps
logging.basicConfig(level=logging.INFO)

# ========================================
# AZURE OPENAI MODEL INITIALIZATION
# ========================================

# Initialize Azure OpenAI chat model for answer generation
# This model will be used to generate final answers based on retrieved context
model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),  # API key from environment variables
    azure_endpoint=os.getenv("Azure_endpoint"),  # Azure OpenAI endpoint URL
    deployment_name="gpt-4o-mini",  # Specific deployment name in Azure
    api_version="2024-12-01-preview",  # API version for compatibility
    temperature=0.7,  # Controls randomness (0=deterministic, 1=creative)
    max_tokens=512  # Maximum tokens in the response
)

# Initialize Azure OpenAI embeddings model for document vectorization
# This model converts text into numerical vectors for similarity search
embedding = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/",  # Embedding endpoint
    api_key=os.getenv("Embedding_api_key"),  # Separate API key for embeddings
    deployment="text-embedding-ada-002",  # Embedding model deployment name
    api_version="2023-05-15"  # API version for embedding service
)

# ========================================
# DOCUMENT LOADING & PROCESSING
# ========================================

# Define URLs to scrape for document content
# These URLs contain information about Redis that will be used as knowledge base
urls = [
    "https://github.com/redis/redis",      # Main Redis repository
    "https://github.com/redis/redis-py"   # Redis Python client repository
]

# Load documents from web URLs
docs = []
for url in urls:
    # Create WebBaseLoader instance for each URL
    loader = WebBaseLoader(url)
    
    # Set user agent to avoid bot detection warnings
    # This helps identify our scraper as a legitimate browser request
    loader.session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Load and extend the documents list with content from this URL
    docs.extend(loader.load())

# ========================================
# DOCUMENT CHUNKING & VECTORIZATION
# ========================================

# Split documents into smaller chunks for better retrieval
# Smaller chunks improve relevance and fit within context windows
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Maximum characters per chunk
    chunk_overlap=10    # Overlap between chunks to maintain context
)
docs_splits = text_splitter.split_documents(docs)

# Create vector store using ChromaDB for document storage and retrieval
# This converts document chunks into embeddings and stores them for similarity search
vector_store = Chroma.from_documents(
    documents=docs_splits,  # The chunked documents to store
    embedding=embedding,    # The embedding model to use for vectorization
    collection_name="RAG_with_multiple_docs",  # Name for this document collection
    persist_directory="./chroma_db_RAG_STATE_MACHINE_WORKFLOW_02"  # Local storage directory
)
# Note: Persistence is handled automatically when persist_directory is specified

# ========================================
# RAG CHAIN SETUP
# ========================================

# Create prompt template for the language model
# This template structures how the context and question are presented to the LLM
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say "I don't know".

Question: {question}
Context:
{context}
"""
)

# Define RAG chain: prompt → model → output parser
# This chain processes the question and context through the LLM and returns a string
rag_chain = prompt | model | StrOutputParser()

# ========================================
# STATE DEFINITION FOR LANGGRAPH
# ========================================

# Define the state structure for the RAG workflow
# TypedDict provides type hints for better code clarity and IDE support
class RagState(TypedDict):
    question: str           # The user's input question
    web_search: List[str]   # List of document contents for reference
    answer: str            # The final generated answer
    documents: List[object] # Retrieved documents from vector store

# ========================================
# WORKFLOW NODE FUNCTIONS
# ========================================

# Node 1: Document Retrieval
# This function handles the first step of RAG - finding relevant documents
def retrieve_documents(state: RagState) -> RagState:
    """
    Retrieve relevant documents from the vector store based on the user's question.
    
    Args:
        state (RagState): Current workflow state containing the question
        
    Returns:
        RagState: Updated state with retrieved documents
    """
    logging.info("🔍 Retrieving documents for question: %s", state['question'])
    
    # Create a retriever from the vector store
    # search_type="similarity": Use cosine similarity for document matching
    # k=3: Retrieve top 3 most relevant documents
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Invoke retriever with the question and store results in state
    state['documents'] = retriever.invoke(state['question'])
    
    return state

# Node 2: Answer Generation
# This function generates the final answer using retrieved documents as context
def generate_answer(state: RagState) -> RagState:
    """
    Generate an answer using the retrieved documents as context.
    
    Args:
        state (RagState): Current workflow state with question and documents
        
    Returns:
        RagState: Updated state with generated answer and web search context
    """
    logging.info("✍️ Generating answer from retrieved documents...")
    
    # Combine all document contents into a single context string
    # Each document is separated by double newlines for clarity
    context = "\n\n".join(doc.page_content for doc in state['documents'])
    
    # Store document contents for reference (useful for debugging/logging)
    state['web_search'] = [doc.page_content for doc in state['documents']]
    
    # Generate answer using the RAG chain (prompt + model + parser)
    state['answer'] = rag_chain.invoke({"question": state['question'], "context": context})
    
    logging.info("✅ Answer generated: %s", state['answer'])
    return state

# ========================================
# LANGGRAPH WORKFLOW CREATION
# ========================================

# Create LangGraph workflow with state machine architecture
def create_workflow():
    """
    Build and compile the RAG workflow state machine.
    
    Workflow Structure:
    START → retrieve → generate_answer → END
    
    Returns:
        Compiled StateGraph: Ready-to-execute workflow
    """
    # Initialize StateGraph with our custom RagState type
    workflow = StateGraph(RagState)
    
    # Add nodes (processing steps) to the workflow
    workflow.add_node("retrieve", retrieve_documents)        # Step 1: Document retrieval
    workflow.add_node("generate_answer", generate_answer)    # Step 2: Answer generation
    
    # Define workflow edges (execution flow)
    workflow.add_edge(START, "retrieve")                     # Start → retrieve documents
    workflow.add_edge("retrieve", "generate_answer")         # retrieve → generate answer
    workflow.add_edge("generate_answer", END)                # generate answer → end
    
    # Compile workflow with memory checkpointing for state persistence
    return workflow.compile(checkpointer=MemorySaver())

# ========================================
# ASYNC EXECUTION RUNNER
# ========================================

# Main async function to run the RAG workflow
async def run_main():
    """
    Execute the RAG workflow with user input and error handling.
    """
    # Create the compiled workflow graph
    graph = create_workflow()
    
    # Configuration for workflow execution
    config = {
        "configurable": {"thread_id": "1"},  # Unique thread ID for this session
        "recursion_limit": 20                # Maximum recursive calls to prevent infinite loops
    }

    # Get user input
    question = input("🧠 Enter your query: ")
    input_state = {"question": question}

    try:
        # Stream workflow execution asynchronously
        # stream_mode="updates" provides real-time updates of state changes
        async for event in graph.astream(input_state, config=config, stream_mode="updates"):
            # Check for errors in the workflow execution
            if "error" in event:
                print(f"❌ Error: {event['error']}")
                break
            # Display each workflow step update
            print("📦 Event:", event)
    except Exception as e:
        # Handle any unexpected errors during workflow execution
        print(f"❌ Workflow execution failed: {str(e)}")

# ========================================
# MAIN EXECUTION ENTRY POINT
# ========================================

# Run the async workflow when script is executed directly
if __name__ == "__main__":
    asyncio.run(run_main())


"""📝 Complete Code Documentation Summary
I've added detailed comments throughout the entire RAG State machine workflow, organized into clear sections:

1. File Header & Import Documentation
Purpose: Clear explanation of what the script does
Import Documentation: Each import has comments explaining its role in the RAG pipeline
Logical Grouping: Related imports are grouped with explanatory headers
2. Configuration & Model Initialization
Environment Setup: Explanation of dotenv and logging configuration
Azure OpenAI Models: Detailed comments on both chat and embedding model parameters
Parameter Explanations: Clear descriptions of API keys, endpoints, temperature, etc.
3. Document Processing Pipeline
URL Selection: Explanation of why specific URLs are chosen
Web Scraping: Comments on User-Agent headers and bot detection avoidance
Document Chunking: Detailed explanation of chunk size and overlap parameters
Vector Store Creation: ChromaDB configuration and persistence explanation
4. RAG Chain Setup
Prompt Template: Clear explanation of how context and questions are structured
Chain Composition: Comments on the prompt → model → parser flow
State Definition: TypedDict documentation with field explanations
5. Workflow Node Functions
Function Docstrings: Complete docstrings with Args and Returns
Step-by-Step Comments: Detailed explanation of each operation
Logging Context: Explanation of logging statements and their purpose
6. LangGraph Workflow Architecture
Workflow Structure: Visual representation in comments (START → retrieve → generate → END)
Node Registration: Explanation of each workflow node
Edge Definitions: Clear description of execution flow
Compilation: Comments on checkpointing and memory management
7. Execution & Error Handling
Async Execution: Explanation of async/await patterns
Configuration Options: Thread ID and recursion limit documentation
Streaming: Comments on real-time workflow updates
Error Handling: Exception handling and user feedback
Key Benefits of the Added Comments
✅ Educational Value: New developers can understand each step of the RAG workflow ✅ Maintenance: Future modifications are easier with clear documentation ✅ Debugging: Comments help identify specific workflow stages during troubleshooting ✅ Best Practices: Comments explain design decisions and parameter choices ✅ Architecture Understanding: Clear explanation of state machine pattern implementation

The script remains fully functional while now being extremely well-documented and educational for anyone learning about RAG
"""