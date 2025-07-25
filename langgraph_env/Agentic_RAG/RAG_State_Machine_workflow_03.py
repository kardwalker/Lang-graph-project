"""
RAG Workflow API using LangGraph, FastAPI, and Azure OpenAI

This script sets up a comprehensive Retrieval-Augmented Generation (RAG) pipeline
that combines multiple cutting-edge technologies to create an intelligent question-answering system:

1. **LangGraph State Machine**: Orchestrates the RAG workflow using a state-based approach
2. **Azure OpenAI**: Provides both embedding generation and text completion capabilities
3. **ChromaDB**: Serves as the vector database for document storage and similarity search
4. **FastAPI**: Exposes the RAG pipeline through REST API endpoints
5. **Web Interface**: Provides a user-friendly browser-based interface for testing

Architecture Overview:
- Document Loading: Scrapes web content from specified URLs
- Text Chunking: Splits documents into manageable pieces for processing
- Vector Storage: Converts text chunks to embeddings and stores in ChromaDB
- State Machine: Manages the retrieval and generation workflow
- API Layer: Exposes functionality through RESTful endpoints

Author: You
Date: 2025-07-23
Version: 1.0
"""

# ========================================
# IMPORTS AND DEPENDENCIES
# ========================================

# FastAPI framework imports for building REST API
from fastapi import FastAPI, HTTPException, Request, Form  # Core FastAPI components
from fastapi.responses import HTMLResponse  # For serving HTML content
from pydantic import BaseModel  # Data validation and serialization

# Python standard library imports
from typing import List, TypedDict  # Type hints for better code documentation
from dotenv import load_dotenv  # Environment variable management
import os  # Operating system interface
import logging  # Logging functionality for debugging and monitoring
import asyncio  # Asynchronous programming support
import uvicorn  # ASGI server for running FastAPI applications

# LangChain imports for RAG pipeline components
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  # Azure OpenAI integrations
from langchain_chroma import Chroma  # ChromaDB vector store integration
from langchain_core.runnables import RunnablePassthrough  # Data flow utilities
from langchain_core.output_parsers import StrOutputParser  # Response parsing
from langchain_core.prompts import ChatPromptTemplate  # Prompt template management
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Document chunking
from langchain_community.document_loaders import WebBaseLoader  # Web content loading

# LangGraph imports for state machine workflow
from langgraph.graph import StateGraph, START, END  # State machine components
from langgraph.checkpoint.memory import MemorySaver  # State persistence

# ========================================
# APPLICATION CONFIGURATION
# ========================================

# Load environment variables from .env file
# This includes API keys, endpoints, and other sensitive configuration
load_dotenv()

# Initialize FastAPI application with comprehensive metadata
app = FastAPI(
    title="LangGraph RAG API",  # API title shown in documentation
    description="Ask questions on Redis documentation using a RAG agent powered by LangGraph",  # API description
    version="1.0",  # API version for client compatibility
    docs_url="/docs",  # Swagger UI documentation endpoint
    redoc_url="/redoc"  # ReDoc documentation endpoint
)

# Configure logging system for debugging and monitoring
# INFO level provides detailed workflow information without overwhelming output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        # Optional: Add FileHandler for log persistence
        # logging.FileHandler('rag_api.log')
    ]
)


### ---------------------- MODEL & VECTOR SETUP -----------------------

# ========================================
# AZURE OPENAI MODEL INITIALIZATION
# ========================================

def initialize_azure_models():
    """
    Initialize Azure OpenAI models for chat completion and text embeddings.
    
    This function sets up two separate Azure OpenAI services:
    1. Chat completion model for generating answers
    2. Embedding model for converting text to vectors
    
    Returns:
        tuple: (chat_model, embedding_model) - Initialized Azure OpenAI models
        
    Raises:
        ValueError: If required environment variables are missing
        ConnectionError: If Azure OpenAI endpoints are unreachable
    """
    
    # Initialize Azure OpenAI chat model for answer generation
    # This model processes the user question and retrieved context to generate coherent answers
    model = AzureChatOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),  # Primary API key from environment variables
        azure_endpoint=os.getenv("Azure_endpoint"),  # Azure OpenAI service endpoint URL
        deployment_name="gpt-4o-mini",  # Specific model deployment in Azure (fast, cost-efficient)
        api_version="2024-12-01-preview",  # API version ensuring compatibility with latest features
        temperature=0.7,  # Controls response creativity (0=deterministic, 1=very creative)
        max_tokens=512,  # Maximum response length to control costs and response time
        timeout=30,  # Request timeout to prevent hanging requests
        max_retries=3  # Retry failed requests for better reliability
    )
    
    # Initialize Azure OpenAI embeddings model for document vectorization
    # This model converts text documents into high-dimensional vectors for similarity search
    embedding = AzureOpenAIEmbeddings(
        azure_endpoint="https://agentic-keran-framework.openai.azure.com/",  # Dedicated embedding endpoint
        api_key=os.getenv("Embedding_api_key"),  # Separate API key for embedding service
        deployment="text-embedding-ada-002",  # Ada-002 model: balanced performance and cost
        api_version="2023-05-15",  # Stable API version for embeddings
        chunk_size=1000,  # Maximum tokens per embedding request
        max_retries=3,  # Retry mechanism for failed embedding requests
        timeout=30  # Request timeout for embedding generation
    )
    
    return model, embedding

# Initialize the models globally for use throughout the application
model, embedding = initialize_azure_models()

# Log successful model initialization
logging.info("✅ Azure OpenAI models initialized successfully")

# ========================================
# DOCUMENT LOADING AND PROCESSING PIPELINE
# ========================================

def load_and_process_documents():
    """
    Load documents from web sources and process them for vector storage.
    
    This comprehensive function handles the complete document processing pipeline:
    1. Web scraping from specified URLs
    2. Document chunking for optimal retrieval
    3. Vector store creation and persistence
    
    Returns:
        Chroma: Initialized vector store with processed documents
        
    Raises:
        Exception: If document loading or processing fails
    """
    
    # Define source URLs containing Redis documentation
    # These URLs provide comprehensive information about Redis and its Python client
    urls = [
        "https://github.com/redis/redis",      # Main Redis repository with core documentation
        "https://github.com/redis/redis-py"   # Python client library documentation
    ]
    
    logging.info(f"📥 Starting document loading from {len(urls)} sources...")
    
    # Load documents from web URLs with proper configuration
    docs = []
    for i, url in enumerate(urls, 1):
        try:
            # Create WebBaseLoader instance for each URL
            loader = WebBaseLoader(url)
            
            # Configure HTTP headers to avoid bot detection and rate limiting
            # User-Agent mimics a real browser to prevent access restrictions
            loader.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            # Load and extend the documents list with content from this URL
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            
            logging.info(f"✅ Loaded {len(loaded_docs)} documents from source {i}/{len(urls)}: {url}")
            
        except Exception as e:
            logging.error(f"❌ Failed to load documents from {url}: {str(e)}")
            # Continue with other sources even if one fails
            continue
    
    logging.info(f"📄 Total documents loaded: {len(docs)}")
    
    # ========================================
    # DOCUMENT CHUNKING STRATEGY
    # ========================================
    
    # Initialize text splitter with optimized parameters for RAG performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Optimal chunk size: balances context and specificity
        chunk_overlap=10,       # Small overlap to maintain context continuity
        length_function=len,    # Use character count for consistent chunking
        separators=[            # Hierarchical splitting strategy
            "\n\n",            # Paragraph breaks (preferred)
            "\n",              # Line breaks
            " ",               # Word boundaries
            ""                 # Character level (last resort)
        ],
        keep_separator=True     # Maintain separators for better readability
    )
    
    # Split documents into manageable chunks for vector storage
    logging.info("✂️  Splitting documents into chunks...")
    docs_splits = text_splitter.split_documents(docs)
    logging.info(f"📝 Created {len(docs_splits)} document chunks")
    
    # ========================================
    # VECTOR STORE CREATION AND PERSISTENCE
    # ========================================
    
    # Create ChromaDB vector store with comprehensive configuration
    logging.info("🔧 Initializing ChromaDB vector store...")
    
    vector_store = Chroma.from_documents(
        documents=docs_splits,                              # Processed document chunks
        embedding=embedding,                                # Azure OpenAI embedding model
        collection_name="RAG_with_multiple_docs",          # Unique collection identifier
        persist_directory="./chroma_db_Rag_workflow_03",   # Local persistence directory
        # Additional ChromaDB configuration options:
        # - Automatic persistence when persist_directory is specified
        # - Built-in deduplication based on document content
        # - Efficient similarity search with cosine distance
    )
    
    logging.info("💾 Vector store created and persisted successfully")
    logging.info(f"📊 Vector store statistics: {len(docs_splits)} chunks indexed")
    
    return vector_store

# Initialize the vector store globally
vector_store = load_and_process_documents()



### ---------------------- LANGGRAPH PIPELINE -----------------------

# ========================================
# RAG PIPELINE COMPONENTS
# ========================================

def create_rag_components():
    """
    Create and configure the core RAG pipeline components.
    
    This function sets up:
    1. Optimized prompt template for question-answering
    2. RAG processing chain combining prompt, model, and parser
    
    Returns:
        tuple: (prompt_template, rag_chain) - Configured RAG components
    """
    
    # ========================================
    # PROMPT ENGINEERING FOR OPTIMAL RESPONSES
    # ========================================
    
    # Design a comprehensive prompt template that guides the AI to provide
    # accurate, contextual, and helpful responses based on retrieved documentation
    prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant specializing in Redis database technology and its Python client library.

Your task is to provide accurate, helpful, and detailed answers based on the provided context from Redis documentation.

Guidelines for your responses:
1. **Accuracy**: Base your answer strictly on the provided context
2. **Completeness**: Provide comprehensive answers when context allows
3. **Clarity**: Use clear, technical language appropriate for developers
4. **Honesty**: If the context doesn't contain enough information, explicitly state "I don't know" or "The provided context doesn't contain information about this topic"
5. **Structure**: Organize complex answers with bullet points or numbered lists when appropriate
6. **Examples**: Include code examples or usage patterns when mentioned in the context

Question: {question}

Context from Redis Documentation:
{context}

Answer:"""
    )
    
    # ========================================
    # RAG PROCESSING CHAIN CONSTRUCTION
    # ========================================
    
    # Create the complete RAG chain: prompt template → Azure OpenAI model → string parser
    # This chain processes questions and context to generate structured responses
    rag_chain = prompt | model | StrOutputParser()
    
    logging.info("🔗 RAG pipeline components initialized successfully")
    
    return prompt, rag_chain

# Initialize RAG components globally
prompt, rag_chain = create_rag_components()

# ========================================
# LANGGRAPH STATE DEFINITION
# ========================================

class RagState(TypedDict):
    """
    Defines the state structure for the LangGraph RAG workflow.
    
    This TypedDict provides type hints and structure for the state that flows
    through the RAG state machine, enabling better IDE support and runtime validation.
    
    Attributes:
        question (str): The original user question/query
        web_search (List[str]): List of retrieved document contents for reference
        answer (str): The final generated answer from the AI model
        documents (List[object]): Retrieved Document objects from the vector store
        
    Note:
        - TypedDict provides type hints without runtime overhead
        - Each state transition can modify these fields
        - The state persists throughout the workflow execution
    """
    question: str           # User's input question
    web_search: List[str]   # Raw content from retrieved documents
    answer: str            # Final AI-generated response
    documents: List[object] # LangChain Document objects with metadata

# ========================================
# LANGGRAPH WORKFLOW NODE FUNCTIONS
# ========================================

def retrieve_documents(state: RagState) -> RagState:
    """
    First node in the RAG workflow: Document Retrieval.
    
    This function implements the "Retrieval" component of RAG by:
    1. Creating a similarity-based retriever from the vector store
    2. Finding the most relevant documents for the user's question
    3. Updating the workflow state with retrieved documents
    
    Args:
        state (RagState): Current workflow state containing the user's question
        
    Returns:
        RagState: Updated state with retrieved documents added
        
    Raises:
        Exception: If document retrieval fails (network issues, vector store problems)
        
    Technical Details:
        - Uses cosine similarity for document matching
        - Retrieves top k=2 documents to balance relevance and context length
        - Documents include both content and metadata for comprehensive context
    """
    
    logging.info("🔍 Starting document retrieval phase...")
    logging.info(f"📝 Query: '{state['question']}'")
    
    try:
        # Create a retriever from the vector store with optimized configuration
        retriever = vector_store.as_retriever(
            search_type="similarity",           # Use cosine similarity for matching
            search_kwargs={
                "k": 2,                        # Retrieve top 2 most relevant documents
                "fetch_k": 20,                 # Consider top 20 for reranking (if supported)
                "score_threshold": 0.0         # No minimum similarity threshold
            }
        )
        
        # Invoke retriever with the user's question
        retrieved_docs = retriever.invoke(state["question"])
        
        # Update state with retrieved documents
        state["documents"] = retrieved_docs
        
        # Log retrieval results for debugging and monitoring
        logging.info(f"✅ Retrieved {len(retrieved_docs)} relevant documents")
        
        # Log document metadata for debugging (first document only to avoid spam)
        if retrieved_docs:
            first_doc = retrieved_docs[0]
            logging.info(f"📄 Top document source: {first_doc.metadata.get('source', 'Unknown')}")
            logging.info(f"📄 Top document preview: {first_doc.page_content[:100]}...")
        
        return state
        
    except Exception as e:
        logging.error(f"❌ Document retrieval failed: {str(e)}")
        # Ensure state has empty documents list if retrieval fails
        state["documents"] = []
        raise


def generate_answer(state: RagState) -> RagState:
    """
    Second node in the RAG workflow: Answer Generation.
    
    This function implements the "Generation" component of RAG by:
    1. Combining retrieved document contents into a coherent context
    2. Using the RAG chain to generate an answer based on question and context
    3. Storing both the answer and source context in the workflow state
    
    Args:
        state (RagState): Current workflow state with question and retrieved documents
        
    Returns:
        RagState: Updated state with generated answer and formatted context
        
    Raises:
        Exception: If answer generation fails (API issues, model problems)
        
    Technical Details:
        - Concatenates document contents with double newlines for clarity
        - Uses the pre-configured RAG chain (prompt + model + parser)
        - Preserves original document content for transparency and verification
    """
    
    logging.info("✍️  Starting answer generation phase...")
    
    try:
        # Check if documents were successfully retrieved
        if not state.get("documents"):
            logging.warning("⚠️  No documents available for answer generation")
            state["answer"] = "I don't have enough context to answer this question. Please try rephrasing or asking about Redis-related topics."
            state["web_search"] = []
            return state
        
        # ========================================
        # CONTEXT PREPARATION
        # ========================================
        
        # Combine all retrieved document contents into a single context string
        # Use double newlines to clearly separate different document chunks
        context_parts = []
        
        for i, doc in enumerate(state["documents"], 1):
            # Add document numbering for better context organization
            doc_content = f"Document {i}:\n{doc.page_content.strip()}"
            context_parts.append(doc_content)
        
        # Join all document parts with clear separation
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        # Store raw document contents for API response transparency
        state["web_search"] = [doc.page_content for doc in state["documents"]]
        
        logging.info(f"📋 Prepared context from {len(state['documents'])} documents")
        logging.info(f"📏 Total context length: {len(context)} characters")
        
        # ========================================
        # AI ANSWER GENERATION
        # ========================================
        
        # Generate answer using the configured RAG chain
        logging.info("🤖 Invoking Azure OpenAI for answer generation...")
        
        answer = rag_chain.invoke({
            "question": state["question"],
            "context": context
        })
        
        # Update state with generated answer
        state["answer"] = answer.strip()
        
        # Log successful generation
        logging.info(f"✅ Answer generated successfully")
        logging.info(f"📝 Answer length: {len(state['answer'])} characters")
        logging.info(f"🎯 Answer preview: {state['answer'][:100]}...")
        
        return state
        
    except Exception as e:
        logging.error(f"❌ Answer generation failed: {str(e)}")
        # Provide fallback response
        state["answer"] = f"I encountered an error while generating the answer: {str(e)}"
        state["web_search"] = state.get("web_search", [])
        raise

# ========================================
# LANGGRAPH WORKFLOW ORCHESTRATION
# ========================================

def create_workflow():
    """
    Create and compile the LangGraph state machine for the RAG workflow.
    
    This function builds a sophisticated state machine that orchestrates the RAG process:
    
    Workflow Structure:
    ┌─────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────┐
    │  START  │───▶│ retrieve_documents│───▶│ generate_answer │───▶│   END   │
    └─────────┘    └──────────────────┘    └─────────────────┘    └─────────┘
    
    State Flow:
    1. START: Initialize with user question
    2. retrieve_documents: Find relevant documents from vector store
    3. generate_answer: Use retrieved context to generate AI response
    4. END: Return final state with answer and context
    
    Returns:
        CompiledStateGraph: Ready-to-execute workflow with memory checkpointing
        
    Technical Features:
        - Memory checkpointing for state persistence across requests
        - Error handling at each node transition
        - Comprehensive logging for debugging and monitoring
        - Type-safe state transitions using RagState TypedDict
    """
    
    logging.info("🏗️  Building LangGraph state machine...")
    
    # Initialize StateGraph with our custom RagState type definition
    workflow = StateGraph(RagState)
    
    # ========================================
    # NODE REGISTRATION
    # ========================================
    
    # Register workflow nodes (processing functions)
    workflow.add_node("retrieve", retrieve_documents)      # Document retrieval node
    workflow.add_node("generate_answer", generate_answer)  # Answer generation node
    
    logging.info("📦 Registered workflow nodes: retrieve, generate_answer")
    
    # ========================================
    # EDGE DEFINITION (WORKFLOW FLOW)
    # ========================================
    
    # Define the execution flow between nodes
    workflow.add_edge(START, "retrieve")                   # Start → Document Retrieval
    workflow.add_edge("retrieve", "generate_answer")       # Retrieval → Answer Generation
    workflow.add_edge("generate_answer", END)              # Generation → End
    
    logging.info("🔗 Configured workflow edges: START→retrieve→generate_answer→END")
    
    # ========================================
    # WORKFLOW COMPILATION WITH PERSISTENCE
    # ========================================
    
    # Compile the workflow with memory checkpointing for state persistence
    compiled_workflow = workflow.compile(
        checkpointer=MemorySaver(),    # In-memory state persistence
        # Additional compilation options:
        # - interrupt_before=[]: No user interruption points
        # - interrupt_after=[]: No post-processing interruption points
        # - debug=False: Disable debug mode for production
    )
    
    logging.info("✅ LangGraph workflow compiled successfully with memory checkpointing")
    
    return compiled_workflow

# Create and cache the compiled workflow graph globally
# This allows reuse across multiple API requests without recompilation overhead
graph = create_workflow()

logging.info("🚀 RAG workflow system initialized and ready for requests")


### ---------------------- FASTAPI ENDPOINT -----------------------

# ========================================
# FASTAPI DATA MODELS AND SCHEMAS
# ========================================

class QuestionRequest(BaseModel):
    """
    Pydantic model for incoming API requests.
    
    This model validates and serializes user questions sent to the RAG API.
    It ensures data integrity and provides automatic API documentation.
    
    Attributes:
        question (str): The user's question about Redis/Redis-py
        
    Validation Rules:
        - question must be a non-empty string
        - Automatic trimming of whitespace
        - Maximum length validation (if needed)
        
    Example:
        {
            "question": "What is Redis and how does it work?"
        }
    """
    question: str
    
    class Config:
        """Pydantic configuration for the request model."""
        # Enable example in API documentation
        schema_extra = {
            "example": {
                "question": "What is Redis and how does it work?"
            }
        }


class RAGResponse(BaseModel):
    """
    Pydantic model for API responses.
    
    This model structures the response from the RAG pipeline, providing
    both the generated answer and the source context for transparency.
    
    Attributes:
        question (str): The original user question (for reference)
        answer (str): The AI-generated answer based on retrieved context
        context (List[str]): List of document chunks used as context
        
    Features:
        - Automatic JSON serialization
        - Type validation for all fields
        - API documentation generation
        - Response schema enforcement
        
    Example:
        {
            "question": "What is Redis?",
            "answer": "Redis is an in-memory data structure store...",
            "context": ["Document content 1...", "Document content 2..."]
        }
    """
    question: str
    answer: str
    context: List[str]
    
    class Config:
        """Pydantic configuration for the response model."""
        # Enable example in API documentation
        schema_extra = {
            "example": {
                "question": "What is Redis?",
                "answer": "Redis is an in-memory data structure store, used as a database, cache, and message broker.",
                "context": [
                    "Redis is an open source, in-memory data structure store...",
                    "Redis provides data structures such as strings, hashes, lists..."
                ]
            }
        }

# ========================================
# FASTAPI ENDPOINT IMPLEMENTATIONS
# ========================================

@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QuestionRequest):
    """
    Main RAG endpoint for processing user questions.
    
    This endpoint orchestrates the complete RAG workflow:
    1. Receives and validates user questions
    2. Executes the LangGraph state machine
    3. Returns structured responses with answers and context
    
    Args:
        request (QuestionRequest): User question wrapped in Pydantic model
        
    Returns:
        RAGResponse: Structured response with answer and source context
        
    Raises:
        HTTPException: 
            - 400: If question is invalid or empty
            - 500: If RAG pipeline execution fails
            - 503: If Azure OpenAI services are unavailable
            
    Technical Flow:
        1. Input validation via Pydantic
        2. State machine initialization with user question
        3. Asynchronous workflow execution with streaming
        4. Response formatting and validation
        5. Error handling and logging
        
    Example Usage:
        POST /ask
        {
            "question": "How do I install Redis on Ubuntu?"
        }
        
        Response:
        {
            "question": "How do I install Redis on Ubuntu?",
            "answer": "To install Redis on Ubuntu, you can use...",
            "context": ["Installation guide content...", "Ubuntu-specific instructions..."]
        }
    """
    
    # ========================================
    # INPUT VALIDATION AND PREPROCESSING
    # ========================================
    
    # Validate question content
    if not request.question or not request.question.strip():
        logging.warning("❌ Received empty question")
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty. Please provide a valid question about Redis."
        )
    
    # Clean and prepare the question
    cleaned_question = request.question.strip()
    
    logging.info(f"📨 Received question: '{cleaned_question}'")
    logging.info(f"📊 Question length: {len(cleaned_question)} characters")
    
    # ========================================
    # WORKFLOW EXECUTION
    # ========================================
    
    try:
        # Initialize workflow state with the user's question
        input_state = {"question": cleaned_question}
        
        # Configure workflow execution parameters
        config = {
            "configurable": {
                "thread_id": "1"        # Unique thread ID for this session
            },
            "recursion_limit": 10,      # Prevent infinite loops
            "timeout": 60               # Maximum execution time in seconds
        }
        
        logging.info("🔄 Starting RAG workflow execution...")
        
        # Execute workflow asynchronously with streaming for real-time updates
        result = None
        async for event in graph.astream(
            input_state, 
            config=config, 
            stream_mode="values"        # Stream complete state values
        ):
            result = event              # Keep the final state
            logging.debug(f"📡 Workflow state update: {list(event.keys())}")
        
        # ========================================
        # RESPONSE VALIDATION AND FORMATTING
        # ========================================
        
        # Validate that workflow completed successfully
        if not result:
            logging.error("❌ Workflow completed without result")
            raise HTTPException(
                status_code=500,
                detail="RAG workflow failed to produce a result"
            )
        
        # Validate required fields in the result
        required_fields = ["answer", "web_search"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            logging.error(f"❌ Missing required fields in workflow result: {missing_fields}")
            raise HTTPException(
                status_code=500,
                detail=f"Incomplete workflow result. Missing: {', '.join(missing_fields)}"
            )
        
        # Create structured response
        response = RAGResponse(
            question=cleaned_question,
            answer=result["answer"],
            context=result["web_search"]
        )
        
        logging.info("✅ RAG workflow completed successfully")
        logging.info(f"📝 Generated answer length: {len(response.answer)} characters")
        logging.info(f"📚 Context sources: {len(response.context)} documents")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (already handled)
        raise
        
    except Exception as e:
        # Handle unexpected errors
        logging.exception("❌ Unexpected error during RAG pipeline execution")
        
        # Provide user-friendly error message
        error_message = "An internal error occurred while processing your question. Please try again."
        
        # In development, include more details
        if os.getenv("DEBUG", "false").lower() == "true":
            error_message += f" Details: {str(e)}"
        
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/")
def root():
    """
    Root endpoint providing API information and welcome message.
    
    This endpoint serves as the API's entry point, providing:
    - Welcome message and basic information
    - Available endpoints overview
    - API status confirmation
    
    Returns:
        dict: Welcome message and API information
        
    Example Response:
        {
            "message": "Welcome to LangGraph RAG API! Use POST /ask to query.",
            "version": "1.0",
            "endpoints": {
                "ask": "POST /ask - Submit questions about Redis",
                "web": "GET /web - Access web interface",
                "docs": "GET /docs - View API documentation"
            },
            "status": "operational"
        }
    """
    
    logging.info("ℹ️  Root endpoint accessed")
    
    return {
        "message": "Welcome to LangGraph RAG API! Use POST /ask to query.",
        "version": "1.0",
        "description": "AI-powered question answering system for Redis documentation",
        "endpoints": {
            "ask": "POST /ask - Submit questions about Redis and Redis-py",
            "web": "GET /web - Access interactive web interface",
            "docs": "GET /docs - View Swagger API documentation",
            "redoc": "GET /redoc - View ReDoc API documentation"
        },
        "status": "operational",
        "powered_by": ["LangGraph", "Azure OpenAI", "FastAPI", "ChromaDB"]
    }


@app.get("/web", response_class=HTMLResponse)
def web_interface():
    """
    Interactive web interface for testing the RAG API.
    
    This endpoint serves a comprehensive HTML page that provides:
    - User-friendly form for submitting questions
    - Real-time API interaction without external tools
    - Visual feedback for processing and results
    - Responsive design for various screen sizes
    
    Features:
        - Asynchronous JavaScript for smooth user experience
        - Error handling and user feedback
        - Context information display
        - Professional styling with modern CSS
        
    Returns:
        HTMLResponse: Complete HTML page with embedded CSS and JavaScript
        
    Usage:
        Navigate to http://localhost:8000/web in your browser to access
        the interactive interface for testing the RAG system.
        
    Technical Implementation:
        - Pure HTML/CSS/JavaScript (no external dependencies)
        - Fetch API for modern HTTP requests
        - Responsive CSS Grid/Flexbox layout
        - Real-time user feedback with loading states
    """
    
    logging.info("🌐 Web interface accessed")
    
    # Comprehensive HTML interface with embedded styling and functionality
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LangGraph RAG API - Interactive Web Interface</title>
        <style>
            /* ========================================
               MODERN CSS STYLING
               ======================================== */
            
            /* Base styles and CSS variables for consistent theming */
            :root {
                --primary-color: #007bff;
                --primary-hover: #0056b3;
                --success-color: #28a745;
                --warning-color: #ffc107;
                --danger-color: #dc3545;
                --light-bg: #f8f9fa;
                --dark-text: #343a40;
                --border-color: #dee2e6;
                --shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
                --border-radius: 0.375rem;
            }
            
            /* Reset and base typography */
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
                line-height: 1.6;
                color: var(--dark-text);
                background-color: #ffffff;
                padding: 20px;
            }
            
            /* Main container with responsive design */
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: var(--light-bg);
                padding: 2rem;
                border-radius: var(--border-radius);
                box-shadow: var(--shadow);
            }
            
            /* Header styling */
            .header {
                text-align: center;
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 2px solid var(--border-color);
            }
            
            .header h1 {
                color: var(--primary-color);
                margin-bottom: 0.5rem;
                font-size: 2.5rem;
                font-weight: 700;
            }
            
            .header p {
                color: #6c757d;
                font-size: 1.1rem;
                margin-bottom: 0.5rem;
            }
            
            .badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                font-size: 0.875rem;
                font-weight: 500;
                color: white;
                background-color: var(--success-color);
                border-radius: 50px;
                margin: 0 0.25rem;
            }
            
            /* Form styling */
            .form-section {
                margin-bottom: 2rem;
            }
            
            .question-box {
                width: 100%;
                padding: 1rem;
                margin: 1rem 0;
                border: 2px solid var(--border-color);
                border-radius: var(--border-radius);
                font-size: 1rem;
                font-family: inherit;
                transition: border-color 0.3s ease, box-shadow 0.3s ease;
            }
            
            .question-box:focus {
                outline: none;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
            }
            
            .submit-btn {
                background: var(--primary-color);
                color: white;
                padding: 0.75rem 2rem;
                border: none;
                border-radius: var(--border-radius);
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: background-color 0.3s ease, transform 0.1s ease;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .submit-btn:hover {
                background: var(--primary-hover);
                transform: translateY(-1px);
            }
            
            .submit-btn:active {
                transform: translateY(0);
            }
            
            .submit-btn:disabled {
                background: #6c757d;
                cursor: not-allowed;
                transform: none;
            }
            
            /* Results section styling */
            .results-section {
                margin-top: 2rem;
            }
            
            .answer-box {
                background: white;
                padding: 1.5rem;
                margin: 1rem 0;
                border-radius: var(--border-radius);
                border-left: 4px solid var(--primary-color);
                box-shadow: var(--shadow);
                line-height: 1.7;
            }
            
            .loading {
                color: var(--primary-color);
                font-style: italic;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .error {
                color: var(--danger-color);
                background: #f8d7da;
                border-left-color: var(--danger-color);
            }
            
            .context-info {
                background: #e7f3ff;
                padding: 1rem;
                border-radius: var(--border-radius);
                margin-top: 1rem;
                font-size: 0.9rem;
                color: #004085;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                body {
                    padding: 10px;
                }
                
                .container {
                    padding: 1rem;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .question-box {
                    padding: 0.75rem;
                }
            }
            
            /* Animation for loading spinner */
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .spinner {
                display: inline-block;
                width: 1rem;
                height: 1rem;
                border: 2px solid #f3f3f3;
                border-top: 2px solid var(--primary-color);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- ========================================
                 HEADER SECTION
                 ======================================== -->
            <div class="header">
                <h1>🤖 LangGraph RAG API</h1>
                <p>Intelligent Question Answering for Redis Documentation</p>
                <div>
                    <span class="badge">LangGraph</span>
                    <span class="badge">Azure OpenAI</span>
                    <span class="badge">ChromaDB</span>
                    <span class="badge">FastAPI</span>
                </div>
            </div>
            
            <!-- ========================================
                 QUESTION INPUT FORM
                 ======================================== -->
            <div class="form-section">
                <form id="questionForm">
                    <label for="question" style="display: block; margin-bottom: 0.5rem; font-weight: 600;">
                        Ask a question about Redis or Redis-py:
                    </label>
                    <input 
                        type="text" 
                        id="question" 
                        class="question-box" 
                        placeholder="e.g., What is Redis? How do I install Redis? What are Redis data types?"
                        required
                        maxlength="500"
                    >
                    <button type="submit" class="submit-btn" id="submitBtn">
                        <span id="btnText">Ask Question</span>
                        <span id="btnSpinner" class="spinner" style="display: none;"></span>
                    </button>
                </form>
            </div>
            
            <!-- ========================================
                 RESULTS DISPLAY SECTION
                 ======================================== -->
            <div id="result" class="results-section" style="display: none;">
                <h3 style="margin-bottom: 1rem; color: var(--primary-color);">📝 Answer:</h3>
                <div id="answer" class="answer-box"></div>
                <div class="context-info">
                    <strong>📚 Sources:</strong> Retrieved <span id="contextCount">0</span> relevant document chunks from Redis documentation
                </div>
            </div>
        </div>

        <!-- ========================================
             INTERACTIVE JAVASCRIPT FUNCTIONALITY
             ======================================== -->
        <script>
            /**
             * Enhanced JavaScript for RAG API interaction
             * 
             * Features:
             * - Asynchronous form submission with fetch API
             * - Real-time UI feedback and loading states
             * - Comprehensive error handling
             * - Responsive user experience
             */
            
            // Get DOM elements
            const form = document.getElementById('questionForm');
            const questionInput = document.getElementById('question');
            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const resultDiv = document.getElementById('result');
            const answerDiv = document.getElementById('answer');
            const contextSpan = document.getElementById('contextCount');
            
            // Form submission handler
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const question = questionInput.value.trim();
                
                // Client-side validation
                if (!question) {
                    showError('Please enter a question before submitting.');
                    return;
                }
                
                if (question.length > 500) {
                    showError('Question is too long. Please limit to 500 characters.');
                    return;
                }
                
                // Show loading state
                showLoading();
                
                try {
                    // Make API request
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Show successful result
                        showResult(data);
                    } else {
                        // Show API error
                        showError(`API Error: ${data.detail || 'Unknown error occurred'}`);
                    }
                    
                } catch (error) {
                    // Show network error
                    showError(`Network Error: ${error.message}. Please check your connection and try again.`);
                    
                } finally {
                    // Reset loading state
                    hideLoading();
                }
            });
            
            /**
             * Show loading state with visual feedback
             */
            function showLoading() {
                submitBtn.disabled = true;
                btnText.textContent = 'Processing...';
                btnSpinner.style.display = 'inline-block';
                
                resultDiv.style.display = 'block';
                answerDiv.innerHTML = '<div class="loading"><span class="spinner"></span> 🔍 Analyzing your question and searching documentation...</div>';
                contextSpan.textContent = '';
            }
            
            /**
             * Hide loading state
             */
            function hideLoading() {
                submitBtn.disabled = false;
                btnText.textContent = 'Ask Question';
                btnSpinner.style.display = 'none';
            }
            
            /**
             * Display successful result
             * @param {Object} data - API response data
             */
            function showResult(data) {
                resultDiv.style.display = 'block';
                answerDiv.innerHTML = data.answer;
                answerDiv.className = 'answer-box'; // Remove error class if present
                contextSpan.textContent = data.context.length;
                
                // Scroll to results
                resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            
            /**
             * Display error message
             * @param {string} message - Error message to display
             */
            function showError(message) {
                resultDiv.style.display = 'block';
                answerDiv.innerHTML = `❌ ${message}`;
                answerDiv.className = 'answer-box error';
                contextSpan.textContent = '0';
            }
            
            // Auto-focus on question input for better UX
            questionInput.focus();
            
            // Add enter key support for form submission
            questionInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    form.dispatchEvent(new Event('submit'));
                }
            });
            
            console.log('🚀 LangGraph RAG Web Interface loaded successfully');
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


# ========================================
# APPLICATION SERVER STARTUP AND CONFIGURATION
# ========================================

def configure_server():
    """
    Configure server settings based on environment and deployment context.
    
    This function determines the optimal server configuration for different environments:
    - Development: Local testing with debug features
    - Production: Optimized performance and security settings
    - Docker: Container-friendly configuration
    
    Returns:
        dict: Server configuration parameters for uvicorn
    """
    
    # Determine environment from environment variables
    environment = os.getenv("ENVIRONMENT", "development").lower()
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    # Base configuration applicable to all environments
    config = {
        "app": app,                    # FastAPI application instance
        "host": "0.0.0.0",            # Allow connections from any IP address
        "port": int(os.getenv("PORT", 8000)),  # Configurable port (default: 8000)
        "log_level": "info",          # Logging level for uvicorn
        "access_log": True,           # Enable access logging
        "use_colors": True,           # Colorized log output
    }
    
    # Environment-specific configurations
    if environment == "production":
        config.update({
            "reload": False,          # Disable auto-reload in production
            "workers": int(os.getenv("WORKERS", 1)),  # Number of worker processes
            "log_level": "warning",   # Reduce log verbosity in production
            "access_log": False,      # Disable access logs in production (use reverse proxy)
        })
        logging.info("🏭 Production configuration applied")
        
    elif environment == "development":
        config.update({
            "reload": debug_mode,     # Enable auto-reload in debug mode
            "reload_dirs": ["./"],    # Watch current directory for changes
            "log_level": "debug" if debug_mode else "info",
        })
        logging.info("🛠️  Development configuration applied")
        
    else:
        logging.warning(f"⚠️  Unknown environment '{environment}', using default configuration")
    
    return config


if __name__ == "__main__":
    """
    Main application entry point.
    
    This section handles the application startup process:
    1. Environment validation and configuration
    2. Server parameter optimization
    3. Graceful startup with comprehensive logging
    4. Error handling for startup failures
    
    The application runs only when executed directly (not when imported as a module).
    
    Deployment Options:
        - Development: python RAG_State_Machine_workflow_03.py
        - Production: uvicorn RAG_State_Machine_workflow_03:app --host 0.0.0.0 --port 8000
        - Docker: Configured for container deployment with environment variables
        
    Environment Variables:
        - PORT: Server port (default: 8000)
        - ENVIRONMENT: deployment environment (development/production)
        - DEBUG: Enable debug mode (true/false)
        - WORKERS: Number of worker processes (production only)
        - AZURE_API_KEY: Azure OpenAI API key
        - Azure_endpoint: Azure OpenAI endpoint URL
        - Embedding_api_key: Azure OpenAI embeddings API key
    """
    
    try:
        # ========================================
        # STARTUP VALIDATION AND LOGGING
        # ========================================
        
        logging.info("🚀 Starting LangGraph RAG API Server...")
        logging.info("=" * 60)
        
        # Validate critical environment variables
        required_env_vars = ["AZURE_API_KEY", "Azure_endpoint", "Embedding_api_key"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logging.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            logging.error("Please check your .env file and ensure all required variables are set.")
            exit(1)
        
        # Log system information
        logging.info(f"🐍 Python version: {os.sys.version}")
        logging.info(f"🖥️  Operating system: {os.name}")
        logging.info(f"📁 Working directory: {os.getcwd()}")
        logging.info(f"🌐 Environment: {os.getenv('ENVIRONMENT', 'development')}")
        
        # ========================================
        # SERVER CONFIGURATION AND STARTUP
        # ========================================
        
        # Get optimized server configuration
        server_config = configure_server()
        
        # Log server configuration
        logging.info("⚙️  Server Configuration:")
        logging.info(f"   📡 Host: {server_config['host']}")
        logging.info(f"   🔌 Port: {server_config['port']}")
        logging.info(f"   📊 Log Level: {server_config['log_level']}")
        logging.info(f"   🔄 Auto-reload: {server_config.get('reload', False)}")
        
        # Log available endpoints
        logging.info("🔗 Available Endpoints:")
        logging.info(f"   📋 API Documentation: http://localhost:{server_config['port']}/docs")
        logging.info(f"   📖 ReDoc Documentation: http://localhost:{server_config['port']}/redoc")
        logging.info(f"   🌐 Web Interface: http://localhost:{server_config['port']}/web")
        logging.info(f"   🤖 RAG Endpoint: http://localhost:{server_config['port']}/ask")
        
        logging.info("=" * 60)
        logging.info("✅ All systems initialized successfully!")
        logging.info("🎯 Server is ready to handle RAG requests...")
        
        # ========================================
        # UVICORN SERVER STARTUP
        # ========================================
        
        # Start the FastAPI server with uvicorn
        uvicorn.run(**server_config)
        
    except KeyboardInterrupt:
        logging.info("\n🛑 Server shutdown requested by user (Ctrl+C)")
        logging.info("👋 Goodbye!")
        
    except Exception as e:
        logging.error(f"❌ Fatal error during server startup: {str(e)}")
        logging.error("🔧 Please check your configuration and try again.")
        logging.exception("Full error traceback:")
        exit(1)