from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os
from typing import Dict, List, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
import pydantic
from pydantic import Field
import numpy as np

load_dotenv()


embedding = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/", 
    api_key=os.getenv("Embedding_api_key"),  # Replace with actual key
    deployment="text-embedding-ada-002",  # Deployment name must match what you named the embedding model in Azure is very important
    model="text-embedding-ada-002",       # Model name (optional, but   helps for clarity)
    api_version="2023-05-15",        # Use correct version (based on Azure docs or trial-and-error)
)

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),  # API key from environment variables
    azure_endpoint=os.getenv("Azure_endpoint"),  # Azure OpenAI endpoint URL
    deployment_name="gpt-4o-mini",  # Specific deployment name in Azure
    api_version="2024-12-01-preview",  # API version for compatibility
    temperature=0.7,  # Controls randomness (0=deterministic, 1=creative)
    max_tokens=512  # Maximum tokens in the response
)

# Initialize embeddings
embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("Azure_endpoint"),  # Use environment variable
    api_key=os.getenv("Embedding_api_key"),
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    api_version="2023-05-15",
)

# Initialize model
model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    deployment_name="gpt-4o-mini",
    api_version="2024-12-01-preview",
    temperature=0.7,
    max_tokens=512
)

# Load and process documents
urls = [
    "https://www.oreilly.com/radar/agentic-rag/",
    "https://github.com/infiniflow/ragflow",
    "https://github.com/NirDiamant/RAG_Techniques"
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [doc for sublist in docs for doc in sublist]

# Create improved chunking with better parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # Increased overlap for better context
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    length_function=len,
)
docs_split = text_splitter.split_documents(docs_list)

# Create vector store with ChromaDB
vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embedding,
    collection_name="advanced_rag",
)

# Enhanced retriever with MMR for diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
)

# Enhanced prompts
rag_prompt = ChatPromptTemplate.from_template(
    """You are an expert assistant with deep knowledge in RAG (Retrieval-Augmented Generation) techniques.
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    1. Answer the question based solely on the provided context
    2. If the context doesn't contain sufficient information, clearly state what information is missing
    3. Provide specific examples and details when available
    4. Structure your answer in a clear and organized manner
    
    Answer:"""
)

# Enhanced RAG chain
rag_chain = (
    rag_prompt
    | model
    | StrOutputParser()
)

# Enhanced state for the graph
class GraphState(TypedDict):
    question: str
    original_question: str
    context: List[Document]
    answer: str
    retrieval_attempts: int
    confidence_score: float

# Enhanced scoring models
class RelevanceScore(pydantic.BaseModel):
    score: bool = Field(description="True if context is relevant to the question")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the relevance assessment")

class HallucinationChecker(pydantic.BaseModel):
    is_hallucination: bool = Field(description="True if the answer contains hallucinations")
    confidence: float = Field(description="Confidence score between 0 and 1")
    problematic_claims: List[str] = Field(description="List of potentially hallucinated claims")

class AnswerVerifier(pydantic.BaseModel):
    is_answer_correct: bool = Field(description="True if the answer correctly addresses the question")
    completeness_score: float = Field(description="Score from 0-1 indicating answer completeness")
    missing_aspects: List[str] = Field(description="List of aspects not covered in the answer")

# Enhanced prompt templates
retrieval_prompt = ChatPromptTemplate.from_template(
    """You are an expert in assessing document relevance for RAG systems.
    
    Document: {context}
    Question: {question}
    
    Assess whether this document contains information relevant to answering the question.
    Consider:
    1. Direct relevance to the question topic
    2. Presence of supporting facts or examples
    3. Contextual information that helps answer the question
    
    Provide your assessment with confidence score and reasoning."""
)

hallucination_prompt = ChatPromptTemplate.from_template(
    """You are an expert in detecting hallucinations in AI-generated content.
    
    Context provided: {context}
    Generated answer: {answer}
    
    Analyze the answer for:
    1. Claims not supported by the context
    2. Fabricated details or facts
    3. Incorrect interpretations of the context
    
    List any problematic claims found."""
)


# Continuing from where we left off...

answer_verification_prompt = ChatPromptTemplate.from_template(
    """You are an expert in evaluating answer quality for RAG systems.
    
    Question: {question}
    Context: {context}
    Answer: {answer}
    
    Evaluate:
    1. Whether the answer directly addresses the question
    2. Completeness of the answer (0-1 score)
    3. Any important aspects missing from the answer
    
    Be thorough in your assessment."""
)

# Create structured output chains
retrieval_checker = retrieval_prompt | model.with_structured_output(RelevanceScore)
hallucination_scorer = hallucination_prompt | model.with_structured_output(HallucinationChecker)
answer_verification_scorer = answer_verification_prompt | model.with_structured_output(AnswerVerifier)

# Enhanced node functions
def retrieve_documents(state: GraphState) -> Dict:
    """Enhanced retrieval with reranking and multiple strategies"""
    question = state['question']
    
    # Use the reranked retriever
    context = reranked_retriever.get_relevant_documents(question)
    
    # Track retrieval attempts
    attempts = state.get('retrieval_attempts', 0) + 1
    
    return {
        "context": context,
        "retrieval_attempts": attempts
    }

def check_relevance(state: GraphState) -> Dict:
    """Enhanced relevance checking with confidence scoring"""
    question = state['question']
    context = state['context']
    
    relevant_docs = []
    total_confidence = 0.0
    
    for doc in context:
        relevance_result = retrieval_checker.invoke({
            "context": doc.page_content,
            "question": question
        })
        
        if relevance_result.score:
            relevant_docs.append(doc)
            total_confidence += relevance_result.confidence
    
    avg_confidence = total_confidence / len(context) if context else 0.0
    
    return {
        "context": relevant_docs,
        "confidence_score": avg_confidence
    }

def generate_answer(state: GraphState) -> Dict:
    """Generate answer with context awareness"""
    question = state['question']
    context = state['context']
    
    # Prepare context string with metadata
    context_parts = []
    for i, doc in enumerate(context):
        # Include source information if available
        source = doc.metadata.get('source', 'Unknown')
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    
    context_str = "\n\n".join(context_parts)
    
    answer = rag_chain.invoke({
        "question": question,
        "context": context_str
    })
    
    return {"answer": answer}

def check_answer_quality(state: GraphState) -> str:
    """Enhanced answer quality check with multiple criteria"""
    answer = state['answer']
    context = state['context']
    question = state['question']
    
    context_str = "\n".join([doc.page_content for doc in context])
    
    # Check for hallucinations
    hallucination_result = hallucination_scorer.invoke({
        "context": context_str,
        "answer": answer
    })
    
    if hallucination_result.is_hallucination and hallucination_result.confidence > 0.7:
        return "hallucinated"
    
    # Verify answer correctness and completeness
    verification_result = answer_verification_scorer.invoke({
        "context": context_str,
        "answer": answer,
        "question": question
    })
    
    if not verification_result.is_answer_correct:
        return "incorrect"
    
    if verification_result.completeness_score < 0.6:
        return "incomplete"
    
    return "good"

def should_continue_retrieval(state: GraphState) -> str:
    """Decide whether to continue with retrieval or transform query"""
    confidence = state.get('confidence_score', 0)
    attempts = state.get('retrieval_attempts', 0)
    
    # If we have good context or exceeded max attempts
    if confidence > 0.7 or attempts >= 3:
        if state['context']:
            return "generate"
        else:
            return "no_context"
    
    # Low confidence, try query transformation
    return "transform"

def transform_query(state: GraphState) -> Dict:
    """Advanced query transformation with context awareness"""
    question = state['question']
    original_question = state.get('original_question', question)
    attempts = state.get('retrieval_attempts', 0)
    
    # Different transformation strategies based on attempts
    if attempts == 1:
        # First attempt: Make more specific
        transform_prompt = ChatPromptTemplate.from_template(
            """Transform this question to be more specific and detailed:
            Original: {question}
            
            Add specific technical terms, expand acronyms, and include related concepts."""
        )
    elif attempts == 2:
        # Second attempt: Break down into sub-questions
        transform_prompt = ChatPromptTemplate.from_template(
            """Break down this complex question into simpler components:
            Original: {original}
            Current: {question}
            
            Focus on the core concept that needs to be retrieved."""
        )
    else:
        # Final attempt: Rephrase with synonyms
        transform_prompt = ChatPromptTemplate.from_template(
            """Rephrase this question using alternative terminology:
            Original: {original}
            Current: {question}
            
            Use synonyms and related terms that might match the documents better."""
        )
    
    transformer = transform_prompt | model | StrOutputParser()
    
    transformed = transformer.invoke({
        "question": question,
        "original": original_question
    })
    
    return {
        "question": transformed,
        "original_question": original_question
    }

def generate_fallback_answer(state: GraphState) -> Dict:
    """Generate a helpful response when no relevant content is found"""
    original_question = state.get('original_question', state['question'])
    
    fallback_prompt = ChatPromptTemplate.from_template(
        """As an expert in RAG systems, provide a helpful response explaining that
        no relevant information was found for this question, and suggest:
        1. How the user might rephrase their question
        2. What type of information might be available
        3. Alternative resources they could consult
        
        Question: {question}
        
        Be helpful and constructive in your response."""
    )
    
    fallback_chain = fallback_prompt | model | StrOutputParser()
    
    answer = fallback_chain.invoke({"question": original_question})
    
    return {"answer": answer}

# Build the enhanced workflow graph
workflow = StateGraph(GraphState)

# Add all nodes
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("transform_query", transform_query)
workflow.add_node("fallback_answer", generate_fallback_answer)

# Set entry point
workflow.set_entry_point("retrieve")

# Add edges
workflow.add_edge("retrieve", "check_relevance")

# Conditional routing after relevance check
workflow.add_conditional_edges(
    "check_relevance",
    should_continue_retrieval,
    {
        "generate": "generate_answer",
        "transform": "transform_query",
        "no_context": "fallback_answer"
    }
)

# Loop back from query transformation
workflow.add_edge("transform_query", "retrieve")

# Conditional routing after answer generation
workflow.add_conditional_edges(
    "generate_answer",
    check_answer_quality,
    {
        "good": END,
        "hallucinated": "transform_query",
        "incorrect": "transform_query",
        "incomplete": "transform_query"
    }
)

# End after fallback answer
workflow.add_edge("fallback_answer", END)

# Compile the graph
app = workflow.compile()

# Utility functions for analysis
def analyze_retrieval_performance(question: str, verbose: bool = True):
    """Analyze retrieval performance across different strategies"""
    results = {}
    
    # Test each retriever
    retrievers = {
        "Ensemble": ensemble_retriever,
        "Multi-Query": multi_query_retriever,
        "MMR": retriever_mmr,
        "BM25": bm25_retriever,
        "Parent-Child": parent_child_retriever,
        "Reranked": reranked_retriever
    }
    
    for name, retriever in retrievers.items():
        docs = retriever.get_relevant_documents(question)
        results[name] = {
            "count": len(docs),
            "avg_length": np.mean([len(doc.page_content) for doc in docs]) if docs else 0,
            "sources": list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
        }
        
        if verbose:
            print(f"\n{name} Retriever:")
            print(f"  Documents: {results[name]['count']}")
            print(f"  Avg Length: {results[name]['avg_length']:.0f}")
            print(f"  Sources: {results[name]['sources']}")
    
    return results

# Enhanced execution function
def run_rag_pipeline(question: str, analyze: bool = False):
    """Run the RAG pipeline with optional analysis"""
    
    if analyze:
        print("Analyzing retrieval strategies...")
        analyze_retrieval_performance(question)
        print("\n" + "="*50 + "\n")
    
    # Initialize state
    initial_state = {
        "question": question,
        "original_question": question,
        "context": [],
        "answer": "",
        "retrieval_attempts": 0,
        "confidence_score": 0.0
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Format output
    print(f"Question: {question}")
    print(f"Retrieval Attempts: {result.get('retrieval_attempts', 1)}")
    print(f"Confidence Score: {result.get('confidence_score', 0):.2f}")
    print(f"\nAnswer:\n{result['answer']}")
    
    # Show sources if available
    if result['context']:
        print("\nSources used:")
        sources = set()
        for doc in result['context']:
            source = doc.metadata.get('source', 'Unknown')
            sources.add(source)
        for source in sources:
            print(f"  - {source}")
    
    return result

# Example usage with different test cases
if __name__ == "__main__":
    # Test questions
    test_questions = [
        "What are the different search methods in RAG?",
        "How does semantic search differ from keyword search in retrieval systems?",
        "Explain the concept of hybrid search in RAG applications",
        "What are the best practices for chunking documents in RAG?"
    ]
    
    # Run tests
    for question in test_questions:
        print("\n" + "="*80 + "\n")
        run_rag_pipeline(question, analyze=True)
        print("\n" + "="*80 + "\n")

# Additional utility for batch processing
def batch_process_questions(questions: List[str], export_results: bool = False):
    """Process multiple questions and optionally export results"""
    results = []
    
    for question in questions:
        result = app.invoke({
            "question": question,
            "original_question": question,
            "context": [],
            "answer": "",
            "retrieval_attempts": 0,
            "confidence_score": 0.0
        })
        
        results.append({
            "question": question,
            "answer": result['answer'],
            "confidence": result.get('confidence_score', 0),
            "attempts": result.get('retrieval_attempts', 1)
        })
    
    if export_results:
        import json
        with open('rag_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

# Configuration for production use
RAG_CONFIG = {
    "retrieval": {
        "ensemble_weights": [0.4, 0.3, 0.3],
        "rerank_top_k": 5,
        "max_retrieval_attempts": 3,
        "confidence_threshold": 0.7
    },
    "chunking": {
        "standard_chunk_size": 1000,
        "standard_overlap": 200 }
        }