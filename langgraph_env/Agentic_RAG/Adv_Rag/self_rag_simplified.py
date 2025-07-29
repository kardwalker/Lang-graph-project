"""
Enhanced Self-RAG Implementation with Improved Workflow
Incorporates true self-reflection mechanisms and adaptive retrieval strategies

This implementation creates an advanced RAG (Retrieval-Augmented Generation) system that:
1. Self-reflects on whether it needs more information before answering
2. Adaptively retrieves documents based on previous attempts and context quality
3. Assesses the relevance and utility of retrieved information at multiple stages
4. Makes intelligent decisions about when to stop, continue, or regenerate responses
5. Maintains a complete history of all reflection decisions for transparency

Key Components:
- Self-Reflection Mechanisms: The system evaluates its own knowledge and decisions
- Adaptive Retrieval: Retrieval strategy changes based on previous attempts
- Multi-Stage Assessment: Relevance, support, and utility evaluations
- State Management: Comprehensive tracking of the entire reasoning process
- Conditional Workflows: Dynamic routing based on assessment outcomes
"""

import os
import getpass
from typing import List, Dict, TypedDict, Optional, Literal
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
import pydantic
from pydantic import BaseModel, Field
import time
from dataclasses import dataclass
from enum import Enum

# Load environment variables from .env file
# This allows us to keep sensitive API keys and configuration separate from code
load_dotenv()

# Initialize Azure OpenAI models for embeddings and chat completion
# These are the core AI components that power the RAG system
embedding = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/", 
    api_key=os.getenv("Embedding_api_key"),
    deployment="text-embedding-ada-002",  # Model for converting text to vector embeddings
    model="text-embedding-ada-002",
    api_version="2023-05-15",
)

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    deployment_name="gpt-4o-mini",  # Main language model for reasoning and generation
    api_version="2024-12-01-preview",
    temperature=0.1,  # Low temperature for consistent, deterministic responses
    max_tokens=1024   # Sufficient tokens for detailed reflective responses
)

# Self-RAG Reflection Tokens and Actions
# These enums define the vocabulary for the system's self-reflection capabilities
# They represent the different types of decisions and assessments the system can make

class ReflectionAction(str, Enum):
    """Actions the system can decide to take during self-reflection"""
    RETRIEVE = "retrieve"        # Get more documents from the knowledge base
    NO_RETRIEVE = "no_retrieve"  # Skip retrieval and proceed with current context
    CONTINUE = "continue"        # Continue with current workflow step
    REGENERATE = "regenerate"    # Regenerate the answer with same context
    END = "end"                  # Finish the process and return final answer

class SupportLevel(str, Enum):
    """How well the retrieved context supports the generated answer"""
    FULLY_SUPPORTED = "fully_supported"        # All claims backed by evidence
    PARTIALLY_SUPPORTED = "partially_supported" # Some claims lack support
    NOT_SUPPORTED = "not_supported"             # Claims contradict or lack evidence

class RelevanceLevel(str, Enum):
    """Relevance of retrieved documents to the user's question"""
    HIGHLY_RELEVANT = "highly_relevant"      # Directly answers the question
    RELEVANT = "relevant"                    # Contains useful related information
    PARTIALLY_RELEVANT = "partially_relevant" # Some useful parts, some irrelevant
    NOT_RELEVANT = "not_relevant"            # Does not help answer the question

class UtilityLevel(str, Enum):
    """Overall usefulness of the generated answer to the user"""
    VERY_USEFUL = "very_useful"           # Comprehensive, clear, actionable
    USEFUL = "useful"                     # Good answer with minor gaps
    SOMEWHAT_USEFUL = "somewhat_useful"   # Partial answer, needs improvement
    NOT_USEFUL = "not_useful"             # Inadequate or confusing response

# Reflection Decision Models
# These Pydantic models define the structure of self-reflection outputs
# Each model captures different aspects of the system's decision-making process

class RetrievalDecision(BaseModel):
    """Model for initial retrieval decisions - whether to search for more information"""
    should_retrieve: bool = Field(description="Whether to retrieve more information")
    action: ReflectionAction = Field(description="Next action to take")
    reasoning: str = Field(description="Reasoning behind the decision")
    confidence: float = Field(description="Confidence in decision (0-1)")

class RelevanceAssessment(BaseModel):
    """Model for assessing relevance of retrieved documents to the question"""
    relevance: RelevanceLevel = Field(description="Relevance level of retrieved documents")
    reasoning: str = Field(description="Detailed reasoning for relevance assessment")
    confidence: float = Field(description="Confidence in assessment (0-1)")
    should_use: bool = Field(description="Whether to use these documents")

class SupportAssessment(BaseModel):
    """Model for evaluating how well context supports the generated answer"""
    support_level: SupportLevel = Field(description="How well context supports the answer")
    reasoning: str = Field(description="Reasoning for support level")
    confidence: float = Field(description="Confidence in assessment (0-1)")
    citations: List[str] = Field(description="Specific parts of context that support answer")

class UtilityAssessment(BaseModel):
    """Model for final evaluation of answer quality and usefulness"""
    utility: UtilityLevel = Field(description="Utility of the generated answer")
    reasoning: str = Field(description="Why this answer is/isn't useful")
    confidence: float = Field(description="Confidence in utility assessment (0-1)")
    next_action: ReflectionAction = Field(description="Recommended next action")

# Enhanced State with Reflection History
# This TypedDict defines the complete state that flows through the system
# It tracks all aspects of the reasoning process including history and metrics
class GraphState(TypedDict):
    question: str                    # Original user question
    context: List[Document]          # Retrieved documents from vector store
    answer: str                      # Generated answer
    reflection_history: List[Dict]   # Complete history of all reflection decisions
    retrieval_attempts: int          # Number of retrieval operations performed
    generation_attempts: int         # Number of answer generation attempts
    current_phase: str               # Current stage in the workflow
    final_assessment: Dict           # Final evaluation and metrics

# Document Processing (enhanced from original)
def setup_vectorstores():
    """
    Setup multiple vectorstores with different chunking strategies
    
    This function:
    1. Loads documents from various web sources about RAG techniques
    2. Splits documents into chunks for better retrieval
    3. Creates embeddings and stores them in a Chroma vectorstore
    4. Returns a retriever configured for maximum marginal relevance (MMR) search
    
    Returns:
        Retriever object configured for adaptive document retrieval
    """
    # Define URLs containing RAG and agentic AI information
    urls = [
        "https://www.oreilly.com/radar/agentic-rag/",
        "https://github.com/infiniflow/ragflow",
        "https://github.com/NirDiamant/RAG_Techniques"
    ]
    
    # Load documents from all URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [doc for sublist in docs for doc in sublist]
    
    # Configure text splitter for optimal chunk sizes
    # Smaller chunks = more precise retrieval but may lose context
    # Larger chunks = more context but may include irrelevant information
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # Size of each text chunk
        chunk_overlap=150,     # Overlap between chunks to maintain context
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # Split hierarchy
        length_function=len,   # Function to measure chunk length
    )
    chunks = text_splitter.split_documents(docs_list)
    
    # Create vector store with embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="enhanced_rag",
        persist_directory="./chroma_db_enhanced"  # Persist to disk for reuse
    )
    
    # Return retriever with MMR (Maximum Marginal Relevance) search
    # MMR balances relevance and diversity in retrieved documents
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,              # Number of documents to retrieve
            "fetch_k": 15,       # Number of documents to consider before MMR filtering
            "lambda_mult": 0.6   # Balance between relevance (1.0) and diversity (0.0)
        }
    )

retriever = setup_vectorstores()

# Enhanced Prompt Templates with Self-Reflection
retrieval_decision_prompt = ChatPromptTemplate.from_template(
    """You are a self-reflective RAG system. Analyze whether you need to retrieve information to answer this question.

Question: {question}
Current context: {current_context}
Previous retrieval attempts: {retrieval_attempts}
Generation attempts: {generation_attempts}

Consider:
1. Do you have sufficient information to provide a complete, accurate answer?
2. Would additional retrieval improve answer quality significantly?
3. Have you already retrieved enough relevant information?
4. Is the question complex enough to warrant multiple retrieval rounds?

Previous reflection history:
{reflection_history}

Make a decision about whether to retrieve more information and explain your reasoning."""
)

relevance_assessment_prompt = ChatPromptTemplate.from_template(
    """Assess the relevance of retrieved documents for answering the question.

Question: {question}
Retrieved Documents:
{context}

Evaluate each document's relevance:
1. Does it directly address the question?
2. Does it provide background information that helps?
3. Does it contain specific examples, comparisons, or technical details?
4. How recent and authoritative is the information?

Provide detailed reasoning for your relevance assessment."""
)

generation_prompt = ChatPromptTemplate.from_template(
    """Generate a comprehensive answer using the provided context.

Question: {question}
Context:
{context}

Instructions:
1. Use only information from the provided context
2. Structure your answer logically with clear sections
3. Include specific examples and technical details when available
4. If context is insufficient, clearly state what information is missing
5. Provide citations to specific parts of the context

Answer:"""
)

support_assessment_prompt = ChatPromptTemplate.from_template(
    """Evaluate how well the context supports your generated answer.

Question: {question}
Context: {context}
Generated Answer: {answer}

Assessment criteria:
1. Are all claims in the answer supported by the context?
2. Did you make any inferences not directly stated in context?
3. Are there any potential factual errors or contradictions?
4. What specific parts of context support key points in your answer?

Provide detailed analysis of support level with specific citations."""
)

utility_assessment_prompt = ChatPromptTemplate.from_template(
    """Evaluate the utility and completeness of the generated answer.

Question: {question}
Answer: {answer}
Context used: {context}

Evaluation criteria:
1. Does the answer fully address all aspects of the question?
2. Is it clear, well-structured, and easy to understand?
3. Does it provide sufficient detail and examples?
4. Would a user be satisfied with this answer?
5. What could be improved?

Based on this assessment, recommend the next action."""
)

# Create reflection chains - these combine prompts with the language model
# Each chain handles a specific type of reflection or decision-making
retrieval_decision_chain = retrieval_decision_prompt | model.with_structured_output(RetrievalDecision)
relevance_assessment_chain = relevance_assessment_prompt | model.with_structured_output(RelevanceAssessment)
generation_chain = generation_prompt | model | StrOutputParser()
support_assessment_chain = support_assessment_prompt | model.with_structured_output(SupportAssessment)
utility_assessment_chain = utility_assessment_prompt | model.with_structured_output(UtilityAssessment)

# Enhanced Node Functions with Self-Reflection
def initial_reflection(state: GraphState) -> Dict:
    """
    Initial self-reflection to decide if retrieval is needed
    
    This function performs the first reflection step where the system analyzes:
    - Whether it has enough knowledge to answer the question
    - If additional information retrieval would be beneficial
    - The complexity of the question and what it might require
    
    Args:
        state: Current graph state containing question and history
    
    Returns:
        Dict with updated reflection history and current phase
    """
    question = state['question']
    current_context = "None" if not state.get('context') else f"{len(state['context'])} documents"
    
    # Invoke the retrieval decision chain to get AI's reflection
    decision = retrieval_decision_chain.invoke({
        "question": question,
        "current_context": current_context,
        "retrieval_attempts": state.get('retrieval_attempts', 0),
        "generation_attempts": state.get('generation_attempts', 0),
        "reflection_history": str(state.get('reflection_history', []))
    })
    
    # Create a structured record of this reflection step
    reflection_entry = {
        "phase": "initial_reflection",
        "decision": decision.action,
        "reasoning": decision.reasoning,
        "confidence": decision.confidence,
        "timestamp": time.time()
    }
    
    # Add to reflection history for tracking decision-making process
    history = state.get('reflection_history', [])
    history.append(reflection_entry)
    
    return {
        "reflection_history": history,
        "current_phase": "initial_reflection"
    }

def adaptive_retrieve(state: GraphState) -> Dict:
    """
    Retrieve documents with adaptive strategy based on reflection history
    
    This function implements intelligent retrieval that adapts based on:
    - Number of previous retrieval attempts
    - Quality of previously retrieved documents
    - Feedback from relevance assessments
    
    Strategy:
    - First attempt: Standard approach (6 documents)
    - Second attempt: More diverse results (8 documents)  
    - Further attempts: Focus on precision (4 documents)
    
    Args:
        state: Current graph state with question and retrieval history
    
    Returns:
        Dict with updated context and retrieval attempt count
    """
    question = state['question']
    retrieval_attempts = state.get('retrieval_attempts', 0)
    
    # Adjust retrieval strategy based on previous attempts
    if retrieval_attempts == 0:
        # First retrieval - standard approach
        k = 6
    elif retrieval_attempts == 1:
        # Second retrieval - get more diverse results
        k = 8
    else:
        # Subsequent retrievals - focus on precision
        k = 4
    
    # Retrieve documents using the configured strategy
    documents = retriever.invoke(question)[:k]
    
    # Add to existing context (don't replace to accumulate knowledge)
    existing_context = state.get('context', [])
    
    # Simple deduplication to avoid duplicate content
    # Compare first 100 characters to identify similar documents
    existing_content = {doc.page_content[:100] for doc in existing_context}
    new_docs = [doc for doc in documents 
                if doc.page_content[:100] not in existing_content]
    
    # Combine existing and new documents
    combined_context = existing_context + new_docs
    
    return {
        "context": combined_context,
        "retrieval_attempts": retrieval_attempts + 1
    }

def assess_relevance(state: GraphState) -> Dict:
    """
    Assess relevance of retrieved documents to the user's question
    
    This function evaluates whether the retrieved documents are actually useful
    for answering the question. It considers:
    - Direct relevance to the question
    - Quality and depth of information
    - Authority and recency of sources
    
    Args:
        state: Current graph state with question and retrieved context
    
    Returns:
        Dict with updated reflection history including relevance assessment
    """
    question = state['question']
    context = "\n\n".join([f"Doc {i+1}: {doc.page_content}" 
                          for i, doc in enumerate(state['context'])])
    
    # Use the relevance assessment chain to evaluate documents
    assessment = relevance_assessment_chain.invoke({
        "question": question,
        "context": context
    })
    
    # Record this assessment in the reflection history
    reflection_entry = {
        "phase": "relevance_assessment",
        "relevance": assessment.relevance,
        "reasoning": assessment.reasoning,
        "confidence": assessment.confidence,
        "should_use": assessment.should_use,
        "timestamp": time.time()
    }
    
    history = state.get('reflection_history', [])
    history.append(reflection_entry)
    
    return {
        "reflection_history": history,
        "current_phase": "relevance_assessment"
    }

def generate_with_reflection(state: GraphState) -> Dict:
    """
    Generate answer with self-reflection capabilities
    
    This function creates an answer based on the retrieved context.
    It focuses on:
    - Using only information from the provided context
    - Structuring the answer logically
    - Including specific examples and citations
    - Being clear about limitations if context is insufficient
    
    Args:
        state: Current graph state with question and context
    
    Returns:
        Dict with generated answer and updated generation attempt count
    """
    question = state['question']
    context = "\n\n".join([doc.page_content for doc in state['context']])
    
    # Generate answer using the generation chain
    answer = generation_chain.invoke({
        "question": question,
        "context": context
    })
    
    generation_attempts = state.get('generation_attempts', 0) + 1
    
    return {
        "answer": answer,
        "generation_attempts": generation_attempts,
        "current_phase": "generation"
    }

def assess_support(state: GraphState) -> Dict:
    """
    Assess how well the context supports the generated answer
    
    This function evaluates whether the generated answer is well-supported
    by the retrieved context. It checks for:
    - Factual accuracy and evidence backing
    - Potential contradictions or gaps
    - Quality of citations and references
    - Inference vs. direct statement analysis
    
    Args:
        state: Current graph state with question, context, and answer
    
    Returns:
        Dict with updated reflection history including support assessment
    """
    question = state['question']
    context = "\n\n".join([doc.page_content for doc in state['context']])
    answer = state['answer']
    
    # Evaluate how well context supports the answer
    assessment = support_assessment_chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    })
    
    # Record support assessment in reflection history
    reflection_entry = {
        "phase": "support_assessment",
        "support_level": assessment.support_level,
        "reasoning": assessment.reasoning,
        "confidence": assessment.confidence,
        "citations": assessment.citations,
        "timestamp": time.time()
    }
    
    history = state.get('reflection_history', [])
    history.append(reflection_entry)
    
    return {
        "reflection_history": history,
        "current_phase": "support_assessment"
    }

def assess_utility(state: GraphState) -> Dict:
    """
    Final utility assessment and next action decision
    
    This function provides the final evaluation of the generated answer,
    considering:
    - Completeness and comprehensiveness
    - Clarity and structure
    - User satisfaction potential
    - Areas for improvement
    - Recommendation for next steps
    
    Args:
        state: Current graph state with all components
    
    Returns:
        Dict with final assessment and complete reflection history
    """
    question = state['question']
    answer = state['answer']
    context = "\n\n".join([doc.page_content for doc in state['context']])
    
    # Perform final utility assessment
    assessment = utility_assessment_chain.invoke({
        "question": question,
        "answer": answer,
        "context": context
    })
    
    # Create final reflection entry
    reflection_entry = {
        "phase": "utility_assessment",
        "utility": assessment.utility,
        "reasoning": assessment.reasoning,
        "confidence": assessment.confidence,
        "next_action": assessment.next_action,
        "timestamp": time.time()
    }
    
    history = state.get('reflection_history', [])
    history.append(reflection_entry)
    
    # Compile comprehensive final assessment with metrics
    final_assessment = {
        "utility": assessment.utility,
        "next_action": assessment.next_action,
        "total_retrievals": state.get('retrieval_attempts', 0),
        "total_generations": state.get('generation_attempts', 0),
        "reflection_steps": len(history)
    }
    
    return {
        "reflection_history": history,
        "final_assessment": final_assessment,
        "current_phase": "final_assessment"
    }

# Improved Conditional Edge Functions
# These functions implement the decision-making logic for routing between nodes
# They analyze the current state and reflection history to determine the next step

def decide_initial_action(state: GraphState) -> str:
    """
    Decide initial action based on reflection
    
    This function examines the initial reflection decision and routes the workflow:
    - If retrieval is needed, go to the retrieve node
    - If sufficient knowledge exists, skip directly to generation
    
    Args:
        state: Current graph state with reflection history
    
    Returns:
        String indicating the next node to execute
    """
    latest_reflection = state['reflection_history'][-1]
    action = latest_reflection['decision']
    
    if action == ReflectionAction.RETRIEVE:
        return "retrieve"
    else:
        return "generate_direct"

def decide_after_relevance(state: GraphState) -> str:
    """
    Decide next step after relevance assessment
    
    This function analyzes the relevance assessment and determines whether:
    - Documents are relevant enough to proceed with generation
    - More retrieval is needed (if attempts are under the limit)
    - Generation should proceed despite low relevance
    
    Args:
        state: Current graph state with relevance assessment
    
    Returns:
        String indicating the next node to execute
    """
    latest_reflection = state['reflection_history'][-1]
    should_use = latest_reflection['should_use']
    retrieval_attempts = state.get('retrieval_attempts', 0)
    
    if should_use:
        return "generate"
    elif retrieval_attempts < 3:  # Max 3 retrieval attempts to prevent infinite loops
        return "retrieve_more"
    else:
        return "generate"  # Generate with what we have after max attempts

def decide_after_support(state: GraphState) -> str:
    """
    Decide next step after support assessment
    
    This function evaluates how well the context supports the answer and decides:
    - If fully supported, proceed to utility assessment
    - If partially supported and retrieval attempts remain, try more retrieval
    - If generation attempts remain, try regenerating the answer
    - Otherwise, proceed to final assessment
    
    Args:
        state: Current graph state with support assessment
    
    Returns:
        String indicating the next node to execute
    """
    latest_reflection = state['reflection_history'][-1]
    support_level = latest_reflection['support_level']
    retrieval_attempts = state.get('retrieval_attempts', 0)
    generation_attempts = state.get('generation_attempts', 0)
    
    if support_level == SupportLevel.FULLY_SUPPORTED:
        return "assess_utility"
    elif support_level == SupportLevel.PARTIALLY_SUPPORTED and retrieval_attempts < 3:
        return "retrieve_more"
    elif generation_attempts < 2:  # Try regenerating once with same context
        return "regenerate"
    else:
        return "assess_utility"  # Accept current answer after exhausting options

def decide_final_action(state: GraphState) -> str:
    """
    Decide final action based on utility assessment
    
    This function makes the final routing decision based on:
    - Utility assessment recommendation
    - Number of previous attempts (to prevent infinite loops)
    - Overall system constraints and limits
    
    Args:
        state: Current graph state with final assessment
    
    Returns:
        String indicating the next node or END to finish workflow
    """
    assessment = state['final_assessment']
    next_action = assessment['next_action']
    
    # Respect attempt limits to prevent infinite processing
    if next_action == ReflectionAction.RETRIEVE and assessment['total_retrievals'] < 3:
        return "retrieve_more"
    elif next_action == ReflectionAction.REGENERATE and assessment['total_generations'] < 2:
        return "regenerate"
    else:
        return END  # End the workflow and return final result

# Build Enhanced Workflow Graph
def build_enhanced_workflow():
    """
    Build the enhanced Self-RAG workflow graph
    
    This function constructs the complete workflow graph that defines:
    1. All processing nodes (reflection, retrieval, generation, assessment)
    2. Conditional edges that route between nodes based on decisions
    3. The overall flow from initial question to final answer
    
    The workflow implements a sophisticated decision tree that allows for:
    - Multiple retrieval attempts with different strategies
    - Regeneration of answers when needed
    - Comprehensive assessment at each stage
    - Adaptive routing based on reflection outcomes
    
    Returns:
        Compiled LangGraph workflow ready for execution
    """
    workflow = StateGraph(GraphState)
    
    # Add all processing nodes to the graph
    # Each node represents a specific stage in the self-reflective RAG process
    workflow.add_node("initial_reflection", initial_reflection)      # Decide if retrieval needed
    workflow.add_node("retrieve", adaptive_retrieve)                 # Get documents from vectorstore
    workflow.add_node("assess_relevance", assess_relevance)          # Evaluate document relevance
    workflow.add_node("generate", generate_with_reflection)          # Create answer from context
    workflow.add_node("generate_direct", generate_with_reflection)   # Generate without retrieval
    workflow.add_node("assess_support", assess_support)              # Check context support
    workflow.add_node("assess_utility", assess_utility)              # Final answer evaluation
    workflow.add_node("retrieve_more", adaptive_retrieve)            # Additional retrieval rounds
    workflow.add_node("regenerate", generate_with_reflection)        # Regenerate with same context
    
    # Define the workflow routing logic with conditional edges
    # START -> Initial reflection to decide first action
    workflow.add_edge(START, "initial_reflection")
    
    # After initial reflection: either retrieve or generate directly
    workflow.add_conditional_edges(
        "initial_reflection",
        decide_initial_action,
        {
            "retrieve": "retrieve",              # Need more information
            "generate_direct": "generate_direct"  # Sufficient knowledge exists
        }
    )
    
    # After retrieval: assess relevance of retrieved documents
    workflow.add_edge("retrieve", "assess_relevance")
    
    # After relevance assessment: generate or retrieve more
    workflow.add_conditional_edges(
        "assess_relevance",
        decide_after_relevance,
        {
            "generate": "generate",           # Documents are relevant
            "retrieve_more": "retrieve_more"  # Need better documents
        }
    )
    
    # Additional retrieval goes back to relevance assessment
    workflow.add_edge("retrieve_more", "assess_relevance")
    
    # After generation with context: assess how well context supports answer
    workflow.add_edge("generate", "assess_support")
    
    # Direct generation (no retrieval) goes straight to final assessment
    workflow.add_edge("generate_direct", "assess_utility")
    
    # After support assessment: decide next action based on support quality
    workflow.add_conditional_edges(
        "assess_support",
        decide_after_support,
        {
            "assess_utility": "assess_utility",   # Well supported, proceed to final assessment
            "retrieve_more": "retrieve_more",     # Need better supporting context
            "regenerate": "regenerate"            # Try generating different answer
        }
    )
    
    # Regenerated answers go back to support assessment
    workflow.add_edge("regenerate", "assess_support")
    
    # Final utility assessment decides whether to end or continue
    workflow.add_conditional_edges(
        "assess_utility",
        decide_final_action,
        {
            END: END,                          # Satisfied with answer quality
            "retrieve_more": "retrieve_more",  # Final attempt at better context
            "regenerate": "regenerate"         # Final attempt at better answer
        }
    )
    
    # Compile the workflow into an executable graph
    return workflow.compile()

# Test the Enhanced System
if __name__ == "__main__":
    """
    Main execution block for testing the Enhanced Self-RAG System
    
    This section demonstrates how to use the system and provides:
    1. System initialization and setup
    2. Example question processing
    3. Comprehensive result analysis and reporting
    4. Reflection history visualization
    """
    print("Starting Enhanced Self-RAG System...")
    print("=" * 80)
    
    # Build and compile the workflow graph
    app = build_enhanced_workflow()
    
    # Define a complex test question that requires multi-step reasoning
    # This question is designed to test the system's ability to:
    # - Retrieve relevant information about RAG approaches
    # - Compare and contrast different methodologies
    # - Provide comprehensive analysis with examples
    test_question = "What are the key differences between traditional RAG and agentic RAG approaches, and how do they handle complex multi-step reasoning?"
    
    print(f"Question: {test_question}\n")
    print("Running enhanced workflow with self-reflection...")
    print("-" * 80)
    
    try:
        # Execute the workflow with initial empty state
        # The system will populate all fields through its processing
        result = app.invoke({
            "question": test_question,
            "context": [],                    # Start with no context
            "answer": "",                     # No answer initially
            "reflection_history": [],         # Empty reflection history
            "retrieval_attempts": 0,          # No retrieval attempts yet
            "generation_attempts": 0,         # No generation attempts yet
            "current_phase": "start",         # Starting phase
            "final_assessment": {}            # Empty final assessment
        })
        
        # Display the final generated answer
        print(f"\nFINAL ANSWER:")
        print("=" * 80)
        print(result['answer'])
        
        # Show comprehensive system performance metrics
        print(f"\nSYSTEM REFLECTION SUMMARY:")
        print("=" * 80)
        assessment = result['final_assessment']
        print(f"Total Retrieval Steps: {assessment['total_retrievals']}")
        print(f"Total Generation Steps: {assessment['total_generations']}")
        print(f"Total Reflection Steps: {assessment['reflection_steps']}")
        print(f"Final Utility Assessment: {assessment['utility']}")
        
        # Provide detailed reflection history for transparency
        print(f"\nREFLECTION HISTORY:")
        print("-" * 80)
        for i, reflection in enumerate(result['reflection_history'], 1):
            print(f"{i}. {reflection['phase'].upper()}")
            print(f"   Reasoning: {reflection['reasoning'][:100]}...")
            print(f"   Confidence: {reflection['confidence']:.2f}")
            print()
            
    except Exception as e:
        # Handle any errors during execution with detailed error reporting
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()