
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, START, END
import pydantic
from pydantic import Field


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

urls = [
    "https://www.oreilly.com/radar/agentic-rag/",
    "https://github.com/infiniflow/ragflow",
    "https://github.com/NirDiamant/RAG_Techniques"


]
docs = [WebBaseLoader(url).load() for url in urls ]
docs_list = [doc for sublist in docs for doc in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs_split = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embedding,
    collection_name="self_rag_01"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",  # Use similarity search
    search_kwargs={"k": 2}  # Retrieve top 2 similar documents
)

# 
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert in RAG (Retrieval-Augmented Generation) techniques.
    Your task is to answer questions based on the provided context. 
    If the context does not contain relevant information, respond with "I don't know".
    Question: {question}
    Context: {context}  
    Answer:
    """
)

rag_chain = (
    prompt
    | model
    | StrOutputParser()
)

class Graph_state(TypedDict):
    question: str
    context: List[Document]  # Use Document type to handle retriever output
    answer: str

class Relevance_score(pydantic.BaseModel):
    score: bool = Field(description="Indicates if the context is relevant to the question in 'yes or no' format.")

retrival_prompt = ChatPromptTemplate.from_template(
"""    You are an expert in RAG (Retrieval-Augmented Generation) techniques.
    Your task is to determine if the provided context is relevant to the question.
    Document: {context}
    Question: {question}
    is the context relevant to the question? Answer in 'yes or no' format."""
)

retricval_checker = retrival_prompt | AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),  # API key from environment variables
    azure_endpoint=os.getenv("Azure_endpoint"),  # Azure OpenAI endpoint URL    
    deployment_name="gpt-4o-mini",  # Specific deployment name in Azure
    api_version="2024-12-01-preview",  # API version for compatibility
    temperature=0.7,  # Controls randomness (0=deterministic, 1=creative)
    max_tokens=512  # Maximum tokens in the response
).with_structured_output(
    Relevance_score 
    )


# hallunication_checker

class Hallunication_checker(pydantic.BaseModel):
    is_hallucinationswer : bool = Field(description = "indicates if the answer is a hallucination or not in 'yes or no' format.")


hallunication_prompt = ChatPromptTemplate.from_template(
    """You are an expert in RAG (Retrieval-Augmented Generation) techniques
    Your task is to determine if the provided answer is a hallucination or not.
    content: {context}
    answer: {answer}
    """
)

hallucination_scorer = hallunication_prompt | AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    deployment_name="gpt-4o-mini",
    api_version="2024-12-01-preview",
    temperature=0.7,
    max_tokens=512
).with_structured_output(
    Hallunication_checker
)

class answer_verifier(pydantic.BaseModel):
    is_answer_correct: bool = Field(description="Indicates if the answer is correct or not in 'yes or no' format.")

answer_verification_prompt = ChatPromptTemplate.from_template(
    """you are an expert in RAG (Retrieval-Augmented generation)techniques
    answer : {answer}
    context: {context}
    Does the answer correctly address the question based on the context? Answer in 'yes or no' format.""")

answer_verification_scorer = answer_verification_prompt | AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    deployment_name="gpt-4o-mini",
    api_version="2024-12-01-preview",
    temperature=0.0,        
    max_tokens=512
).with_structured_output(
    answer_verifier
)


# Defining the final agent workflow

def state(graph: Graph_state) -> Dict[str, str]:
    """
    Main function to process the state graph and return the final answer.
    """
    question = graph['question']
    
    # Step 1: Retrieve relevant context
    context = retriever.invoke(question)
    
    return {"context": context}


def generate_answer(graph: Graph_state):
    question = graph['question']
    context = graph['context']
    
    # Convert context to string format for the prompt
    context_str = "\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in context])
    
    answer = rag_chain.invoke({"question": question, "context": context_str})
    return {"answer": answer}


def check_relevance(graph: Graph_state) -> Dict[str, str]:
    question = graph['question']
    context = graph['context']
    relevance = []
    for ctx in context:
        ctx_content = ctx.page_content if hasattr(ctx, 'page_content') else str(ctx)
        relevance_score = retricval_checker.invoke({"context": ctx_content, "question": question})
        if relevance_score.score:  # Access attribute directly
            relevance.append(ctx)
    return {"context": relevance}


def multi_query(graph: Graph_state) -> str:
    """
    decide whether to use multi-query or proceed with the current context"""
    if not graph["context"]:
        return "query_transformation"
    return "check_relevance"  # First check relevance, then generate answer

def relevance_check(graph: Graph_state) -> str:
    """
    Check if we have relevant context after relevance filtering
    """
    if not graph["context"]:
        return "query_transformation"
    return "generate_answer"

def hallucination_check(graph: Graph_state) -> str:
    """
    Check if the answer contains hallucinations.
    """
    answer = graph['answer']
    context = graph['context']
    
    # Convert context to string
    context_str = "\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in context])
    
    hallucination_score = hallucination_scorer.invoke({"context": context_str, "answer": answer})
    if hallucination_score.is_hallucinationswer:  # Access attribute directly
        return "not_supported"
    
    answer_check = answer_verification_scorer.invoke({"context": context_str, "answer": answer})
    if answer_check.is_answer_correct:  # Access attribute directly
        return "relevant"
    else:
        return "irrelevant"    

def query_transformation(graph: Graph_state) -> Dict[str, str]:
    """
    Transform the query if no relevant context is found.
    """
    question = graph['question']
    q_trans_pmt = ChatPromptTemplate.from_template(  # Fixed typo
        """You are an expert in RAG (Retrieval-Augmented Generation) techniques.
        Your task is to transform the question to make it more specific or detailed.
        Question: {question}
        Transform the question to make it more specific or detailed."""
    )
    
    question_transformer = q_trans_pmt | AzureChatOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("Azure_endpoint"),
        deployment_name="gpt-4o-mini",
        api_version="2024-12-01-preview",
        temperature=0.0,
        max_tokens=512
    ) | StrOutputParser()
    
    transformed_question = question_transformer.invoke({"question": question})
    
    # Retrieve new context with transformed question
    new_context = retriever.invoke(transformed_question)
    
    return {"question": transformed_question, "context": new_context}


def transform_query(graph: Graph_state) -> Dict[str, str]:
    """Wrapper function for query_transformation"""
    return query_transformation(graph)


## Workflow graph - FIXED to prevent recursion

builder = StateGraph(Graph_state)
builder.add_node("retrieve", state)
builder.add_node("generate_answer", generate_answer)
builder.add_node("check_relevance", check_relevance)
builder.add_node("query_transformation", transform_query)

# Fixed flow to prevent infinite recursion
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "check_relevance")

builder.add_conditional_edges(
    "check_relevance", 
    relevance_check,
    {
        "query_transformation": "query_transformation",
        "generate_answer": "generate_answer"    
    }
)

# FIXED: Remove the circular path back to query_transformation
builder.add_conditional_edges(
    "generate_answer", 
    hallucination_check,
    {
        "not_supported": END,  # End instead of going back to transformation
        "irrelevant": END,     # End instead of going back to transformation
        "relevant": END
    }
)

# FIX: Add the missing edge - query_transformation must go somewhere!
builder.add_edge("query_transformation", "generate_answer")


# Compile the graph
graph = builder.compile()

# Test inputs
inputs = {
    "question": "tell me about chunking strategies",  # Fixed grammar
}

rs = graph.invoke(inputs)
print(rs)

# Visualize the graph
from langchain_core.runnables.graph import MermaidDrawMethod
import random

print("\nGenerating graph visualization...")
try:
    mermaid_png = graph.get_graph(xray=1).draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    output_folder = "."
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"self_rag_graph_{random.randint(1,10000)}.png")
    with open(filename, "wb") as f:
        f.write(mermaid_png)
    print(f"Graph visualization saved as: {filename}")
except Exception as e:
    print(f"Error generating graph visualization: {e}")
    print("You can still see the graph structure by running: graph.get_graph().print_ascii()")

   