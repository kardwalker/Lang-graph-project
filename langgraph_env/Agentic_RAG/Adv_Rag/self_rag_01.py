from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import ChatPromptTemplate
from langchain.chains import StrOutputParser
import os
from typing import  Dict,List, TypedDict


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
    colection_name = "self_rag_01",  # Name of the collection in Chroma
 #   persist_directory="chroma_db"  # Directory to store the vector database
)

retriever = vectorstore.as_retriever(
    search_type="similarity",  # Use similarity search
    search_kwargs={"k": 2 }# Retrieve top 5 similar documents
)

# 
prompt = ChatPromptTemplate.from_messages(
    """
    You are an expert in RAG (Retrieval-Augmented Generation) techniques.
    Your task is to answer questions based on the provided context. 
    If the context does not contain relevant information, respond with "I don't know
    Question: {question}
    Context: {context}  
    Answer:
    """,)

rag_chain  = (
    prompt
    | model
    | retriever
    | StrOutputParser()
)

class Graph_state(TypedDict):
    question: str
    context: List[List[str]] | List[str]
    answer: str
import pydantic
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


