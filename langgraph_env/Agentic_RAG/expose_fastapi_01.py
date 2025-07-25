# Expose It as an API with FastAPI


#pip install fastapi uvicorn

from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# Define request schema
class Query(BaseModel):
    question: str

# Load vector store
embedding = AzureOpenAIEmbeddings(
    azure_endpoint="https://<your-resource>.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key"),
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    api_version="2023-05-15",
)

vector_store = Chroma(
    collection_name="agentic_rag",
    embedding_function=embedding,
    persist_directory="./chroma_store"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Load chat model
llm = AzureChatOpenAI(
    azure_endpoint="https://<your-resource>.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key"),
    deployment_name="gpt-35-turbo",
    model_name="gpt-35-turbo",
    api_version="2023-05-15"
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# FastAPI app
app = FastAPI()

@app.post("/ask")
def ask(query: Query):
    response = rag_chain.invoke(query.question)
    return {
        "answer": response["result"],
        "sources": [doc.metadata for doc in response["source_documents"]]
    }


# Run the api in bash
# uvicorn main:app --reload
