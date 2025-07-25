from langchain_openai import AzureOpenAIEmbeddings , AzureChatOpenAI
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma         
from 
load_dotenv()  # Load environment variables
import os

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key")                    ,
    deployment="text-embedding-ada-002",  # Deployment name must match what you
    # created in Azure
    model="text-embedding-ada-002",       # Model name (optional, but helps for clarity)
    api_version="2023-05-15",        # Use correct version (based on Azure docs or trial-and-error))
)

doc = [
    Document(
        page_content = """
Steps to Generate a Response with an LLM
Retrieve Relevant Document: Perform a similarity search to find the most relevant document for the user's question. 
Combine Context and Query: Format the retrieved document and user question into a prompt for the LLM.
Generate a Response: Use the LLM to generate a response based on the combined prompt.""",
metadata={"source": "doc1"}
    ),
    Document(
        page_content = "FAISS is a vector store that allows for efficient simiarity search and retrieval of high-dimensional vectors.",
        metadata={"source": "doc2"}
        ),
]


# index the documents in chrom vector store

db = Chroma.from_documents(
    documents=doc,  
    embedding=embedding_model,
    collection_name="agentic_rag",)
print("Vector store created with", len(db.get()), "documents.")
#define a retriver to fetch the documents

retriever = db.as_retriever(search_kwargs={"k": 2}, search_type="similarity")    


# Define a prompt tempate for the LLM
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("user", "{context}\n\nQuestion: {question}"),
    ]
)

# Intialize the chat model 
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

# set up the Rag chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),      
    }
    | prompt_template
    | model
    | StrOutputParser() )
# Use the rag pipline to answer the query

question = "What are the steps to generate a response with an LLM?"
response = rag_chain.invoke(question)

for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
