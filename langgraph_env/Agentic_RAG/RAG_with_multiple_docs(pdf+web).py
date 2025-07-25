from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv  # Load environment variables
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
import os
import tiktoken

load_dotenv()  # Load environment variables

# initialize the OpenAI model and embeddings _model

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key")                    ,
    deployment="text-embedding-ada-002",  # Deployment name must match what you
    # created in Azure
    model="text-embedding-ada-002",       # Model name (optional, but helps for clarity)
    api_version="2023-05-15",        # Use correct version (based on Azure docs or trial-and-error))
)

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

file_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\LangGraph_Project\\2311.12983v1.pdf"

### loading the PDF file
# file_path = " C:\Users\Aman\Desktop\Virtual__house\LangGraph_Project\2311.12983v1.pdf"

pdf_loader = PyPDFLoader(file_path)
pdf = pdf_loader.load()

# split the document into smaller chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,  # Adjust chunk size as needed
    chunk_overlap=20,  # Adjust overlap as needed
    
)
splitdocs = text_splitter.split_documents(pdf)

### step 2 Load and process the web url
urls = ["https://github.com/facebookresearch/faiss","https://github.com/facebookresearch/faiss/wiki"]
# Load web documents
web_loader = WebBaseLoader(urls)
web_docs = web_loader.load()

# SPLIT THE WEB DOCUMENTS
web_split = text_splitter.split_documents(web_docs)

all_docs = web_split + splitdocs

### sstrep3 

# index the documents iinot a vector store
vector_store = Chroma.from_documents(
    all_docs,
    embedding_model,
    collection_name="long_doc_collection",
 #  persist_directory="chroma_db"
)

## Define the Retrivers

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Adjust k as needed for the number of results)
)

## prompt template for the query    

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks.
      Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
      Use three sentences maximum and keep the answer concise."""),
    ("user", "Question: {question}\nContext: {context}\nAnswer:")
])


## step 4 set up RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser() )
## step 5 Ask a quesiton and generate the answer
question = "What is the purpose of the faiss library?"  
#nswer = rag_chain.invoke({"question": question})

for i in rag_chain.stream(question):
    print(i ,end="", flush=True)  # Print each chunk of the answer as it streams
    

    ###Docstring of the code
    """
    This code implement a Retrieval-Augmented Generation (RAG) pipeline using LangChain and Azure OpenAI.

    Local Document and Web Content are are loaded and split into smaller chunks

    Combining the local document and web content, the code creates a vector store using Chroma.

    Embedding and Indexing are performed using Azure OpenAI embeddings.

    
    Retrieval is done using a retriever that fetches relevant documents based on the query.

    Prompt Construction is done using chat prompt templates to format the question and context for the LLM.

    RAG chain is executed with streming

    
After retrieving relevant context from your documents and formatting it,
 the RAG chain passes a user question and the constructed prompt to model (your Azure-hosted GPT model).
The model receives both the prompt (which includes instructions and context) and the user's question,
 and produces a natural language answer.
    """