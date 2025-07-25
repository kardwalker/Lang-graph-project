#AG_State_machine_workflow.py
print("the following cod eisa the implementation of RAG workflow in Langgraph with a state machine approach")
print("This is a state machine workflow for RAG (Retrieval-Augmented Generation) using LangGraph.")
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List ,TypedDict
from langchain_community.document_loaders import WebBaseLoader
import os
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()  # Load environment variables

# intialize the model and embeddings model
#rom langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    deployment_name="gpt-4o-mini",  # Use deployment_name for Azure
    api_version="2024-12-01-preview",
    temperature=0.7,
    max_tokens=512
)

embedding = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key"),
    deployment="text-embedding-ada-002",  # Use deployment for embeddings
    api_version="2023-05-15"
)
"""
this is wrong way to initialize the model and embeddings model
You should use deployment_name and deployment for Azure models and embeddings respectively.
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    temperature=0.7,
    max_tokens=512,
    model_kwargs={"deployment":"gpt-4o-mini"}  # ✅ Correct way now
)

embedding = AzureOpenAIEmbeddings(
    azure_endpoint="https://agentic-keran-framework.openai.azure.com/",
    api_key=os.getenv("Embedding_api_key"),
    api_version="2023-05-15",
    model_kwargs={"deployment": "text-embedding-ada-002"}  # ✅ Correct way
)
"""
url = [
    "https://github.com/redis/redis",
    "https://github.com/redis/redis-py"
    ,
]

docs = [WebBaseLoader(url).load() for url in url]
docs_list  = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10,
)

docs_splits = text_splitter.split_documents(docs_list)  
vector_Store = Chroma.from_documents(
    documents=docs_splits,
    embedding=embedding,  # Use the correct variable name
    collection_name="RAG_with_multiple_docs",

)

prompt = ChatPromptTemplate.from_template(
    """
you are a helpful assistant for question-answering tasks, use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Qestion : {question}
Context :   
{context}"""

)

rag_chain = (
 # Pass the quest
 prompt
 | model | StrOutputParser()
)

# define the graph

class Rag_state(TypedDict):
    """
Represent the state of the RAG workflow.
    Args:
        question (str): The question to be answered.
        web_search (List[str]): The context documents used for answering the question.
        answer (str): The answer to the question.
        documents (List[str]): The documents used in the RAG process.       
    """
    question: str
    web_search: List[str]
    answer: str
    documents: List[str]


# retrieve the documents from the vector store
def retrieve_documents(state: Rag_state) -> Rag_state:
    """
    Retrieve documents from the vector store based on the question.
    Args:
        state (Rag_state): The current state of the RAG workflow.
    Returns:
        Rag_state: Updated state with retrieved documents.
    """
    question = state['question']
  #  documents = retriver.invoke(question)
    retriever = vector_Store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    state['documents'] = retriever.invoke(state['question'])
    return state

def generate(state : Rag_state) -> Rag_state:
    """
    Generate an answer based on the retrieved documents and the question.
    Args:
        state (Rag_state): The current state of the RAG workflow.
    Returns:
        Rag_state: Updated state with the generated answer.
    """

    context = "\n\n".join(doc.page_content for doc in state['documents'])
    state['web_search'] = context
    state['answer'] = rag_chain.invoke({"question": state['question'], "context": context})
    return state

def create_workflow():
    workflow = StateGraph(Rag_state)
    # add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate_answer", generate)  
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile(checkpointer = MemorySaver())


async def run_main():
    graph = create_workflow()
    config = {
        "configurable": {"thread_id" : "1"},
              "recursion_limit" : 20 
              }
    inputaa = {"question" : input("write your query")}
    try:
        async for event in graph.astream(inputaa, config= config,stream_mode = "updates"):
            if "error" in event:
                print(f"Error: {event['error']}")
                break
            print(event)
    except Exception as e:
        print(f"Workflow exceution failed : {str(e)}")

if __name__ == "__main__":
        asyncio.run(run_main())                
    






