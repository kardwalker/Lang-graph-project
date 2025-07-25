from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

import os
load_dotenv()


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

checkpointer = MemorySaver()

# 
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}     

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)      
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END) 

sys_mes = SystemMessage(
    content="You are a helpful assistant."
)       
# Compile the workflow
def interact_with_user_with_memory(thread_id: str):
    thread_id = "session_1"
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the session.")
            break
        
        messages = [HumanMessage(content=user_input)]
        
        # Use the checkpointer to maintain memory across sessions
        config = {"configurable": {"thread_id": thread_id}}
        
        response = builder.invoke({"messages": [sys_mes, *messages], "config": config})
        
        for r in response["messages"]:
            r.pretty_print()


