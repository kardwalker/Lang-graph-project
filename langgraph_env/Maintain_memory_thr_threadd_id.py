from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import os
load_dotenv()
checkpointer = MemorySaver()

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

# Function to handle the user query and call the LLM
def call_model(state :MessagesState):
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
# without memory
graph = builder.compile()
messages = [HumanMessage(content="Hi")]

respon = graph.invoke({"messages": [sys_mes, *messages]})
print(respon)

# when we use memory, we need to specify a thread_iD


config = { "configurable" : {
    "thread_id": "1"}}

messages = [HumanMessage(content="tell me different of architecture of AI Agent")]
resp = graph.invoke({"messages": [sys_mes, *messages], "config": config})
for r in resp["messages"]:
    r.pretty_print()