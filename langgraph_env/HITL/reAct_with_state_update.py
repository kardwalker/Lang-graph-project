

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from googleapiclient.discovery import build
import os
import requests
from typing import TypedDict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Define the agent's memory state
class State(TypedDict):
    query: str
    result: str
    messages: List[Any]

# ✅ Define YouTube Search Tool
@tool(description="Search youtube video based on query")
def youtube_search(query: str):
    api_key = os.getenv("YOUTUBE_API_KEY")
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "key": api_key,
        "maxResults": 3
    }
    response = requests.get(url, params=params)
    data = response.json()

    videos = [
        {
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        }
        for item in data.get("items", [])
    ]
    return {"videos": videos}

# ✅ Create ToolNode for LangGraph
tools = ToolNode([youtube_search])

# ✅ Define Azure OpenAI model
model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini"
)
"""
Problem
In your graph setup:

The call_model node calls the model and gets a response.

The youtube_search tool exists as a node, but it isn’t actually being requested by the model in its output — because:

The system prompt isn't instructing the model that it can call tools.

You’re not using a Tool Calling capable model invocation (functions / tool_calls in OpenAI API terms or LangChain tool-augmented model).

There’s no logic parsing the model’s output for an intent to invoke youtube_search.

Right now, the model just responds with text like "I don't have real-time access", instead of calling your tool.

"""
# System prompt message
system_prompt = SystemMessage(content="You are a helpful assistant. You have access to the following tools: youtube_search. If a query requires a YouTube video search, you should call the youtube_search tool.")

model = model.bind_tools([youtube_search])
# ✅ Define the Agent's Node Logic
def call_model(state: State):
    messages = [
        system_prompt,
        HumanMessage(content=state["query"])
    ]
    response = model.invoke(messages)
    return {"messages": state.get("messages", []) + [response], "result": response.content}

# ✅ Build LangGraph
builder = StateGraph(State)

# Add the model node
builder.add_node("call_model", call_model)

# Add the tool node
builder.add_node("youtube_search", tools)

# Graph Edges
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "youtube_search")
builder.add_edge("youtube_search", END)

# Compile graph with MemorySaver checkpoint
graph = builder.compile(checkpointer=MemorySaver())

# ✅ Run graph with a test query
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    mess = {"query": "Latest LangGraph tutorials on YouTube"}

    for event in graph.stream(mess, config=config, stream_mode="updates"):
        print(event)
""" output 

{'call_model': {'messages': [AIMessage(content='I don\'t have real-time access to the internet to check
 the latest YouTube tutorials directly. However, to find the latest LangGraph tutorials,
 you can follow these steps:\n\n1. **Search on YouTube**: Go to YouTube and type "LangGraph tutorials"
 in the search bar. Sort the results by upload date to see the most recent videos.\n\n2. **Check Official
 Channels**: Look for official channels related to LangGraph or those of recognized creators in the field
 of programming or data science. They often post the latest tutorials.\n\n3. **Explore Programming Communities**:
=Websites like Reddit, Stack Overflow, or Discord communities may also share links to the latest tutorials and resources.\n\n4.
 **Follow Influencers**: Follow influencers or educators in the AI and data science space on social media platforms to get updates
 on their latest tutorials.\n\nIf you need more specific recommendations or help on a topic related to LangGraph, feel free to ask!',
 additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint':
 'fp_efad92c60b'}, id='run--388e6134-ae97-4fbf-9c7b-3be4310490f2-0')], 'result': 'I don\'t have real-time access to the internet
 to check the latest YouTube tutorials directly. However, to find the latest LangGraph tutorials, you can follow these steps:\n\n1.
 **Search on YouTube**: Go to YouTube and type "LangGraph tutorials" in the search bar. Sort the results by upload date to see the
 most recent videos.\n\n2. **Check Official Channels**: Look for official channels related to LangGraph or those of recognized creators
 in the field of programming or data science. They often post the latest tutorials.\n\n3. **Explore Programming Communities**: Websites
 like Reddit, Stack Overflow, or Discord communities may also share links to the latest tutorials and resources.\n\n4. **Follow Influencers**: Follow influencers
 or educators in the AI and data science space on social media platforms to get updates on their latest tutorials.\n\nIf you need more specific recommendations or
 help on a topic related to LangGraph, feel free to ask!'}}
{'youtube_search': {'messages': []}}"""

"""
Why model.bind_tools() alone isn’t working here
👉 model.bind_tools() tells the model what tools are available, but you still need to:

Use a tool-calling capable message format (like ChatMessage or via OpenAI function-calling schema).

Check the model's output for a tool_calls or function_call in its response.

Then route accordingly

"""