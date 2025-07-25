from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
import requests
from typing import TypedDict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# ✅ Define the agent's memory state
class State(TypedDict):
    query: str
    result: Any
    messages: List[Any]

# ✅ YouTube Search Tool Function
def youtube_search(query: str):
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return {"error": "Missing YOUTUBE_API_KEY in environment variables."}

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
        if item["id"]["kind"] == "youtube#video"
    ]
    return {"videos": videos}

# ✅ Wrap YouTube Search as a LangChain Tool
youtube_tool = Tool.from_function(
    func=youtube_search,
    name="youtube_search",
    description="Search YouTube videos based on a query."
)

# ✅ Azure OpenAI model
model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini"
)

# ✅ Create a Function-Calling Agent
agent = create_openai_functions_agent(
    llm=model,
    tools=[youtube_tool],
    system_message="You are a helpful assistant with access to the youtube_search tool."
)

# ✅ AgentExecutor to run tool-invoking reasoning chains
executor = AgentExecutor(agent=agent, tools=[youtube_tool], verbose=True)

# ✅ LangGraph: Define Node logic
def call_agent(state: State):
    # Agent input is the query
    result = executor.run(state["query"])
    return {
        "messages": state.get("messages", []) + [result],
        "result": result
    }

# ✅ Build LangGraph
builder = StateGraph(State)

# Add the agent node
builder.add_node("call_agent", call_agent)

# Edges
builder.add_edge(START, "call_agent")
builder.add_edge("call_agent", END)

# ✅ Compile graph
graph = builder.compile(checkpointer=MemorySaver())

# ✅ Run graph with a test query
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    mess = {"query": "Find latest LangGraph tutorials on YouTube"}

    for event in graph.stream(mess, config=config, stream_mode="updates"):
        print(event)
