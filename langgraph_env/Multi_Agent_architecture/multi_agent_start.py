from typing import Annotated, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
import os

tavily_search_tool = TavilySearchResults(
    api_key = os.getenv("tavily_api_key"),
    max_results = 3,)

python_repl_tool = PythonREPLTool()

from langchain_core.messages import HumanMessage

def agent_node(state , agent, name):
    result = agent.invoke(state)
    
    return {
        "messages" : [HumanMessage(content=result["messages"][-1].content, name = name)],
    }

members = {"Researcher","Coder","Debugger", "Reviwer"}

sys_prmt = (
    "You are supervisor tasked with managing a conversation between following agents :{members}" 
    "Given the following user request, respond with the worker to act next." 
    "Each worker will perform a spedific task and respond with their results and status"
    "When finished, respond with FINISH "

)

options = ["FINSH"] + members

sypr =ChatPromptTemplate.from_messages(
    [
        ("system", sys_prmt),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(options = str(options), members = ", ".join(members))

model  = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",  # Ensure this matches your deployment name
)


def supervisor_node(state):
    supervisor_chain  = sypr | model.with_structured_output(routeRespose)


import functools 
import operator
from typing import Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent

class Agent_State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], Field(description="List of messages in the conversation")]
    next : str


research_agent = create_react_agent(
    model , tools=[tavily_search_tool]
)

research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# Note : This perfomms arbitary code execution, use with caution

coder_agent = create_react_agent(
    model, tools=[python_repl_tool]
)

# Note : This performs arbitary code execution, use with caution
coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")



## Define the workflow of the  graph

builder = StateGraph(Agent_State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("research", research_node)
builder.add_node("coder", coder_node)

## this is something new

for member in members:
    builder.add_edge(member, "supervisor")

conditional_map = {k : k  for k in members}
conditional_map["FINISH"] = END
builder.add_edge("supervisor",lambda state: state["next"], conditional_map=conditional_map)

builder.add_edge(START, "supervisor")

graph = builder.compile()

