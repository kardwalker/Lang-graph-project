from functools import partial
import operator
from typing import Annotated, Sequence, TypedDict , Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel, Field
import os
from langgraph.graph import StateGraph , START , END
from langgraph.prebuilt import create_react_agent


# Define RouteResponse for customer service superviosr

class RouteResponse(BaseModel):
    next : Literal["Query_Agent" ,"Resolution_Agent", "Query_transformers_Agent", "Escalation_Agent", "FINSH"]

members_cs = ["Query_Agent", "Resolution_Agent", "Escalation_Agent", "Query_transformers_Agent"]

system_promt_cs = f"You are a customer service supervisor managing agents: {', '.join(members_cs)}. " 

promt_cs = ChatPromptTemplate.from_messages([
    ("system", system_promt_cs),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Choose the next agent to act based from {optoins}")
]).partial(
    options=str(members_cs))


model_cs = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"), 
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.2,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",  # Ensure this matches your deployment name
)

def supervisor_node_cs(state):
    supervisor_chain_cs = promt_cs | model_cs.with_structured_output(RouteResponse)
    return supervisor_chain_cs.invoke(state)

def agent_node(state, agent , name):
    result = agent.invoke(state)
    
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)],
    }


query_agent = create_react_agent(
    agent_name="Query_Agent",
    llm=model_cs,
    tools = [ TavilySearchResults(
        api_key=os.getenv("tavily_api_key"))]


resolution_agent = create_react_agent(
    agent_name="Resolution_Agent",
    llm=model_cs,
    tools=[PythonREPLTool()]
)

escalation_agent = create_react_agent(
    agent_name="Escalation_Agent",
    llm=model_cs,
)

query_node = partial(agent_node, agent=query_agent, name="Query_Agent")
resolution_node = partial(agent_node, agent=resolution_agent, name="Resolution_Agent")
escalation_node = partial(agent_node, agent=escalation_agent, name="Escalation_Agent")

class AgentState(TypedDict)::
    messages : Annotated[Sequence[BaseMessage], Field(description="List of messages in the conversation")]
    next : str


workflow_cs = StateGraph(AgentState)
workflow_cs.add_node("supervisor", supervisor_node_cs)
workflow_cs.add_node("query", query_node)   
workflow_cs.add_node("resolution", resolution_node)
workflow_cs.add_node("escalation", escalation_node)
for member in members_cs:
    workflow_cs.add_edge(member, "supervisor")

conditonal_map = {k: k for k in members_cs}
conditional_map["FINSH"] = END

workflow_cs.add_edge("supervisor", lambda state: state["next"], conditional_map=conditional_map)
workflow_cs.add_edge("query", "supervisor")
workflow_cs.add_edge("resolution", "supervisor")
workflow_cs.add_edge("escalation", "supervisor")
workflow_cs.add_edge(START, "supervisor")
graph_cs = workflow_cs.compile()

inputs_cs = {
    "supervisor": {"messages": []},
}