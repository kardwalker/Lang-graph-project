print("this code is implemetation of Plan and Execute agent using LangGraph.")

print("but it has some issues, so it is not working properly")
"""
KeyError: 'agent'
During task with name 'replan' and id '573e0068-bccf-4fba-6476-c844a1e129c8'
"""
import asyncio
from langchain_core.tools import tool
import operator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union, Optional
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
load_dotenv()
@tool
def identify_product(issue: str)->str:
    """Identifies the product based on the issue description."""
    if not issue or not isinstance(issue, str):
        return "Error: No issue provided"
    return f"Identified product related to {issue}."

@tool
def search_manual(product: str, issue: str) -> str:
    """Searches the product manual for troubleshooting steps."""
    if not product or not issue or not isinstance(product, str) or not isinstance(issue, str):
        return "Error: Invalid product or issue provided"
    return f"Searched manual for {product} issue: {issue}. Suggested steps: ..."

@tool
def escalate_to_support(product: str, issue: str) ->str:
    """Escalates the issue to a human support team."""
    if not product or not issue or not isinstance(product, str) or not isinstance(issue, str):
        return "Error: Invalid product or issue provided"
    return f"Escalated {product} issue: {issue} to support."

tools = [identify_product, search_manual, escalate_to_support]

# Improved system prompts
SYSTEM_PROMPT = """You are an customer support assistant. Your task is to help customers troubleshoot product issues 
using the available tools. Always identify the issue and product from the input and use it consistently across all steps.
Available tools:
1. identify_product - Identifies the product based on the input.
2. search_manual - Searches the product manual for troubleshooting steps
3. escalate_to_support - Escalates the issue to a human support team

Ensure each step is completed before moving to the next one."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}")
])

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini"  # Ensure this matches your deployment name
)
agent_executor = create_react_agent(llm, tools, prompt=prompt)

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    error: Optional[str]

class Plan(BaseModel):
    steps: List[str] = Field(description="Numbered unique steps to follow, in order")

class Response(BaseModel):
    response: str = Field(description="Response to user.")

class Act(BaseModel):
    action: Union[Response, Plan] = Field(description="Action to perform")

# Planning step
async def plan_step(state: PlanExecute) -> dict:
    try:
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{messages}")
        ])
        planner = planner_prompt | llm.with_structured_output(Plan)
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}
    except Exception as e:
        return {"error": f"Planning error: {str(e)}"}

async def execute_step(state: PlanExecute):
    try:
        if "error" in state:
            return {"response": f"Workflow failed: {state['error']}"}
        
        plan = state["plan"]
        if not plan:
            return {"response": "No plan steps available to execute"}
            
        task = plan[0]
        agent_response = await agent_executor.ainvoke({"messages": [("user", task + " " + state["input"])]})
        return {"past_steps": [(task, agent_response["messages"][-1].content)]}
    except Exception as e:
        return {"error": f"Execution error: {str(e)}"}
    
# Enhanced replanning with better error handling
async def replan_step(state: PlanExecute) -> dict:
    try:
        if "error" in state:
            return {"response": f"Workflow failed: {state['error']}"}
            
        replanner_prompt = ChatPromptTemplate.from_template("""
            Given the objective: {input}
            Original plan: {plan}
            Completed steps: {past_steps}
            
            Please either:
            1. Provide next steps if more work is needed
            2. Provide a final response if the workflow is complete
            
            Only include steps that still need to be done.
            """)
        
        replanner = replanner_prompt | llm.with_structured_output(Act)
        output = await replanner.ainvoke(state)
        
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        return {"plan": output.action.steps}
    except Exception as e:
        return {"error": f"Replanning error: {str(e)}"}

workflow = StateGraph(PlanExecute)

# Add nodes
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

# Add edges
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    lambda s: END if ("response" in s or "error" in s) else "agent",
    [END]
)
app = workflow.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "1"}, "recursion_limit": 50}

async def run_plan_and_execute():
    inputs = {"input": "Help troubleshoot my smartphone issue."}
    async for event in app.astream(inputs, config=config):
        print(event)

if __name__ == "__main__":
    asyncio.run(run_plan_and_execute())