# Import necessary components
import operator
import os
from langgraph.graph import StateGraph, START, END
#from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain import hub
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
load_dotenv()
##from display_graph import display_graph
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
# Define diagnostic and action tools
tools = [TavilySearch(max_results=3)]  # Search tool used for subtask execution

# Set up the model and agent executor
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant."""),
        ("placeholder", "{messages}")
    ]
)
prompt.pretty_print()

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",
)

agent_executor = create_react_agent(llm, tools, prompt=prompt)

# Define the Plan and Execution structure
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    steps: List[str] = Field(description="Numbered unique steps to follow, in order")

class Response(BaseModel):
    response: str = Field(description="Response to user.")

class Act(BaseModel):
    action: Union[Response, Plan] = Field(description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan.")

# Planning step
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step-by-step plan. \
               This plan should involve individual numbered tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
               The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",
)
# Re-planning step
replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step numbered plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt |AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",
)

# Execution step function
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"For the following plan:\n{plan_str}\n\nYou are tasked with executing step 1, {task}."
    agent_response = await agent_executor.ainvoke({"messages": [("user", task_formatted)]})
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

# Planning step function
async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

# Re-planning step function
async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)

    # If the re-planner decides to return a response, we use it as the final answer
    if isinstance(output.action, Response):  # Final response provided
        return {"response": output.action.response}  # Return the response to the user
    else:
        # Otherwise, we continue with the new plan (if re-planning suggests more steps)
        return {"plan": output.action.steps}

# Conditional check for ending
def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

# Build the workflow
workflow = StateGraph(PlanExecute)

# Add nodes to the workflow
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

# Add edges to transition between nodes
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])

# Compile the workflow into an executable application
app = workflow.compile()

# Visualization of the workflow
# Display the graph
#display_graph(app, file_name=os.path.basename(__file__))

# Example of running the agent
config = {"recursion_limit": 50}

import asyncio

# Function to run the Plan-and-Execute agent
async def run_plan_and_execute():
    # Input from the user
    inputs = {"input": "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace weighs. What are their combined weights in pounds?"}

    # Configuration for recursion limit
    config = {"recursion_limit": 50}

    # Run the Plan-and-Execute agent asynchronously
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_plan_and_execute())