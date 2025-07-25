print("the following code is implementation of PLan and Execute agent using LangGraph.")

import operator
import os
from langchain_tavily import TavilySearch
from langgraph.graph import START , END , StateGraph
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel , Field
from typing import Annotated , List , Tuple , Union , TypedDict
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Set up environment variable
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
tools = [TavilySearch(max_results=5)]

promt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can search the web for information."),
        ("placeholder", "{messages}")
    ])

print("Prompt template:", promt)  # Changed from pretty_print() which doesn't exist
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

agent_executor = create_react_agent(model=model, tools=tools, prompt=promt)

class PlanAndExecuteInput(BaseModel):
     input_query : str = Field(..., description="The input query for the agent to process.")
     plan : List[str] = Field(default_factory=list, description="The plan created by the agent.") 
     past_steps : Annotated[List[tuple],operator.add] = Field(default_factory=list)
     response : Union[str, None] = Field(None, description="The response from the agent after executing the steps.") 

class Plan(BaseModel):
    steps: List[str] = Field(..., description="The steps to be executed by the agent." )

class Response(BaseModel):
    response: str = Field(..., description="The response from the agent after executing the steps.")

class action(BaseModel):
    action: List[Union[Response, Plan]]= Field(..., description="The action to be performed by the agent."
    "if you need to further use tools to get the answer, use plan")  

planner_prmt = ChatPromptTemplate.from_messages(
    [
        ("system","""For a given task , come up with a simple and effective plan to achiecve the goal
         this plan should involve individdual numbered steps, that if excuted in order  correctly will
         yield the desired result.
         The result of the final step should be the final answer, Make sure that each step has all the information needed - do not skips steps.""",
         ),
         ("placeholder", "{messages}"),
    ])

planner = planner_prmt | AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",  # Ensure this matches your deployment name
).with_structured_output(Plan)

replanner_prmt = ChatPromptTemplate.from_messages([
    ("human", """For the given task, come up with a simple step by step numbered plan.
    This plan should involve individual tasks that if executed in order correctly will yield the desired result.
    Do not add any superfluous information or step.
    The result of the final step should be the final answer, Make sure that each step has all the information needed - do not skip steps
    your objective was this:
    {input_query}
    your original plan was this:
    {plan}
    you have currently done these steps:
    {past_steps}
    update your plan accordingly to include the steps you have already done and the steps you need to do to achieve the goal.
    """)
])

replanner = replanner_prmt | AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",  # Ensure this matches your deployment name
).with_structured_output(action)  


# Execution step function
async def execute_plan(state : PlanAndExecuteInput) -> PlanAndExecuteInput:
    """
    Execute the plan step by step and return the final response.
    """
    steps = state.plan
    plan_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    task = steps[0]  # Fixed: changed from plan[0] to steps[0]
    task_format = f"for the following task:\n{plan_str}\n\n you are tasked with executing step 1: {task}\n."

    agent_response = await agent_executor.ainvoke(
        {"messages":[ ("user", task_format)]} ,
    )

    return {
        "past_steps" : [(task , agent_response["messages"][-1].content)],
    }

# Planning step function
async def plan_step(state : PlanAndExecuteInput):
    plan = await planner.ainvoke(
        {"messages": [("user", state.input_query)]}
    )

    return {
        "plan": plan.steps,}    

# Re-planning step function
async def replan_step(state: PlanAndExecuteInput):
    output = await replanner.ainvoke({
        "input_query": state.input_query,
        "plan": state.plan,
        "past_steps": state.past_steps
    })
    # if the replanner decides to return a plan , we use it as the final answer
    if isinstance(output.action[0], Response):  # Fixed: check first element of action list
        return {
            "response": output.action[0].response,
            "past_steps": state.past_steps + [(state.plan[0], output.action[0].response)],
        }

    else:
        return {
            "plan": output.action[0].steps,  # Fixed: get steps from first element
            "past_steps": state.past_steps + [(state.plan[0], "Re-planned")],
        }

def should_end(state: PlanAndExecuteInput) -> str:
    """
    Check if the plan has been fully executed.
    """
    if hasattr(state, "response") and state.response is not None:
        return "__end__"
    else:
        return "execute_plan"

workflow = StateGraph(PlanAndExecuteInput)
workflow.add_node("plan_step", plan_step)
workflow.add_node("execute_plan", execute_plan)
workflow.add_node("replan_step", replan_step)               

workflow.add_edge(START, "plan_step")
workflow.add_edge("plan_step", "execute_plan")      
workflow.add_edge("execute_plan", "replan_step")
workflow.add_conditional_edges("replan_step", should_end, {"execute_plan": "execute_plan", "__end__": "__end__"})       

graph = workflow.compile()

config = {"recursion_limit": 30}

import asyncio  

async def run():
    inputs = {"input_query":  "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace weighs. What are their combined weights in pounds?"}
    async for event in graph.astream(inputs, config=config):
        for node_name, node_output in event.items():
            print(f"Node: {node_name}")
            print(f"Output: {node_output}")
            print("---")


if __name__ == "__main__":
    asyncio.run(run())