import asyncio
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent   
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()
import operator
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()
import operator

# Define TypedDict for state
class plan_n_excecute(TypedDict):
    input_query: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: Union[str, None]
    error: Union[str, None]

@tool
def indentify_product(issue : str) -> str:
    """
    Identify the product based on the issue description.
    """
    # Placeholder for actual product identification logic
    if not issue or not isinstance(issue, str):
        return "Invalid issue description provided."

    return f"Product identified based on the issue {issue}."

@tool
def escaluate_to_support(issue: str, product : str) -> str:
    """Escalates the issue to a human support team.""" 
    if not product or not issue or not isinstance(product, str) or not isinstance(issue, str):
        return "Invalid product or issue description provided."


tools = [indentify_product, escaluate_to_support]

SYSTEM_PROMPT = """You are a customer support agent.
YOU are responsible for assisting customers with their inquiries and issues.
Your goal is to provide accurate information and escalate issues to the appropriate teams when necessary.

 Always identify the issue and product from the input and use it consistently across all steps.

 Available tools :
  1. identify_product - Identifies the product based on the input.
 
  2. escalate_to_support - Escalates the issue to a human support team.
  Ensure each step is completed before moving to the next one.
"""

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"), 
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.3,
    max_tokens=512, )

agent_executor = create_react_agent(
    model=model,    
    tools=tools,
    prompt=ChatPromptTemplate.from_messages([    
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}")
    ])
)

class Plan(BaseModel):
    steps: List[str] = Field(..., description="The steps to be executed by the agent.")
    

class Response(BaseModel):
    response: str = Field(..., description="The response from the agent after executing the steps.")

class Action(BaseModel):
    action: Union[Response, Plan] = Field(..., description="The action to be performed by the agent. If you need to further use tools to get the answer, use Plan.")  



async def plan_step(state: plan_n_excecute) -> dict:
    try:
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Create a plan to address the following user query: {input_query}")
        ])
        planner = planner_prompt | AzureChatOpenAI(
            api_key=os.getenv("AZURE_API_KEY"), 
            azure_endpoint=os.getenv("Azure_endpoint"),
            api_version="2024-12-01-preview",
            model="gpt-4o-mini",
            streaming=True,
            temperature=0.2,
            max_tokens=1024,
            azure_deployment="gpt-4o-mini",).with_structured_output(Plan)
        
        plan = await planner.ainvoke({
            "input_query": state['input_query']
        })
        return {"plan": plan.steps}
    except Exception as e:
        return {"error": str(e)}
    
async def execute_steps(state: plan_n_excecute) -> dict:
    try:
        if state.get("error"):
            return {"error": state["error"]}
        plan = state.get("plan", [])
        if not plan:
            return {"response": "No plan generated."}   
        
        task = plan[0]
        remaining_plan = plan[1:]
        
        agent_response = await agent_executor.ainvoke(
            {"messages" : [("user", f"Execute this step: {task}. Full query for context: {state['input_query']}")]})
        
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
            "plan": remaining_plan
        }
    except Exception as e:
        return {"error": str(e)}
    


async def replan_step(state : plan_n_excecute) -> dict:
    try:
        if state.get("error"):
            return {"response": state["error"]}

        if not state.get("plan"):
            # If the plan is empty, we are done
            return {"response": state['past_steps'][-1][1]}

        # If there are still steps, continue execution
        return {}

    except Exception as e:
        return {"error": str(e)}


builder = StateGraph(plan_n_excecute)
builder.add_node("planner", plan_step)
builder.add_node("executor", execute_steps)  
builder.add_node("replanner", replan_step)

## add edges

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")     
builder.add_edge("executor", "replanner")
builder.add_conditional_edges(
    "replanner", 
    lambda state: "executor" if state.get("plan") else END,
    {
        "executor": "executor",
        END: END
    }
)

graph = builder.compile(
    checkpointer=MemorySaver(),
    name="Plan and Execute Workflow",)

config = {"configurable":{"thread_id":"1"}, "recursion_limit": 50}

async def run_agent():
    query = input("write your query : ")
    if not query:
        print("No query provided. Exiting.")
        return
        
    inputs = {"input_query": query}
    async for event in graph.astream(inputs, config=config):
        for node, output in event.items():
            print(f"--- Node: {node} ---")
            print(output)
            print("---")

if __name__ == "__main__":
    asyncio.run(run_agent())    

    """write your query : help me hardware of windows laptop
--- Node: planner ---
{'plan': ['Identify the product as a Windows laptop.', 'Determine the specific hardware issue the user is experiencing with their Windows laptop.', 'Provide troubleshooting steps or solutions for the identified hardware issue.', 'If the issue cannot be resolved, escalate the issue to the human support team for further assistance.']}
---
--- Node: executor ---
{'past_steps': [('Identify the product as a Windows laptop.', 'The product has been identified as a Windows laptop. \n\nNow, please provide me with more details about the specific issue you are facing with the hardware of your Windows laptop, so I can assist you further.')], 'plan': ['Determine the specific hardware issue the user is experiencing with their Windows laptop.', 'Provide troubleshooting steps or solutions for the identified hardware issue.', 'If the issue cannot be resolved, escalate the issue to the human support team for further assistance.']}
---
--- Node: replanner ---
None
---
--- Node: executor ---
{'past_steps': [('Determine the specific hardware issue the user is experiencing with their Windows laptop.', 'I have identified the product as a Windows laptop and escalated the hardware issue to our human support team for further assistance. They will be able to help you with your inquiry.')], 'plan': ['Provide troubleshooting steps or solutions for the identified hardware issue.', 'If the issue cannot be resolved, escalate the issue to the human support team for further assistance.']}
---
--- Node: replanner ---
None
---
--- Node: executor ---
{'past_steps': [('Provide troubleshooting steps or solutions for the identified hardware issue.', 'The issue regarding the hardware of your Windows laptop has been successfully escalated to our human support team. They will reach out to you shortly to provide the necessary troubleshooting steps or solutions. \n\nIf you have any more questions or need further assistance, feel free to ask!')], 'plan': ['If the issue cannot be resolved, escalate the issue to the human support team for further assistance.']}
---
--- Node: replanner ---
None
---
--- Node: executor ---
{'past_steps': [('If the issue cannot be resolved, escalate the issue to the human support team for further assistance.', 'The issue regarding the hardware of your Windows laptop has been successfully escalated to the human support team for further assistance. They will reach out to you shortly. If you have any other questions or need further help, feel free to ask!')], 'plan': []}
---
--- Node: replanner ---
{'response': 'The issue regarding the hardware of your Windows laptop has been successfully escalated to the human support team for further assistance. They will reach out to you shortly. If you have any other questions or need further help, feel free to ask!'}
---"""