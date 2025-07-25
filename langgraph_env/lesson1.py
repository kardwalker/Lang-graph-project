from typing import TypedDict


class welcome_state(TypedDict):
    greeting: str

# Define the node function

def welcome_node(state : welcome_state) -> welcome_state:
    """
    A simple node function that returns a greeting message.
    """
    state['greeting'] = "Welcome to LangGraph! " + state['greeting']
    return state


from langgraph.graph import StateGraph, START , END

builder = StateGraph(welcome_state)
builder.add_node("greet", welcome_node)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)

graph = builder.compile()

result = graph.invoke(welcome_state(greeting="Agent"))
result1 = graph.invoke({'greeting': "Agent"})
print(result1)


# code to visualize 
from langchain_core.runnables.graph import MermaidDrawMethod
import random
import os
import sys
import subprocess
mermaid_png = graph.get_graph(xray=1).draw_mermaid_png(draw_method=MermaidDrawMethod.API)
output_folder = "."
os.makedirs(output_folder, exist_ok=True)
filename = os.path.join(output_folder, f"graph_{random.randint(1,10000)}.png")
with open(filename,"wb") as f:
    f.write(mermaid_png)
    