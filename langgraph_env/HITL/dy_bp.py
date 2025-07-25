from langgraph.graph import StateGraph , START , END
import os
from typing import TypedDict
from langgraph.errors import NodeInterrupt

class State(TypedDict):
    input : str


builder = StateGraph(State)
def step_dynamic_interrupt(state :State):
    input_lenght = len(state["input"])
    if input_lenght > 10 :
        raise NodeInterrupt("Input_ length {input_length exceeds threadhold of 10")
    return state


builder.add_node("dp", step_dynamic_interrupt)
builder.add_edge(START, "dp")
builder.add_edge("dp", END)

graph = builder.compile()

ini_inp = {"input": "This is a long input"}
for event in graph.stream(ini_inp):
    print(event)