import os 
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages : str

def gen_rep(state: State):
    print("generating the report based on the provided messages")
    return state

builder = StateGraph(State)
builder.add_node("gen_rep", gen_rep)
builder.add_edge(START, "gen_rep")
builder.add_edge("gen_rep", END)

graph = builder.compile(interrupt_before=["gen_rep"], checkpointer=MemorySaver())
ini_input = {"work":"job_description of AI compaines"}
config = {"configurable":{"thread_id":"2"}}
for event in graph.stream(ini_input, config):
    print(event)
approval = input("approve gen_rep ? (yes/no)")
if approval.lower() == "yes":
    for event in graph.stream(None, config):
        print(event)
else:
    print("report generation cancelled")
    



