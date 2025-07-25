import os
from typing import TypedDict
from langgraph.graph import START, StateGraph , END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    input : str

def step1(state: State):
    print("step 12")
    return state

def step2(state : State):
    print("step 2")
    return state
memory = MemorySaver()
config = {"configurable":{"thread_id" : "thread_1"}}
builder = StateGraph(State)
builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", END)

graph = builder.compile(checkpointer=memory, interrupt_after=["step1"])
graph.invoke({"messages" : "woor"},config=config)
graph.update_state(config=config, values={"messages" : "Let's excel indefinitely" })
for event in graph.stream(None,config=config):
    print(event)