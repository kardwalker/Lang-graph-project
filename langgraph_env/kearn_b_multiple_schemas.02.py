
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from typing import TypedDict

# Define the input and output types for the state
class OverallState(TypedDict):
    partial_message: str
    user_input: str
    message_output: str

class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    message_output: str

class PrivateState(TypedDict):
    private_message: str

# Define the nodes (functions) that operate on the states
def add_world(state: InputState) -> OverallState:
    # This node appends " World" to the greeting
    partial_message = state["user_input"] + " World"
    print(f"Node 1 - add_world: Transformed '{state['user_input']}' to '{partial_message}'")
    return {"partial_message": partial_message, "user_input": state["user_input"], "message_output": ""}

def add_exclamation(state: OverallState) -> PrivateState:
    # This node appends "!" to the partial message
    private_message = state["partial_message"] + "!"
    print(f"Node 2 - add_exclamation: Transformed '{state['partial_message']}' to '{private_message}'")
    return {"private_message": private_message}

def finalize_message(state: PrivateState) -> OutputState:
    # This node sets the final output
    message_output = state["private_message"]
    print(f"Node 3 - finalize_message: Finalized message to '{message_output}'")
    return {"message_output": message_output}

# Create the graph builder with nodes and edges
builder = StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node("add_world", add_world)
builder.add_node("add_exclamation", add_exclamation)
builder.add_node("finalize_message", finalize_message)
builder.add_edge(START, "add_world")
builder.add_edge("add_world", "add_exclamation")
builder.add_edge("add_exclamation", "finalize_message")
builder.add_edge("finalize_message", END)

# Compile and invoke the graph
graph = builder.compile()
result = graph.invoke({"user_input": "Hello"})
print(result)  # {'full_message': 'Hello World!'}