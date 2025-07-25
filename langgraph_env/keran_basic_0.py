from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State_Message(TypedDict):
    message : str
    is_final : bool

def welcome_message(state: State_Message):
    state['message'] = "Welcome to our service!"
    return state



def Premium_longue_greeting(state : State_Message):
    state['message'] = "Hello, welcome to our premium service! How can I assist you today?"
    return state

def Classic_longue_greeting(state : State_Message):
    state['message'] = "Hello, welcome to our classic service! How can I assist you today?"
    return state
def check_premium(state: State_Message):
    if state['is_final']:
        state['user_type'] = "premium"
    else:
        state['user_type'] = "classic"
    return state



"""def check_premium(state : State_Message):
    if state['is_final']:
        return "Premium_longue_greeting"
    else:
        return 'Classic_longue_greeting'
"""
graph_builder = StateGraph(State_Message)
graph_builder.add_node("welcome", welcome_message)
graph_builder.add_node("o_premium_longue_greeting", Premium_longue_greeting)
graph_builder.add_node("classic_longue_greeting", Classic_longue_greeting)     
graph_builder.add_node("check_premium", check_premium)
graph_builder.add_edge(START, "welcome") 
graph_builder.add_edge("welcome", "check_premium")
graph_builder.add_conditional_edges(
    "check_premium",
    lambda state: state["user_type"],
    {
        "premium": "o_premium_longue_greeting",
        "classic": "classic_longue_greeting",
    }
)
#graph_builder.add_conditional_edges("check_premium", check_premium)
graph_builder.add_edge("o_premium_longue_greeting", END)  
graph_builder.add_edge("classic_longue_greeting", END)  

graph = graph_builder.compile()
print(graph.invoke({'message': '', 'is_final': True}))  # Example invocation with is_final set to True
print(graph.invoke({'message': '', 'is_final': False}))  # Example invocation with is
