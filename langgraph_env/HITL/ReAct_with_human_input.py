import os
from langgraph.graph import START, END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from typing import TypedDict

class State(TypedDict):
    inp : str 
    user_feedback :str

def agent_reasoning(state : State):
    print(f"Agentis reasoning: {state["inp"]}")
    if len(state["inp"]) > 10:
        print("agent needs clarifiacation")
        return state
    else:
        state["user_feedback"] = "No clarification needed"
        return state
def ask_human(state : State):
    print("---Asking for human feedback ----")
    feedback = input("please provide the feedback:")
    state["user_feedback"] = feedback
    return state


def perform_action(user_feedback : str):
    return {"user_feedback":f"Feedback -processed {user_feedback}"}

buidler = StateGraph(State)
buidler.add_node("agent" , agent_reasoning)
buidler.add_node("ask_human", ask_human)
buidler.add_node("perform_action", perform_action)
buidler.add_edge(START, "agent")
buidler.add_conditional_edges("agent",lambda state : "ask_human"if len(state["inp"]) > 10 else "perform_action",
                              {
                                  "ask_human" : "ask_human" , "perform_action" : "perform_action"
                              })

buidler.add_edge("ask_human", "perform_action")
buidler.add_edge("perform_action", END)

graph = buidler.compile(checkpointer=MemorySaver(), interrupt_before=["ask_human"])

i = {"inp" :"Procced with reasoning"}

config = {"configurable" : {"thread_id": "threa_1"}}
for event in graph.stream(i, config, stream_mode="values"):
    print(event)

user_feed = input("user feedback")
graph.update_state(config, {"user_feedback":user_feed}, as_node="ask_human")
for event in graph.stream(None, config=config, stream_mode="values"):
    print(event)

