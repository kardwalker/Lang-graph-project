from typing_extensions import TypedDict
from langgraph.graph import StateGraph , START , END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel , Field


class State(BaseModel):
    amount :  int | float= Field(..., description="The enter to amount for transaction")

def define_transaction(state : State) :
    print("defining the amount")
    return state

def verify_amount(state : State):
    print(f"verifying transaction amount : {state.amount}")
    return state

buidler = StateGraph(State)   
buidler.add_node("define_transaction", define_transaction) 
buidler.add_node("verify_amount", verify_amount)
buidler.add_edge(START , "define_transaction")
buidler.add_edge("define_transaction","verify_amount")
buidler.add_edge("verify_amount", END)  

graph = buidler.compile(interrupt_after=["define_transaction"], checkpointer=MemorySaver())
config = {"configurable" : {"thread_id" : "bdsjk"}}
intiat = {"amount": 1000.0}
for event in graph.stream(intiat, config):
    print(event)

approval = input("approve the transaction ? (yes / no): " )
if approval.lower() == "yes":
    for event in graph.stream(None , config):
        print(event)

else:
    print("Transaction cancelled")        