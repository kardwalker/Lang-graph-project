from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
load_dotenv()
import wandb


# initialize wandb run
wandb.init(
    project="simple React",
    name="azure-agent-run",
    config={
         "model": "gpt-4o-mini",
        "temperature": 0.8,
        "api_version": "2024-12-01-preview"}
)


# fucntion to log tool calls to wandb
def log_tool_call(name, inputs, ouptut):
    wandb.log({
        "tool" :name,
        "inputs": inputs,
        "output" : ouptut
    })

def add(a: int, b: int) -> int:
    """Adds two numbers."""
    result = a + b
    log_tool_call("add", {"a" : a, "b" : b}, result)
    return result


def subtract(a: int, b: int) -> int:
    """Subtracts two numbers."""
    result = a- b
    log_tool_call("subtract", {"a": a, "b": b}, result)
    return result


def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    result = a *b
    log_tool_call("multiply", {"a": a, "b": b}, result)
    return result 

def divide(a: int, b: int) -> float:
    """Divides two numbers."""  
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    result = a/b
    log_tool_call("divide", {"a": a, "b": b}, result)
    return  result 


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



# Create a ReAct agent with the defined tools
react_agent = create_react_agent(model = model, tools=[add, subtract, multiply, divide],
                                  prompt="You are a helpful assistant that can perform basic arithmetic operations.")

"""reasoning node (agent node) first processes he input and with the language model 
to determione whether it  can respondedirestly or it needs to cal a tool

Tool Node : if the model identifies a need for at tool  ,it triggers the revelant mode

Result integration node : after the tool completes its exceution , the result is passed back into the mdel 

Final response node : the model generates a final response based on the tool results and the original input"""
inputs = {  
    "messages" : [ ("user", "add 3 and 5 and multiply the result by 19")]
}

wandb.log({"user_input" : str(inputs)})


# Invoke the ReAct agent with the inputs
response = react_agent.invoke(inputs)
print(response)
wandb.log({"final_response" : str(response)})
wandb.finish()
"""THis simple cycle of reaoning . tool calling and responding form the corre the React farmework . the agnet continous  looping between
reasoning and acting untio no furher actions is required"""