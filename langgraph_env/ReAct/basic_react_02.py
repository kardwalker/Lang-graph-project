print("This  code implement React agent with Single-Thread Memory")

print("Integrating MemorySaver")
#to manage to memory within a single - memory session


from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
load_dotenv()


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

# Define tools
def product_info(product_name : str) -> str:
    """Fetch the product"""
    product_catlog = {
        "iPhone 20 " : "The latest iphone feature an A15 chip with improved camera feature",
        "Macbook 3" : "The mew Macbook had M3 chip and 14ich retina display"
    }


checkpointer = MemorySaver()
config = {"configurable":{"thread_id": "thread_1"}}
graph = create_react_agent(model= model , tools=[product_info] , checkpointer=checkpointer)

inp = {"messages":[("user","Hi, Tell me about iphone")]}
messages = graph.invoke(inp , config=config)
for m in messages["messages"]:
    m.pretty_print()

