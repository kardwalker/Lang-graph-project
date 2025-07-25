from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
import os
load_dotenv()
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini" # <--- Verify this deployment name
# Initialize the LLM (using OpenAI in this example)
#api_key = os.getenv("OPENAI_API_KEY")
#model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
from langchain.chat_models import init_chat_model
model  = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("Azure_endpoint"),
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",  # Ensure this matches your deployment name
)


# Function to handle the user query and call the LLM
def call_llm(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages[-1].content)   
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)

# Add the node to call the LLM
workflow.add_node("call_llm", call_llm)

# Define the edges (start -> LLM -> end)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)

# Compile the workflow
app = workflow.compile()

# Example input message from the user
input_message = {
    "messages": [("human", "What is the capital of Kenya?")]
}

res = app.invoke(input_message)
print(res)