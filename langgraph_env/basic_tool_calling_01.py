from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage,AIMessage, ToolCall
print("This code implements a basic tool calling example using LangGraph.")
@tool
def get_weather(location: str) -> str:
    """
    A mock function to simulate getting weather information.
    In a real application, this would call a weather API.
    """
    weather_data = {
        "Mumbai": "It's sunny in Mumbai with a temperature of 30°C.",
        "Kolkata": "It's rainy in Kolkata with a temperature of 25°C.",
        "Delhi": "It's cloudy in Delhi with a temperature of 28°C.",
        "Bangalore": "It's pleasant in Bangalore with a temperature of 22°C.",
        "Chennai": "It's hot in Chennai with a temperature of 35°C.",
        "Hyderabad": "It's humid in Hyderabad with a temperature of 29°C.",
        "Pune": "It's windy in Pune with a temperature of 27°C.",
    }
    return weather_data.get(location, "Weather information not available for this location.")

tools = [get_weather]
tool_node = ToolNode(tools, handle_tool_errors=True)


# Create proper ToolCall object
tool_call = ToolCall(
    name="get_weather",
    args={"location": "Mumbai"},
    id="tool_call_1",
    type="function"
)

message_with_tool = AIMessage(
    content="",
    tool_calls=[tool_call]
)

state = {
    'messages': [message_with_tool]
}
result = tool_node.invoke(state)
print(result)     
# tip  
print(help(ToolCall))
