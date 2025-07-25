from langgraph.graph import StateGraph, MessagesState, START, END
import requests
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

weather_api_key = os.getenv("weather_api_key")

def get_weather(location: str) -> str:
    """
    Fetches weather information for a given location using the OpenWeatherMap API.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        return {"messages" : [f"The weather in {location} is currently {weather_description} with a temperature of {temperature}°C."]}
    else:
        return {"messages" :["Weather information not available for this location."]}
    
builder = StateGraph(MessagesState)

builder.add_node("get_weather", get_weather)
builder.add_edge(START, "get_weather")
builder.add_edge("get_weather", END)    
graph = builder.compile()

def interact_with_user():
    while True:
        user_input = input("Enter a location to get the weather (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the session.")
            break
        
        messages = [HumanMessage(content=user_input)]
        
        response = graph.invoke({"messages": messages})
        
        for r in response["messages"]:
            r.pretty_print()
interact_with_user()            