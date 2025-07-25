from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph , MessagesState , START , END
from typing import TypedDict

class React_Agent_State(TypedDict):
    message : str
    action : str
    sub_action : str

def resoning_node(state : React_Agent_State):
    query = state["message"]
    if "weather" in query:
        return {"action" : "fetch_weather"}
    elif "news" in query:
        return {"action" : "fetch_news"}
    elif "recommend" in query:
        return {"action" : "recommendation" ,"sub_action" : "book"} 
    else:
        return {"action" : "unknown"}

def weather_sgraph_n(state: React_Agent_State):
    return {"message" : "The weather is good as hell"}

def news_sgraph_n(state : React_Agent_State):
    return {"message" : "Jane Street scam upto 4000cr from indian economy"
            