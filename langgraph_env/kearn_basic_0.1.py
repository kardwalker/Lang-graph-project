from langgraph.graph import StateGraph, START, END
from typing import TypedDict ,Literal , Dict

"""class proxy_User_State(TypedDict):
    message: str
    is_final: Literal["premium", "classic", "Guest" ,"admin"]
    user_type: str  
   # info : {Username : str, Password: str, ip_address: str}
   remeber the error of last line
   This is not valid Python syntax for type annotations. You cannot use a dictionary literal to specify a type directly in an annotation."""

class InfoDict(TypedDict):
    Username: str
    Password: str
    ip_address: str

class proxy_User_State(TypedDict):
    message: str
    is_final: Literal["premium", "classic", "Guest", "admin"]
    user_type: str
    info: InfoDict

def welcome( state: proxy_User_State):
    state["message"] += "Welcome to our Proxy Server Network! "
    return state

def subscription_type(state: proxy_User_State):
    if state["is_final"] == "premium":
        state["user_type"] = "premium"
    elif state["is_final"] == "classic":
        state["user_type"] = "classic"
    elif state["is_final"] == "Guest":
        state["user_type"] = "Guest"
    elif state["is_final"] == "admin":
        state["user_type"] = "admin"
    return state    


def premium_proxy_serivce(state: proxy_User_State):
    state["message"] += "You have access to our premium proxy services"
    state["info"] = InfoDict(Username="pre_porxy_i_fgk89", Password="3798bkoj939739hfjdkj", ip_address="127.285.743.0")
    return state

def classic_proxy_service(state: proxy_User_State):
    state["message"] += "You have access to our classic proxy services"
    state["info"] = InfoDict(Username="cls_porxy_i_f83209", Password="373985%^889(*&$98" , ip_address="127.192.743.1")
    return state


def guest_proxy_service(state: proxy_User_State):
    state["message"] += "You have access to our guest proxy services. "
    "You can browse the internet with limited features."
    state["info"] = InfoDict(Username="guest_porxy_i_f83209", Password="guest_password", ip_address="127.904.743.2")


def admin_proxy_serive(state : proxy_User_State):
    state["message"] += "You have access to our admin proxy services. "
    "You can manage the proxy server and monitor user activity."
    state["info"] = InfoDict(Username="admin_porxy_i_f83209", Password="admin_password", ip_address="127.692.743.3")
    return state

def server_status(state: proxy_User_State):
    state["message"] += "you are connected to our proxy server. "
    return state

# Buildiung the Node of the StateGraph
graph_builder = StateGraph(proxy_User_State)
graph_builder.add_node("welcome", welcome)
graph_builder.add_node("subscription_type", subscription_type)
graph_builder.add_node("premium_proxy_serivce", premium_proxy_serivce)
graph_builder.add_node("classic_proxy_service", classic_proxy_service)
graph_builder.add_node("guest_proxy_service", guest_proxy_service)
graph_builder.add_node("admin_proxy_serive", admin_proxy_serive)
graph_builder.add_node("server_status", server_status)

# Building the Edges of the StateGraph

graph_builder.add_edge(START, "welcome")
graph_builder.add_edge("welcome", "subscription_type")
#graph_builder.add_edge("subscription_type", "premium_proxy_serivce")    
#graph_builder.add_edge("subscription_type", "guest_proxy_service")
#graph_builder.add_edge("subscription_type", "admin_proxy_serive")
#graph_builder.add_edge("premium_proxy_serivce", "server_status")
# because we are using conditional edges, we don't need to add these edges directly
# langgraph will dynamically create the edges based on the condition defined in the "conditional_edges"


graph_builder.add_edge("classic_proxy_service", "server_status")
graph_builder.add_edge("guest_proxy_service", "server_status")
graph_builder.add_edge("admin_proxy_serive", "server_status")    
graph_builder.add_edge("server_status", END)
graph_builder.add_conditional_edges(
    "subscription_type",        
    #lambda state: state["is_final"],
    subscription_type,
    {
        "premium": "premium_proxy_serivce",
        "classic": "classic_proxy_service",
        "Guest": "guest_proxy_service",
        "admin": "admin_proxy_serive"
    })

# Compiling the graph

graph = graph_builder.compile()
print(graph.invoke({'message': '', 'is_final': 'premium', 'user_type': '', 'info': InfoDict(Username='', Password='', ip_address='')}))  # Example invocation

