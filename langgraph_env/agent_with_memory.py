from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from datetime import datetime
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_preferences: Dict[str, Any]
    conversation_context: Dict[str, Any]
    task_history: List[Dict[str, Any]]
    current_mood: str
    interaction_count: int

# Enhanced tools with state awareness
@tool
def smart_weather(location: str, preferences: str = "") -> str:
    """Get weather with personalized recommendations based on user preferences."""
    weather_data = {
        "New York": {"temp": 22, "condition": "Sunny", "humidity": 65, "wind": 10},
        "London": {"temp": 15, "condition": "Cloudy", "humidity": 80, "wind": 15},
        "Tokyo": {"temp": 28, "condition": "Rainy", "humidity": 85, "wind": 8},
        "San Francisco": {"temp": 18, "condition": "Foggy", "humidity": 75, "wind": 12}
    }
    
    data = weather_data.get(location, {"temp": 20, "condition": "Unknown", "humidity": 50, "wind": 5})
    
    # Parse preferences for personalized advice
    advice = ""
    if "cold_sensitive" in preferences.lower():
        if data["temp"] < 20:
            advice = " 🧥 Recommendation: Wear a warm jacket!"
    elif "heat_sensitive" in preferences.lower():
        if data["temp"] > 25:
            advice = " 🌂 Recommendation: Stay hydrated and find shade!"
    
    if "exercise" in preferences.lower():
        if data["condition"] == "Sunny" and 18 <= data["temp"] <= 25:
            advice += " 🏃 Great weather for outdoor exercise!"
        elif data["condition"] == "Rainy":
            advice += " 🏠 Consider indoor workout today!"
    
    return f"Weather in {location}: {data['temp']}°C, {data['condition']}, Humidity: {data['humidity']}%, Wind: {data['wind']}km/h{advice}"

@tool
def mood_tracker(current_mood: str, activity: str = "") -> str:
    """Track and provide mood-based recommendations."""
    mood_responses = {
        "happy": "That's wonderful! Here are some activities to maintain your good mood:",
        "sad": "I'm sorry you're feeling down. Here are some mood-lifting suggestions:",
        "stressed": "Let's work on reducing that stress. Here are some calming activities:",
        "excited": "Great energy! Here are ways to channel that excitement:",
        "tired": "You sound tired. Here are some energizing or restful options:"
    }
    
    activity_suggestions = {
        "happy": ["Share your joy with friends", "Try a new hobby", "Plan something fun"],
        "sad": ["Listen to uplifting music", "Call a friend", "Take a walk in nature"],
        "stressed": ["Try deep breathing", "Do some light exercise", "Listen to calming music"],
        "excited": ["Channel energy into a project", "Try something creative", "Share enthusiasm with others"],
        "tired": ["Take a short nap", "Have some green tea", "Do gentle stretching"]
    }
    
    response = mood_responses.get(current_mood.lower(), "How are you feeling today?")
    suggestions = activity_suggestions.get(current_mood.lower(), ["Take care of yourself"])
    
    return f"{response}\n• " + "\n• ".join(suggestions)

@tool
def task_planner(task_description: str, priority: str = "medium", context: str = "") -> str:
    """Plan and organize tasks with context awareness."""
    priority_levels = {"low": 1, "medium": 2, "high": 3, "urgent": 4}
    priority_num = priority_levels.get(priority.lower(), 2)
    
    # Generate task breakdown
    task_plan = {
        "task": task_description,
        "priority": priority,
        "priority_score": priority_num,
        "estimated_time": "30-60 minutes",  # Simplified estimation
        "created_at": datetime.now().isoformat(),
        "status": "planned"
    }
    
    # Add context-based suggestions
    suggestions = []
    if "meeting" in task_description.lower():
        suggestions.extend(["Prepare agenda", "Set up calendar invite", "Gather materials"])
    elif "project" in task_description.lower():
        suggestions.extend(["Break into smaller tasks", "Set milestones", "Identify resources needed"])
    elif "learning" in task_description.lower():
        suggestions.extend(["Find learning materials", "Set study schedule", "Create practice exercises"])
    
    result = f"Task Plan Created:\n📋 {task_description}\n⭐ Priority: {priority}\n⏱️ Estimated time: {task_plan['estimated_time']}"
    
    if suggestions:
        result += f"\n\n💡 Suggestions:\n• " + "\n• ".join(suggestions)
    
    return result

@tool
def memory_manager(action: str, key: str = "", value: str = "", context: str = "") -> str:
    """Manage long-term memory with context and categorization."""
    if not hasattr(memory_manager, 'memory'):
        memory_manager.memory = {
            "preferences": {},
            "facts": {},
            "experiences": {},
            "goals": {}
        }
    
    if action == "store":
        # Auto-categorize based on content
        category = "facts"  # default
        if any(word in key.lower() for word in ["like", "prefer", "favorite", "love", "hate"]):
            category = "preferences"
        elif any(word in key.lower() for word in ["goal", "want", "plan", "wish"]):
            category = "goals"
        elif any(word in key.lower() for word in ["did", "went", "saw", "experience"]):
            category = "experiences"
        
        memory_manager.memory[category][key] = {
            "value": value,
            "stored_at": datetime.now().isoformat(),
            "context": context
        }
        return f"Stored in {category}: {key} = {value}"
    
    elif action == "recall":
        for category, items in memory_manager.memory.items():
            if key in items:
                item = items[key]
                return f"From {category}: {key} = {item['value']} (stored: {item['stored_at']})"
        return f"No memory found for: {key}"
    
    elif action == "list":
        summary = []
        for category, items in memory_manager.memory.items():
            if items:
                summary.append(f"{category.title()}: {len(items)} items")
        return "Memory Summary:\n• " + "\n• ".join(summary) if summary else "No memories stored yet"
    
    return "Available actions: store, recall, list"

class StatefulLangGraphAgent:
    """A stateful agent built with LangGraph"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # Initialize the language model
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("Azure_endpoint"),
            api_version="2024-12-01-preview",
            model = "gpt-4o-mini",
            temperature=0.1,
            max_tokens=512,
        )
        
        # Define tools
        self.tools = [smart_weather, mood_tracker, task_planner, memory_manager]
        self.tool_executor = ToolExecutor(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        
        def agent_node(state: AgentState):
            """Main agent reasoning node"""
            messages = state["messages"]
            user_preferences = state.get("user_preferences", {})
            mood = state.get("current_mood", "neutral")
            interaction_count = state.get("interaction_count", 0)
            
            # Create system message with context
            system_context = f"""You are an intelligent personal assistant with memory and context awareness.
            
            Current user context:
            - Mood: {mood}
            - Interaction count: {interaction_count}
            - Preferences: {json.dumps(user_preferences, indent=2) if user_preferences else 'None stored yet'}
            
            You have access to tools for:
            - Weather information with personalized recommendations
            - Mood tracking and suggestions
            - Task planning and organization
            - Memory management (store/recall personal information)
            
            Be conversational, helpful, and remember context from previous interactions.
            Use tools when appropriate to provide better assistance.
            """
            
            # Add system message if not present
            if not messages or not any(isinstance(msg, AIMessage) for msg in messages):
                messages = [AIMessage(content=system_context)] + messages
            
            # Get LLM response
            response = self.llm.bind_tools(self.tools).invoke(messages)
            
            return {
                "messages": [response],
                "interaction_count": interaction_count + 1
            }
        
        def tool_node(state: AgentState):
            """Execute tools based on agent's decision"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # Execute tools
            tool_invocations = []
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_invocation = ToolInvocation(
                        tool=tool_call["name"],
                        tool_input=tool_call["args"]
                    )
                    tool_invocations.append(tool_invocation)
            
            # Execute all tool calls
            tool_messages = []
            for invocation in tool_invocations:
                try:
                    result = self.tool_executor.invoke(invocation)
                    tool_messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=invocation.tool
                    ))
                except Exception as e:
                    tool_messages.append(ToolMessage(
                        content=f"Error executing {invocation.tool}: {str(e)}",
                        tool_call_id=invocation.tool
                    ))
            
            return {"messages": tool_messages}
        
        def context_updater(state: AgentState):
            """Update context based on conversation"""
            messages = state["messages"]
            current_context = state.get("conversation_context", {})
            user_preferences = state.get("user_preferences", {})
            
            # Extract context from recent messages
            recent_human_messages = [msg for msg in messages[-5:] if isinstance(msg, HumanMessage)]
            
            # Simple mood detection
            mood = "neutral"
            if recent_human_messages:
                last_message = recent_human_messages[-1].content.lower()
                if any(word in last_message for word in ["happy", "great", "good", "awesome", "love"]):
                    mood = "happy"
                elif any(word in last_message for word in ["sad", "bad", "terrible", "awful", "hate"]):
                    mood = "sad"
                elif any(word in last_message for word in ["stress", "worried", "anxious", "overwhelmed"]):
                    mood = "stressed"
                elif any(word in last_message for word in ["excited", "amazing", "fantastic", "thrilled"]):
                    mood = "excited"
                elif any(word in last_message for word in ["tired", "exhausted", "sleepy", "fatigue"]):
                    mood = "tired"
            
            return {
                "current_mood": mood,
                "conversation_context": current_context
            }
        
        def should_continue(state: AgentState):
            """Decide whether to continue with tools or end"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the last message has tool calls, go to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            else:
                return "end"
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("context_updater", context_updater)
        
        # Set entry point
        workflow.set_entry_point("context_updater")
        
        # Add edges
        workflow.add_edge("context_updater", "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def run(self, message: str, state: AgentState = None) -> Dict[str, Any]:
        """Run the agent with a message"""
        if state is None:
            state = {
                "messages": [],
                "user_preferences": {},
                "conversation_context": {},
                "task_history": [],
                "current_mood": "neutral",
                "interaction_count": 0
            }
        
        # Add user message
        state["messages"].append(HumanMessage(content=message))
        
        # Run the graph
        result = self.graph.invoke(state)
        
        return result
    
    def chat(self):
        """Interactive chat mode"""
        print("🤖 Stateful LangGraph Agent - Type 'quit' to exit")
        print("Features: Memory, Mood tracking, Task planning, Personalized weather")
        print("-" * 60)
        
        # Initialize persistent state
        state = {
            "messages": [],
            "user_preferences": {},
            "conversation_context": {},
            "task_history": [],
            "current_mood": "neutral",
            "interaction_count": 0
        }
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! I'll remember our conversation for next time. 😊")
                break
            
            if user_input:
                try:
                    result = self.run(user_input, state)
                    
                    # Update state for next iteration
                    state = result
                    
                    # Get the last AI message
                    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                    if ai_messages:
                        response = ai_messages[-1].content
                        print(f"\nAgent: {response}")
                        
                        # Show current mood if changed
                        if result.get("current_mood", "neutral") != "neutral":
                            print(f"(Detected mood: {result['current_mood']})")
                    
                except Exception as e:
                    print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        agent = StatefulLangGraphAgent()
        
        print("=== Testing Stateful LangGraph Agent ===")
        
        # Test conversation with state
        test_conversations = [
            "Hi! I'm feeling pretty stressed about work today.",
            "I really hate cold weather. What's the weather like in New York?",
            "Can you help me plan a project presentation for next week?",
            "Remember that my favorite color is blue and I love hiking.",
            "What do you remember about me?",
            "I'm excited about my hiking trip this weekend! What's the weather like in San Francisco?"
        ]
        
        state = None
        for i, message in enumerate(test_conversations, 1):
            print(f"\n{'='*50}")
            print(f"Turn {i}: {message}")
            print('='*50)
            
            result = agent.run(message, state)
            state = result
            
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                print(f"Agent: {ai_messages[-1].content}")
            
            print(f"Current mood: {result.get('current_mood', 'neutral')}")
            print(f"Interactions: {result.get('interaction_count', 0)}")
        
        # Uncomment to start interactive mode
        # agent.chat()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: pip install langgraph langchain langchain-openai")