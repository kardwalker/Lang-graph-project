print("this code implement plan and exceut agent")

print("Enhanced Plan and Execute agent using LangGraph with improved error handling and execution logic.")

import operator
import os
from typing import Annotated, List, Tuple, Union, TypedDict, Optional
from enum import Enum
import asyncio
from datetime import datetime
import json

from langchain_tavily import TavilySearch
from langgraph.graph import START, END, StateGraph
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Enhanced configuration
class Config:
    MAX_RETRIES = 3
    MAX_PLAN_LENGTH = 10
    EXECUTION_TIMEOUT = 30  # seconds per step
    TEMPERATURE = 0.7
    MAX_TOKENS = 1024

# Set up environment variable
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# Enhanced tools setup with error handling
def create_tools():
    try:
        return [TavilySearch(max_results=5)]
    except Exception as e:
        print(f"Warning: Failed to initialize Tavily search: {e}")
        return []

tools = create_tools()

# Enhanced prompt with better instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can search the web for information.
    When executing tasks:
    1. Be precise and thorough in your searches
    2. Verify information when possible
    3. If a search fails, try alternative queries
    4. Always provide clear, actionable results"""),
    ("placeholder", "{messages}")
])

# Enhanced model initialization with retry logic
def create_model(temperature=Config.TEMPERATURE, max_tokens=Config.MAX_TOKENS):
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("Azure_endpoint"),
        api_version="2024-12-01-preview",
        model="gpt-4o-mini",
        streaming=True,
        temperature=temperature,
        max_tokens=max_tokens,
        azure_deployment="gpt-4o-mini",
    )

model = create_model()
agent_executor = create_react_agent(model=model, tools=tools, prompt=prompt)

# Enhanced state classes
class ExecutionStatus(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"

class StepResult(BaseModel):
    step: str
    result: str
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None

class PlanAndExecuteState(BaseModel):
    input_query: str = Field(..., description="The input query for the agent to process.")
    plan: List[str] = Field(default_factory=list, description="The plan created by the agent.")
    current_step_index: int = Field(default=0, description="Index of current step being executed")
    past_steps: List[StepResult] = Field(default_factory=list, description="History of executed steps")
    response: Optional[str] = Field(None, description="The final response")
    status: ExecutionStatus = Field(default=ExecutionStatus.PLANNING)
    retry_count: int = Field(default=0, description="Number of retries for current step")
    errors: List[str] = Field(default_factory=list, description="Accumulated errors")
    
    @field_validator('plan')
    @classmethod
    def validate_plan_length(cls, v):
        if len(v) > Config.MAX_PLAN_LENGTH:
            raise ValueError(f"Plan too long. Maximum {Config.MAX_PLAN_LENGTH} steps allowed.")
        return v

class Plan(BaseModel):
    steps: List[str] = Field(..., description="The steps to be executed by the agent.")
    
    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v):
        if not v:
            raise ValueError("Plan must contain at least one step")
        return v

class Response(BaseModel):
    response: str = Field(..., description="The response from the agent after executing the steps.")
    confidence: float = Field(default=1.0, description="Confidence in the response (0-1)")

class Action(BaseModel):
    action: Union[Response, Plan] = Field(..., description="The action to be performed")
    reasoning: str = Field(..., description="Reasoning behind the action")

# Enhanced prompts
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert planner. For a given task, create a clear, numbered plan.
    
    Guidelines:
    1. Break down complex problems into simple, executable steps
    2. Each step should be self-contained with all necessary information
    3. Consider edge cases and potential failures
    4. The final step should produce the desired answer
    5. Limit plans to {max_steps} steps maximum
    
    Format each step clearly and number them sequentially.""".format(max_steps=Config.MAX_PLAN_LENGTH)),
    ("placeholder", "{messages}"),
])

replanner_prompt = ChatPromptTemplate.from_messages([
    ("human", """Based on the execution so far, decide whether to:
    1. Return a final response if the goal is achieved
    2. Create an updated plan if more steps are needed
    
    Original objective: {input_query}
    
    Original plan:
    {plan}
    
    Completed steps:
    {past_steps}
    
    Errors encountered: {errors}
    
    Analyze the progress and decide the best action. If replanning, incorporate lessons learned.""")
])

# Enhanced planners with structured output
planner = planner_prompt | create_model(temperature=0.5).with_structured_output(Plan)
replanner = replanner_prompt | create_model(temperature=0.5).with_structured_output(Action)

# Enhanced execution function with error handling and retries
async def execute_plan(state: PlanAndExecuteState) -> dict:
    """Execute the current step with error handling and timeout."""
    if state.current_step_index >= len(state.plan):
        return {"status": ExecutionStatus.REPLANNING}
    
    current_step = state.plan[state.current_step_index]
    
    # Format the task with context
    steps_context = "\n".join([f"{i+1}. {step}" for i, step in enumerate(state.plan)])
    previous_results = "\n".join([
        f"Step {i+1} result: {step.result}" 
        for i, step in enumerate(state.past_steps) 
        if step.success
    ])
    
    task_prompt = f"""You are executing step {state.current_step_index + 1} of the following plan:

{steps_context}

Previous successful results:
{previous_results}

Current task: {current_step}

Execute this step carefully and provide a clear result."""

    try:
        # Execute with timeout
        agent_response = await asyncio.wait_for(
            agent_executor.ainvoke({"messages": [("user", task_prompt)]}),
            timeout=Config.EXECUTION_TIMEOUT
        )
        
        result_content = agent_response["messages"][-1].content
        
        # Create step result
        step_result = StepResult(
            step=current_step,
            result=result_content,
            success=True
        )
        
        return {
            "past_steps": state.past_steps + [step_result],
            "current_step_index": state.current_step_index + 1,
            "retry_count": 0
        }
        
    except asyncio.TimeoutError:
        error_msg = f"Step execution timed out after {Config.EXECUTION_TIMEOUT} seconds"
        return handle_execution_error(state, error_msg)
    except Exception as e:
        error_msg = f"Step execution failed: {str(e)}"
        return handle_execution_error(state, error_msg)

def handle_execution_error(state: PlanAndExecuteState, error_msg: str) -> dict:
    """Handle execution errors with retry logic."""
    step_result = StepResult(
        step=state.plan[state.current_step_index],
        result="",
        success=False,
        error=error_msg
    )
    
    if state.retry_count < Config.MAX_RETRIES:
        return {
            "past_steps": state.past_steps + [step_result],
            "retry_count": state.retry_count + 1,
            "errors": state.errors + [error_msg]
        }
    else:
        return {
            "past_steps": state.past_steps + [step_result],
            "current_step_index": state.current_step_index + 1,
            "retry_count": 0,
            "errors": state.errors + [error_msg],
            "status": ExecutionStatus.REPLANNING
        }

# Enhanced planning function
async def plan_step(state: PlanAndExecuteState) -> dict:
    """Create initial plan with validation."""
    try:
        plan = await planner.ainvoke(
            {"messages": [("user", state.input_query)]}
        )
        
        return {
            "plan": plan.steps,
            "status": ExecutionStatus.EXECUTING
        }
    except Exception as e:
        return {
            "errors": state.errors + [f"Planning failed: {str(e)}"],
            "status": ExecutionStatus.FAILED
        }

# Enhanced replanning function
async def replan_step(state: PlanAndExecuteState) -> dict:
    """Replan or finalize based on progress."""
    # Format past steps for context
    past_steps_formatted = "\n".join([
        f"{i+1}. {step.step} -> {'Success' if step.success else 'Failed'}: {step.result[:100]}..."
        for i, step in enumerate(state.past_steps)
    ])
    
    errors_formatted = "\n".join(state.errors) if state.errors else "None"
    
    try:
        output = await replanner.ainvoke({
            "input_query": state.input_query,
            "plan": "\n".join([f"{i+1}. {step}" for i, step in enumerate(state.plan)]),
            "past_steps": past_steps_formatted,
            "errors": errors_formatted
        })
        
        if isinstance(output.action, Response):
            return {
                "response": output.action.response,
                "status": ExecutionStatus.COMPLETED
            }
        else:
            # Validate new plan doesn't exceed limits
            remaining_steps = Config.MAX_PLAN_LENGTH - len(state.past_steps)
            new_steps = output.action.steps[:remaining_steps]
            
            return {
                "plan": new_steps,
                "current_step_index": 0,
                "status": ExecutionStatus.EXECUTING
            }
            
    except Exception as e:
        # If replanning fails, try to provide best effort response
        successful_steps = [s for s in state.past_steps if s.success]
        if successful_steps:
            combined_results = "\n".join([s.result for s in successful_steps])
            return {
                "response": f"Based on completed steps:\n{combined_results}",
                "status": ExecutionStatus.COMPLETED
            }
        else:
            return {
                "response": "Unable to complete the task due to errors.",
                "status": ExecutionStatus.FAILED
            }

# Enhanced routing logic
def should_end(state: PlanAndExecuteState) -> str:
    """Determine next step based on state."""
    if state.status == ExecutionStatus.COMPLETED:
        return END
    elif state.status == ExecutionStatus.FAILED:
        return END
    elif state.status == ExecutionStatus.REPLANNING:
        return "replan_step"
    elif state.current_step_index >= len(state.plan):
        return "replan_step"
    else:
        return "execute_plan"

# Build enhanced workflow
workflow = StateGraph(PlanAndExecuteState)
workflow.add_node("plan_step", plan_step)
workflow.add_node("execute_plan", execute_plan)
workflow.add_node("replan_step", replan_step)

workflow.add_edge(START, "plan_step")
workflow.add_edge("plan_step", "execute_plan")
# Complete the workflow setup
workflow.add_conditional_edges("execute_plan", should_end, {
    "execute_plan": "execute_plan",
    "replan_step": "replan_step",
    END: END
})
workflow.add_conditional_edges("replan_step", should_end, {
    "execute_plan": "execute_plan",
    END: END
})

# Compile the graph
graph = workflow.compile()

# Enhanced configuration with logging
config = {
    "recursion_limit": 50,
    "configurable": {
        "thread_id": "plan_and_execute_agent"
    }
}

# Utility functions for better output formatting
def format_step_result(step_result: StepResult) -> str:
    """Format a step result for display."""
    status = "✓" if step_result.success else "✗"
    error_info = f" (Error: {step_result.error})" if step_result.error else ""
    return f"{status} {step_result.step}: {step_result.result[:200]}...{error_info}"

def print_execution_summary(state: PlanAndExecuteState):
    """Print a summary of the execution."""
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Query: {state.input_query}")
    print(f"Status: {state.status.value}")
    print(f"\nPlan ({len(state.plan)} steps):")
    for i, step in enumerate(state.plan):
        print(f"  {i+1}. {step}")
    
    print(f"\nExecuted Steps ({len(state.past_steps)}):")
    for i, step_result in enumerate(state.past_steps):
        print(f"  {i+1}. {format_step_result(step_result)}")
    
    if state.errors:
        print(f"\nErrors ({len(state.errors)}):")
        for error in state.errors:
            print(f"  - {error}")
    
    if state.response:
        print(f"\nFinal Response:")
        print(f"  {state.response}")
    print("="*80 + "\n")

# Enhanced async run function with multiple examples
async def run_single_query(query: str, verbose: bool = True):
    """Run a single query through the plan and execute agent."""
    print(f"\n🚀 Starting execution for: {query}\n")
    
    inputs = {"input_query": query}
    final_state = None
    
    try:
        # Track the execution through the graph
        async for event in graph.astream(inputs, config=config):
            for node_name, node_output in event.items():
                if verbose:
                    print(f"📍 Node: {node_name}")
                    if node_name == "plan_step" and "plan" in node_output:
                        print(f"   Plan created with {len(node_output.get('plan', []))} steps")
                    elif node_name == "execute_plan" and "current_step_index" in node_output:
                        current_step = node_output.get('current_step_index', 0)
                        print(f"   Executing step {current_step}")
                    elif node_name == "replan_step":
                        print(f"   Replanning based on progress...")
                    print("   ---")
                
                # The node_output IS the updated state from LangGraph
                if isinstance(node_output, PlanAndExecuteState):
                    final_state = node_output
        
        if final_state:
            print_execution_summary(final_state)
            return final_state
        else:
            print("❌ No final state received from graph execution")
            return None
        
    except Exception as e:
        print(f"❌ Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Batch execution function
async def run_batch_queries(queries: List[str]):
    """Run multiple queries in batch."""
    results = []
    for query in queries:
        result = await run_single_query(query, verbose=True)
        results.append(result)
        await asyncio.sleep(1)  # Small delay between queries
    
    # Print batch summary
    print("\n" + "="*80)
    print("BATCH EXECUTION SUMMARY")
    print("="*80)
    successful = sum(1 for r in results if r and r.status == ExecutionStatus.COMPLETED)
    print(f"Total queries: {len(queries)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(queries) - successful}")
    print("="*80)
    
    return results

# Error recovery decorator
def with_error_recovery(max_retries=3):
    """Decorator for functions that need error recovery."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"❌ Failed after {max_retries} attempts: {str(e)}")
                        raise
                    print(f"⚠️ Attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            return None
        return wrapper
    return decorator

# Enhanced main execution
@with_error_recovery(max_retries=2)
async def run():
    """Main execution function with multiple test cases."""
    
    # Test cases with varying complexity
    test_queries = [
        # Original math problem
        "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace weighs. What are their combined weights in pounds?",
        
        # Current events query (tests web search)
        "What are the latest developments in AI regulation as of 2024?",
        
        # Multi-step research query
        "Compare the GDP per capita of the top 3 economies in the world and explain the factors contributing to the differences.",
        
        # Complex calculation
        "If a company's revenue grew by 15% annually for 3 years starting from $1 million, and their expenses were 60% of revenue each year, what was their total profit over the 3 years?",
    ]
    
    # You can run a single query or batch
    choice = input("Run single query (s) or batch (b)? [s/b]: ").lower()
    
    if choice == 'b':
        await run_batch_queries(test_queries)
    else:
        # Run single query
        query = input("Enter your query (or press Enter for default): ").strip()
        if not query:
            query = test_queries[0]
        
        await run_single_query(query)

# Additional utility functions
def save_execution_log(state: PlanAndExecuteState, filename: str = "execution_log.json"):
    """Save execution state to a JSON file for analysis."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": state.input_query,
        "status": state.status.value,
        "plan": state.plan,
        "steps_executed": len(state.past_steps),
        "successful_steps": sum(1 for s in state.past_steps if s.success),
        "errors": state.errors,
        "response": state.response
    }
    
    try:
        with open(filename, 'a') as f:
            json.dump(log_data, f)
            f.write('\n')
        print(f"📝 Execution log saved to {filename}")
    except Exception as e:
        print(f"⚠️ Failed to save log: {e}")

# Monitoring functions
class ExecutionMonitor:
    """Monitor and analyze execution patterns."""
    
    def __init__(self):
        self.executions = []
    
    def record_execution(self, state: PlanAndExecuteState):
        """Record an execution for analysis."""
        self.executions.append({
            "timestamp": datetime.now(),
            "query_length": len(state.input_query),
            "plan_length": len(state.plan),
            "steps_executed": len(state.past_steps),
            "success_rate": sum(1 for s in state.past_steps if s.success) / len(state.past_steps) if state.past_steps else 0,
            "had_errors": len(state.errors) > 0,
            "final_status": state.status.value
        })
    
    def print_statistics(self):
        """Print execution statistics."""
        if not self.executions:
            print("No executions recorded.")
            return
        
        print("\n" + "="*80)
        print("EXECUTION STATISTICS")
        print("="*80)
        print(f"Total executions: {len(self.executions)}")
        
        avg_plan_length = sum(e["plan_length"] for e in self.executions) / len(self.executions)
        avg_success_rate = sum(e["success_rate"] for e in self.executions) / len(self.executions)
        error_rate = sum(1 for e in self.executions if e["had_errors"]) / len(self.executions)
        
        print(f"Average plan length: {avg_plan_length:.1f} steps")
        print(f"Average success rate: {avg_success_rate:.1%}")
        print(f"Error rate: {error_rate:.1%}")
        
        status_counts = {}
        for e in self.executions:
            status = e["final_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nFinal status distribution:")
        for status, count in status_counts.items():
            print(f"  {status}: {count} ({count/len(self.executions):.1%})")
        print("="*80)

# Create global monitor instance
monitor = ExecutionMonitor()

# Enhanced run function with monitoring
async def run_with_monitoring():
    """Run queries with execution monitoring."""
    query = "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace weighs. What are their combined weights in pounds?"
    
    state = await run_single_query(query)
    if state:
        monitor.record_execution(state)
        save_execution_log(state)
    
    monitor.print_statistics()

if __name__ == "__main__":
    # Run the agent
    asyncio.run(run())