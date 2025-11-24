from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from dotenv import load_dotenv

import asyncio
import os

from CodeResearcher.utils.common import create_directories
from CodeResearcher.utils.logger import get_logger


# Initialize application-wide logger
logger = get_logger()

# Loading environment variables from .env
load_dotenv()

# Changing directory to main directory for easy data access
working_directory = os.getenv("WORKING_DIRECTORY")
if working_directory:
    os.chdir(working_directory)

# Initalize LLM `qwen2.5:14b`
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"), temperature=0)

# Create a set of parameters for starting an MCP server using standard input/output
server_params = StdioServerParameters(
    # Create executable using - `npx`, or the Node package runner included with Node.js.
    command="npx",

    # Environment variables passed to the launched process.
    env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},

    # CLI passed to `npx`, which tells it ot run the `firecrawl-mcp` package as an MCP server
    args=["firecrawl-mcp"]
)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State for the agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =============================================================================
# GRAPH NODES
# =============================================================================

def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end the conversation."""
    messages = state["messages"]
    last_message = messages[-1]

    # If LLM makes a tool call, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return END

async def call_model(state: AgentState) -> dict:
    """Call the LLM with the current state."""
    messages = state["messages"]
    response = await llm.ainvoke(messages)
    return {"agent": [response]}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_research_agent(tools):
    """Create a LangGraph StateGraph for the agent."""

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create graph
    workflow = StateGraph(AgentState)

    # Define nodes
    async def agent_node(state: AgentState) -> dict:
        """Agent node that calls the LLM."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # Entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    # After tools, always go to the agent
    workflow.add_edge("tools", "agent")

    # Compile and return graph
    return workflow.compile()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Load tools and create agent
            tools = await load_mcp_tools(session)
            agent = create_research_agent(tools)

            # Initialize conversation state
            messages = [
                SystemMessage(
                    content=("You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user.",
                    ))
            ]

            # Print available tools to the user
            print("\nAvailable Tools:", ", ".join([tool.name for tool in tools]))
            print("-" * 60)
            print("Type 'quit' to exit.")

            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("Goodbye!")
                        break

                    if not user_input:
                        print("Agent: Invalid Query.")
                        continue

                    # Add user message to state
                    messages.append(HumanMessage(content=user_input[:100_000]))

                    # Streaming output
                    print("Agent: ", end="", flush=True)

                    # Stream
                    async for event in agent.astream(
                        {"messages": messages},
                        stream_mode="values"
                    ):
                        if "messages" in event:
                            last_msg = event["messages"][-1]
                            if isinstance(last_msg, AIMessage):
                                if last_msg.content and not last_msg.tool_calls:
                                    print(last_msg.content, flush=True)
                    
                    # Update messages with final state
                    result = await agent.ainvoke({"messages": messages})
                    messages = result["messages"]

                except KeyboardInterrupt:
                    logger.error(f"Error during agent execution: {e}")
                    print("\n\n Interrupted. Goodbye!")
                    break

                except Exception as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown requested.")
