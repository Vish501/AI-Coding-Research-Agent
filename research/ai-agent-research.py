# Standard library imports
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict, Sequence

# Environment variable loading
from dotenv import load_dotenv

# LangChain core message types
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# LangGraph workflow components
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# LLM provider (Ollama integration)
from langchain_ollama import ChatOllama

# MCP (Model Context Protocol) client tools & connectivity
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# Tool decorator for exposing functions to LangChain
from langchain.tools import tool

# PDF generation (ReportLab)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

# Project utility imports
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

@tool
async def save_last_message_to_pdf(content: str, filename: str = None) -> None:
    """
    Saves text content to a PDF file.
    
    Args:
    - content: The text content to save
    - filename: Optional custom filename (without extension)
    
    Returns:
    - Success message with file path

    Notes
    - Automatically creates parent directories.
    - Converts Windows Path objects to strings for ReportLab compatibility.
    - Handles multi-line text safely by splitting into Paragraphs.
    """
    # Create timestamp for the filename (UTC-friendly)
    utc_timestamp = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")

    # Directory where reports will be stored
    save_dir = Path(working_directory) / "research" / "reports"

    # Generate a descriptive file name
    if filename:
        save_path = save_dir / f"{filename}.pdf"
    else:
        save_path = save_dir / f"ollama_response - {utc_timestamp}.pdf"

    # Ensure directory exists, else create it
    create_directories([save_path.parent], verbose=False)

    try:
        # Create PDF document
        doc = SimpleDocTemplate(str(save_path), pagesize=letter)
        styles = getSampleStyleSheet()

        # Split text into paragraphs
        story = []
        for line in content.split("\n"):
            clean_line = line.strip()
            if clean_line:
                story.append(Paragraph(clean_line, styles["Normal"]))

        if not story:
            raise ValueError("No valid text content to write to PDF.")

        # Write PDF to disk
        doc.build(story)

        logger.info(f"PDF successfully saved: {save_path}")
        
    except Exception as e:
        logger.error(f"Unable to save PDF file: {e}")

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
    print("\n🚀 Starting Coding Research Agent...")
    print("=" * 60)
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Load tools
                mcp_tools = await load_mcp_tools(session)
                custom_tools = [save_last_message_to_pdf]
                tools = mcp_tools + custom_tools

                # Create agent with tools
                agent = create_research_agent(tools)

                # Initialize conversation state
                messages = [
                    SystemMessage(
                        content=("You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user.",
                        ))
                ]

                # Print available tools to the user
                print("Available Tools:", ", ".join([tool.name for tool in tools]))
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
                        final_message = None

                        async for event in agent.astream({"messages": messages}, stream_mode="values"):
                            if "messages" in event:
                                final_message = event["messages"]
                                last_msg = final_message[-1]
                                if isinstance(last_msg, AIMessage):
                                    if last_msg.content and not last_msg.tool_calls:
                                        print(last_msg.content, flush=True)
                        
                        # Update messages with final state
                        if final_message:
                            messages = final_message

                    except KeyboardInterrupt as ki:
                        logger.error(f"Error during agent execution: {ki}")
                        print("\n\nInterrupted. Goodbye!")
                        break

                    except Exception as e:
                        print(f"Error: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print(f"\n Failed to start agent: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Node.js and npx are installed")
        print("2. Verify FIRECRAWL_API_KEY in .env file")
        print("3. Check internet connectivity")
        print("4. Try: npm install -g @mendable/firecrawl-mcp")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown requested.")
    