from openai import AsyncOpenAI
import chainlit as cl
import json, os, asyncio, logging
from dotenv import load_dotenv
import sys
import traceback
from datetime import datetime
from mcp import ClientSession
import re

# Configure logging
def setup_logging():
    """Configure logging with both console and file handlers"""
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if function called multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logging
logger = setup_logging()

# Load environment variables
load_dotenv()
api_key = os.getenv("openai_key")

# Validate API key
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your .env file")

if api_key == "openai_key" or not api_key.startswith("sk-"):
    logger.error("Invalid OpenAI API key format detected")
    raise ValueError("Invalid OpenAI API key. Please check your .env file contains a valid API key starting with 'sk-'")

logger.info(f"API key loaded successfully: {api_key[:10]}...")
logger.info("Initializing OpenAI client...")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=api_key)

# Instrument the OpenAI client
cl.instrument_openai()

# OpenAI settings
settings = {
    "model": "gpt-4-turbo",  # Upgraded from gpt-4 for larger context (128k tokens)
    "temperature": 0,
    "max_tokens": 1000,
}

logger.info(f"Application starting with model: {settings['model']}")

def trim_conversation_history(history, max_messages=20):
    """
    Trim conversation history to prevent context overflow.
    Keeps system message + recent messages within token budget.
    """
    if len(history) <= max_messages:
        return history
    
    # Always keep the system message (first message)
    system_msg = history[0] if history and history[0].get("role") == "system" else None
    recent_messages = history[-(max_messages-1):] if system_msg else history[-max_messages:]
    
    trimmed = ([system_msg] if system_msg else []) + recent_messages
    logger.info(f"Trimmed conversation history from {len(history)} to {len(trimmed)} messages")
    return trimmed

def summarize_large_tool_result(content, max_length=500):
    """
    Summarize large tool results to reduce context usage.
    """
    if len(content) <= max_length:
        return content
    
    # For screenshots, just return a summary
    if "screenshot" in content.lower() or "image" in content.lower():
        return "Screenshot has been captured and displayed to the user."
    
    # For other large results, truncate with summary
    truncated = content[:max_length]
    return f"{truncated}... [Content truncated - original length: {len(content)} chars]"

@cl.on_chat_start
async def start_chat():
    """Initialize chat session with message history"""
    logger.info("New chat session started")
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are an expert forex trader"}],
    )
    # Initialize empty MCP tools dictionary
    cl.user_session.set("mcp_tools", {})

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    logger.info(f"MCP connection established: {connection.name}")
    
    await cl.Message(
        content=f"Establishing connection with MCP server: `{connection.name}`..."
    ).send()
    
    try:
        # List available tools
        result = await session.list_tools()
        
        if result and hasattr(result, "tools") and result.tools:
            # Process tool metadata in OpenAI function format
            tools_for_llm = []
            for tool in result.tools:
                tools_for_llm.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })
            
            # Store tools for later use
            mcp_tools = cl.user_session.get("mcp_tools", {})
            mcp_tools[connection.name] = tools_for_llm
            cl.user_session.set("mcp_tools", mcp_tools)
            
            tool_names = [tool.name for tool in result.tools]
            
            await cl.Message(
                content=f"**Tools available from `{connection.name}`:**\n{', '.join(tool_names)}"
            ).send()
            
            logger.info(f"Loaded {len(tools_for_llm)} tools from {connection.name}: {tool_names}")
            
            # Auto-execute run-me-first if available
            if "run-me-first" in tool_names:
                try:
                    logger.info("Auto-executing 'run-me-first' tool")
                    auto_result = await session.call_tool("run-me-first", {})
                    
                    if isinstance(auto_result, dict):
                        result_text = json.dumps(auto_result)
                    elif hasattr(auto_result, "content") and auto_result.content:
                        result_text = "".join(
                            getattr(block, "text", "") for block in auto_result.content
                        )
                    else:
                        result_text = str(auto_result)
                    
                    await cl.Message(
                        content=f"**Setup Complete**: {result_text}"
                    ).send()
                    logger.info("Successfully auto-executed 'run-me-first'")
                    
                except Exception as e:
                    logger.error(f"Failed to auto-execute 'run-me-first': {str(e)}")
                    await cl.Message(
                        content=f"⚠️ Setup tool failed to run automatically: {str(e)}"
                    ).send()
        
    except Exception as e:
        logger.error(f"Error connecting to MCP {connection.name}: {str(e)}", exc_info=True)
        await cl.Message(
            content=f"⚠️ Failed to connect to MCP server {connection.name}: {str(e)}"
        ).send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    logger.info(f"MCP connection disconnected: {name}")
    
    await cl.Message(f"MCP connection `{name}` has been disconnected.").send()
    
    # Remove tools from session
    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        removed_tools = len(mcp_tools[name])
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)
        logger.info(f"Removed {removed_tools} tools from disconnected MCP: {name}")

async def _handle_generic_tool_result(result, tool_step):
    """Handle generic tool result processing"""
    if isinstance(result, dict):
        tool_result = json.dumps(result)
    elif hasattr(result, "content") and result.content:
        tool_result = "".join(getattr(block, "text", "") for block in result.content)
    else:
        tool_result = str(result)

    tool_step.output = tool_result
    return tool_result, tool_result

async def _handle_screenshot_result(result, tool_step):
    """Handle browser screenshot/snapshot result processing and UI display"""
    logger.info("Processing screenshot result")
    
    # Parse the result to get screenshot data
    screenshot_data = None
    tool_result_text = "Screenshot captured successfully"
    screenshot_path = None
    
    if isinstance(result, dict):
        screenshot_data = result
    elif hasattr(result, "content") and result.content:
        full_text = "".join(getattr(block, "text", "") for block in result.content)
        
        # First try to parse as JSON for backward compatibility
        try:
            screenshot_data = json.loads(full_text)
        except json.JSONDecodeError:
            # Not JSON, try to extract path from markdown format
            logger.info("Result is not JSON, attempting to extract path from markdown text")
            
            # Try to extract path from JavaScript code using regex
            # Look for path: '/some/path/file.extension'
            path_match = re.search(r"path:\s*['\"]([^'\"]+)['\"]", full_text)
            if path_match:
                screenshot_path = path_match.group(1)
                logger.info(f"Extracted screenshot path from markdown: {screenshot_path}")
            else:
                # Fallback: try to extract from comment line
                comment_match = re.search(r"//.*save it as\s+([^\s]+)", full_text)
                if comment_match:
                    screenshot_path = comment_match.group(1)
                    logger.info(f"Extracted screenshot path from comment: {screenshot_path}")
                else:
                    logger.warning(f"Could not extract screenshot path from text: {full_text}")
            
            screenshot_data = {"extracted_path": screenshot_path} if screenshot_path else {"error": "No path found"}
    else:
        screenshot_data = {"raw_result": str(result)}

    # Set concise tool step output to avoid context bloat
    tool_step.output = "Screenshot processing completed"

    # Check if we have a valid screenshot path
    if not screenshot_path and isinstance(screenshot_data, dict):
        # Look for common screenshot path keys in JSON format
        screenshot_path = (
            screenshot_data.get("filename") or 
            screenshot_data.get("path") or 
            screenshot_data.get("screenshot_path") or
            screenshot_data.get("file_path") or
            screenshot_data.get("extracted_path")
        )
    
    # If we have a screenshot path, display it
    if screenshot_path and os.path.exists(screenshot_path):
        try:
            image = cl.Image(
                path=screenshot_path, 
                name="screenshot", 
                display="inline"
            )
            
            await cl.Message(
                content="📸 Screenshot captured:",
                elements=[image],
            ).send()
            
            tool_result_text = "Screenshot has been captured and displayed to the user."
            logger.info(f"Screenshot displayed successfully from path: {screenshot_path}")
            
        except Exception as e:
            logger.error(f"Failed to display screenshot: {str(e)}")
            await cl.Message(
                content=f"⚠️ Screenshot was captured but failed to display: {str(e)}"
            ).send()
            tool_result_text = f"Screenshot captured but display failed: {str(e)}"
    else:
        # No valid screenshot path found or file doesn't exist
        if screenshot_path:
            logger.warning(f"Screenshot path found but file doesn't exist: {screenshot_path}")
            await cl.Message(
                content=f"⚠️ Screenshot was saved to {screenshot_path} but the file is not accessible."
            ).send()
            tool_result_text = f"Screenshot saved to {screenshot_path} but file not accessible."
        else:
            logger.warning("Screenshot result did not contain a valid file path")
            await cl.Message(
                content="⚠️ Screenshot command executed but no image file path was found in the result."
            ).send()
            tool_result_text = "Screenshot command executed but no image file path was returned."

    # Return concise summary for conversation history
    return tool_result_text, tool_result_text

async def execute_mcp_tool(tool_call, all_mcp_tools):
    """
    Execute any MCP tool with proper error handling and UI feedback.
    Returns (tool_result_text, tool_step_output)
    """
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    
    async with cl.Step(name=tool_name, type="tool", show_input="json") as tool_step:
        tool_step.input = tool_args
        
        try:
            logger.info(f"Executing MCP tool: {tool_name}")
            
            # Find which MCP connection has this tool
            mcp_connection_name = next(
                (
                    conn_name
                    for conn_name, tools in all_mcp_tools.items()
                    if any(t["function"]["name"] == tool_name for t in tools)
                ),
                None,
            )
            
            if not mcp_connection_name:
                raise Exception(f"Tool '{tool_name}' not found in any MCP connection")
            
            # Get the MCP session
            mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_connection_name)
            if not mcp_session:
                raise Exception(f"MCP session for '{mcp_connection_name}' not found")
            
            # Set timeout based on tool type
            timeout_seconds = 60.0
            
            # Execute the tool with timeout
            tool_task = mcp_session.call_tool(tool_name, tool_args)
            result = await asyncio.wait_for(tool_task, timeout=timeout_seconds)
            
            # Process the result based on tool type
            if tool_name in ["browser_take_screenshot", "browser_snapshot"]:
                return await _handle_screenshot_result(result, tool_step)
            else:
                return await _handle_generic_tool_result(result, tool_step)
            
        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out after {timeout_seconds} seconds"
            logger.error(error_msg)
            tool_step.error = error_msg
            return error_msg, error_msg
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            tool_step.error = error_msg
            return error_msg, error_msg

async def continue_conversation(history, aggregated_tools, max_iterations=5):
    """
    Continue the conversation with tool calling support, handling multiple iterations
    """
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Starting conversation iteration {iteration}")
        
        try:
            # Trim history to prevent context overflow
            trimmed_history = trim_conversation_history(history)
            
            # Get AI response
            async with cl.Step(
                name=f"Thinking (iteration {iteration})", type="llm"
            ) as step:
                response = await client.chat.completions.create(
                    model=settings["model"],
                    messages=trimmed_history,
                    tools=aggregated_tools if aggregated_tools else None,
                    tool_choice="auto" if aggregated_tools else None,
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"]
                )
                response_message = response.choices[0].message
                step.output = response_message
            
            if response_message.tool_calls:
                logger.info(f"Processing {len(response_message.tool_calls)} tool call(s)")
                
                # Add assistant message with tool calls to history
                history.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in response_message.tool_calls
                    ],
                })
                
                # Execute all tool calls
                all_mcp_tools = cl.user_session.get("mcp_tools", {})
                for tool_call in response_message.tool_calls:
                    tool_result_text, _ = await execute_mcp_tool(tool_call, all_mcp_tools)
                    
                    # Summarize large tool results before adding to history
                    summarized_result = summarize_large_tool_result(str(tool_result_text))
                    
                    # Add tool result to history
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": summarized_result,
                    })
                
                cl.user_session.set("message_history", history)
                continue  # Continue the conversation loop
                
            else:
                # No more tool calls, provide final response
                if response_message.content:
                    await cl.Message(content=response_message.content).send()
                    history.append({"role": "assistant", "content": response_message.content})
                    cl.user_session.set("message_history", history)
                    logger.info("Conversation completed successfully")
                else:
                    # No content, send default completion message
                    default_msg = "I've completed the requested actions."
                    await cl.Message(content=default_msg).send()
                    history.append({"role": "assistant", "content": default_msg})
                    cl.user_session.set("message_history", history)
                    logger.info("Conversation completed with default message")
                break  # Exit the conversation loop
                
        except Exception as e:
            logger.error(f"Error in conversation iteration {iteration}: {e}", exc_info=True)
            await cl.Message(
                content=f"An error occurred during tool execution: {e}"
            ).send()
            break
    
    if iteration >= max_iterations:
        logger.warning(f"Reached maximum iterations ({max_iterations})")
        await cl.Message(
            content="I've reached the maximum number of tool calls for this request. Please try asking your question again if you need more assistance."
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with proper logging and error handling"""
    try:
        # Log incoming message (be careful with PII)
        logger.info(f"Received message from user (length: {len(message.content)} chars)")
        start_time = datetime.now()
        
        # Get message history from session
        history = cl.user_session.get("message_history")
        history.append({"role": "user", "content": message.content})
        
        async with cl.Step(name="Thinking", type="llm") as thinking_step:
            thinking_step.input = message.content
            
            # Aggregate all available tools
            all_mcp_tools = cl.user_session.get("mcp_tools", {})
            aggregated_tools = [
                tool for conn_tools in all_mcp_tools.values() for tool in conn_tools
            ]
            
            logger.info(f"Available tools: {len(aggregated_tools)} from {len(all_mcp_tools)} connections")
            
            # Trim history to prevent context overflow
            trimmed_history = trim_conversation_history(history)
            
            # Make initial API call
            response = await client.chat.completions.create(
                model=settings["model"],
                messages=trimmed_history,
                tools=aggregated_tools if aggregated_tools else None,
                tool_choice="auto" if aggregated_tools else None,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"]
            )
            response_message = response.choices[0].message
            thinking_step.output = response_message
        
        if response_message.tool_calls:
            logger.info(f"Initial response includes {len(response_message.tool_calls)} tool call(s)")
            
            # Add assistant message with tool calls to history
            history.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in response_message.tool_calls
                ],
            })
            
            cl.user_session.set("message_history", history)
            
            # Execute all tool calls
            for tool_call in response_message.tool_calls:
                tool_result_text, _ = await execute_mcp_tool(tool_call, all_mcp_tools)
                
                # Summarize large tool results before adding to history
                summarized_result = summarize_large_tool_result(str(tool_result_text))
                
                # Add tool result to history
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": summarized_result,
                })
                
                cl.user_session.set("message_history", history)
            
            # Continue conversation to get final response
            await continue_conversation(history, aggregated_tools)
            
        else:
            # No tool calls, just send the response
            if response_message.content:
                await cl.Message(content=response_message.content).send()
                history.append({"role": "assistant", "content": response_message.content})
                cl.user_session.set("message_history", history)
                logger.info("Sent direct response without tools")
        
        # Log successful completion
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Message processed successfully (duration: {duration:.2f}s)")
        
    except Exception as e:
        # Log the error
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        
        # Send user-friendly error message
        error_message = "I'm sorry, I encountered an error processing your request. Please try again."
        await cl.Message(content=error_message).send()

@cl.on_chat_end
async def on_chat_end():
    """Log when a chat session ends"""
    logger.info("Chat session ended")