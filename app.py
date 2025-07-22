from openai import AsyncOpenAI
import chainlit as cl
import json, os, asyncio, logging
from dotenv import load_dotenv
import sys
from datetime import datetime
from mcp import ClientSession

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
    "model": "gpt-4",
    "temperature": 0,
    "max_tokens": 1000,
}

logger.info(f"Application starting with model: {settings['model']}")

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

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with proper logging and error handling"""
    try:
        # Log incoming message (be careful with PII)
        logger.info(f"Received message from user (length: {len(message.content)} chars)")
        
        # Get message history from session
        message_history = cl.user_session.get("message_history")
        message_history.append({"role": "user", "content": message.content})
        
        # Log API call attempt
        logger.info("Making OpenAI API call...")
        start_time = datetime.now()
        
        mcp_tools = cl.user_session.get("mcp_tools", {})
        all_tools = [tool for connection_tools in mcp_tools.values() for tool in connection_tools]
        
        # Convert MCP tools to OpenAI format
        openai_tools = []
        for tool in all_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema", {})
                }
            }
            openai_tools.append(openai_tool)

        # Create message for streaming response
        msg = cl.Message(content="")
        
        # Prepare API call parameters
        api_call_params = {
            "messages": message_history, 
            "stream": True,
            **settings
        }
        
        # Only add tools if we have any
        if openai_tools:
            api_call_params["tools"] = openai_tools
        
        # Make streaming API call
        stream = await client.chat.completions.create(**api_call_params)
        
        # Handle streaming response with tool calls
        tool_calls = {}
        current_tool_call_id = None
        
        async for part in stream:
            # Handle text content
            if content := part.choices[0].delta.content:
                await msg.stream_token(content)
            
            # Handle tool calls
            if tool_calls_delta := part.choices[0].delta.tool_calls:
                for tool_call in tool_calls_delta:
                    if tool_call.id:
                        # New tool call
                        current_tool_call_id = tool_call.id
                        tool_calls[current_tool_call_id] = {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name if tool_call.function.name else "",
                                "arguments": tool_call.function.arguments if tool_call.function.arguments else ""
                            }
                        }
                    else:
                        # Continue existing tool call
                        if current_tool_call_id and tool_call.function.arguments:
                            tool_calls[current_tool_call_id]["function"]["arguments"] += tool_call.function.arguments
        
        # Update message history with assistant response
        assistant_message = {"role": "assistant", "content": msg.content}
        
        # If we have tool calls, add them to the message and execute
        if tool_calls:
            assistant_message["tool_calls"] = list(tool_calls.values())
            message_history.append(assistant_message)
            
            # Execute each tool call
            for tool_call in tool_calls.values():
                try:
                    logger.info(f"Executing tool: {tool_call['function']['name']}")
                    
                    # Parse arguments
                    import json
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    
                    # Create a tool use object for the call_tool function
                    class ToolUse:
                        def __init__(self, name, input_data):
                            self.name = name
                            self.input = input_data
                    
                    tool_use = ToolUse(tool_call["function"]["name"], tool_args)
                    
                    # Call the tool
                    tool_result = await call_tool(tool_use)
                    
                    # Add tool result to message history
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_result)
                    }
                    message_history.append(tool_message)
                    
                    # Show tool result to user
                    await cl.Message(content=f"🔧 Tool '{tool_call['function']['name']}' executed successfully").send()
                    
                except Exception as e:
                    logger.error(f"Error executing tool {tool_call['function']['name']}: {str(e)}")
                    # Add error result to message history
                    error_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"Error: {str(e)}"
                    }
                    message_history.append(error_message)
            
            # Make another API call to get the final response
            logger.info("Making follow-up API call after tool execution...")
            follow_up_stream = await client.chat.completions.create(
                messages=message_history,
                stream=True,
                **{k: v for k, v in settings.items() if k != 'tools'}  # Remove tools for follow-up
            )
            
            # Stream the final response
            final_msg = cl.Message(content="")
            async for part in follow_up_stream:
                if content := part.choices[0].delta.content:
                    await final_msg.stream_token(content)
            
            message_history.append({"role": "assistant", "content": final_msg.content})
            await final_msg.update()
        else:
            # No tool calls, just add the regular response
            message_history.append(assistant_message)
        
        await msg.update()
        
        # Log successful API call
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"OpenAI API call successful (duration: {duration:.2f}s)")
        if tool_calls:
            logger.info(f"Executed {len(tool_calls)} tool(s)")
        logger.info(f"Response sent (length: {len(msg.content)} chars)")
        
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

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    logger.info(f"MCP connection established: {connection.name}")
    
    try:
        # List available tools
        result = await session.list_tools()
        
        # Process tool metadata
        tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        } for t in result.tools]
        
        # Store tools for later use
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)
        
        logger.info(f"Loaded {len(tools)} tools from {connection.name}: {[t['name'] for t in tools]}")
        
    except Exception as e:
        logger.error(f"Error connecting to MCP {connection.name}: {str(e)}", exc_info=True)

def find_mcp_for_tool(tool_name):
    """Find which MCP connection provides a specific tool"""
    mcp_tools = cl.user_session.get("mcp_tools", {})
    for connection_name, tools in mcp_tools.items():
        for tool in tools:
            if tool["name"] == tool_name:
                return connection_name
    return None

@cl.step(type="tool") 
async def call_tool(tool_use):
    """Call an MCP tool with proper error handling"""
    try:
        tool_name = tool_use.name
        tool_input = tool_use.input
        
        logger.info(f"Calling MCP tool: {tool_name}")
        print(f"Calling tool: {tool_name} with input: {tool_input}")
        
        # Find appropriate MCP connection for this tool
        mcp_name = find_mcp_for_tool(tool_name)
        if not mcp_name:
            raise ValueError(f"No MCP connection found for tool: {tool_name}")
        
        # Get the MCP session
        mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
        
        # Call the tool
        result = await mcp_session.call_tool(tool_name, tool_input)
        
        print(f"Result: {result}")
        logger.info(f"MCP tool {tool_name} executed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name}: {str(e)}", exc_info=True)
        raise
    
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    logger.info(f"MCP connection disconnected: {name}")
    
    # Remove tools from session
    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        removed_tools = len(mcp_tools[name])
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)
        logger.info(f"Removed {removed_tools} tools from disconnected MCP: {name}")