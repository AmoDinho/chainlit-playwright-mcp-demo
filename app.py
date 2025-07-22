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
        message_history.append({"role": "tool", "content": json.dumps(all_tools)})

        # Create message for streaming response
        msg = cl.Message(content="")
        
        # Make streaming API call
        stream = await client.chat.completions.create(
            messages=message_history, 
            stream=True, 
            tools=all_tools,
            **settings
        )
        
        # Stream the response
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)
        
        # Update message history with assistant response
        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()
        
        # Log successful API call
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"OpenAI API call successful (duration: {duration:.2f}s)")
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
    # Your connection initialization code here
    # This handler is required for MCP to work
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

@cl.step(type="tool") 
async def call_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input
    
    # Find appropriate MCP connection for this tool
    mcp_name = find_mcp_for_tool(tool_name)
    
    # Get the MCP session
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    
    # Call the tool
    result = await mcp_session.call_tool(tool_name, tool_input)
    
    return result
    
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    # Your cleanup code here
    # This handler is optional