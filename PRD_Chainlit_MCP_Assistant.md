# Product Requirements Document: Chainlit MCP-Enabled AI Assistant

## 1. Product Overview

### 1.1 Objective

Build a Chainlit-based conversational AI application that integrates with the OpenAI Python SDK and dynamically utilizes tools from Model Context Protocol (MCP) servers. The application should provide a seamless chat interface where users can interact with AI assistants that can execute real-world tasks through MCP tools.

### 1.2 Key Value Propositions

- **Dynamic Tool Integration**: Automatically discover and utilize tools from connected MCP servers
- **Conversational Interface**: Natural language interaction with AI that can execute complex workflows
- **Extensible Architecture**: Support for multiple MCP servers and tool types
- **Visual Feedback**: Real-time UI updates showing tool execution progress and results
- **Session Persistence**: Maintain conversation context and tool availability across interactions

## 2. Technical Requirements

### 2.1 Core Dependencies

```python
# Required packages
chainlit >= 1.0.0
openai >= 1.0.0
mcp >= 0.1.0
python-dotenv >= 1.0.0
asyncio (built-in)
logging (built-in)
```

### 2.2 Environment Configuration

```bash
# Required environment variables
OPENAI_API_KEY=sk-...  # OpenAI API key
DEFAULT_MODEL=gpt-4    # Default OpenAI model
```

### 2.3 System Architecture

#### 2.3.1 Core Components

1. **Chainlit Application Framework**: Primary UI and session management
2. **OpenAI Integration Layer**: LLM communication and function calling
3. **MCP Integration Layer**: Dynamic tool discovery and execution
4. **Session Management**: Conversation history and tool state persistence
5. **Logging System**: Comprehensive application monitoring

## 3. Chainlit-Specific Requirements

### 3.1 Essential Decorators

#### 3.1.1 `@cl.on_chat_start`

```python
@cl.on_chat_start
async def start_chat():
    """Initialize new chat session"""
    # Initialize message history with system prompt
    cl.user_session.set("message_history", [
        {"role": "system", "content": "System prompt here"}
    ])
    # Initialize empty MCP tools registry
    cl.user_session.set("mcp_tools", {})
```

**Requirements:**

- Initialize conversation with configurable system prompt
- Set up empty MCP tools dictionary for dynamic tool discovery
- Support custom session initialization parameters

#### 3.1.2 `@cl.on_message`

```python
@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages with tool calling support"""
    # Get conversation history
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})

    # Aggregate available tools from all MCP connections
    all_mcp_tools = cl.user_session.get("mcp_tools", {})
    aggregated_tools = [tool for conn_tools in all_mcp_tools.values() for tool in conn_tools]

    # Make OpenAI API call with tools
    # Handle tool calls and continue conversation
```

**Requirements:**

- Support for both direct responses and tool-mediated responses
- Aggregate tools from multiple MCP connections
- Maintain conversation history in OpenAI format
- Handle errors gracefully with user-friendly messages

#### 3.1.3 `@cl.on_mcp_connect`

```python
@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Handle MCP server connection"""
    # Discover available tools
    result = await session.list_tools()

    # Convert to OpenAI function format
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

    # Store tools in session
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools_for_llm
    cl.user_session.set("mcp_tools", mcp_tools)
```

**Requirements:**

- Automatic tool discovery upon MCP connection
- Transform MCP tool schemas to OpenAI function format
- Store tools per connection for proper session management
- Provide user feedback about available tools
- Support auto-execution of setup tools (e.g., "run-me-first")

#### 3.1.4 `@cl.on_mcp_disconnect`

```python
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Handle MCP server disconnection"""
    # Remove tools from session
    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)

    # Notify user of disconnection
    await cl.Message(f"MCP connection `{name}` has been disconnected.").send()
```

**Requirements:**

- Clean up tools from disconnected servers
- Update user session state
- Provide user notification about connection status

#### 3.1.5 `@cl.on_chat_end`

```python
@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat session ends"""
    logger.info("Chat session ended")
```

**Requirements:**

- Perform cleanup operations
- Log session termination
- Optional: Save conversation history

### 3.2 Chainlit UI Components

#### 3.2.1 `cl.Step`

```python
async with cl.Step(name="Tool Execution", type="tool", show_input="json") as step:
    step.input = tool_arguments
    # Execute tool
    step.output = tool_result
    # Or on error:
    step.error = error_message
```

**Requirements:**

- Visual progress indication for LLM thinking
- Tool execution feedback with input/output display
- Error state visualization
- Support for nested steps in complex workflows

#### 3.2.2 `cl.Message`

```python
# Send messages to user
await cl.Message(content="Tool execution complete").send()

# Stream responses
msg = cl.Message(content="")
async for token in stream:
    await msg.stream_token(token)
await msg.update()
```

**Requirements:**

- Support for markdown formatting
- Real-time message streaming capability
- Rich content display (code blocks, lists, etc.)

#### 3.2.3 `cl.user_session`

```python
# Store session data
cl.user_session.set("key", value)

# Retrieve session data
value = cl.user_session.get("key", default_value)
```

**Requirements:**

- Persistent storage of conversation history
- MCP tools registry per session
- Custom session variables support
- Session isolation between users

## 4. OpenAI SDK Integration Requirements

### 4.1 Client Configuration

```python
# Initialize OpenAI client
client = AsyncOpenAI(api_key=api_key)

# Instrument for Chainlit integration
cl.instrument_openai()

# Configurable settings
settings = {
    "model": "gpt-4",
    "temperature": 0,
    "max_tokens": 1000,
}
```

**Requirements:**

- Async client for non-blocking operations
- Chainlit instrumentation for UI integration
- Configurable model parameters
- Environment-based configuration

### 4.2 Function Calling Integration

```python
# Convert MCP tools to OpenAI format
openai_tools = [{
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema,
    }
} for tool in mcp_tools]

# API call with tools
response = await client.chat.completions.create(
    model=settings["model"],
    messages=history,
    tools=openai_tools if openai_tools else None,
    tool_choice="auto" if openai_tools else None,
)
```

**Requirements:**

- Automatic tool schema conversion from MCP to OpenAI format
- Dynamic tool availability based on connected MCP servers
- Support for multiple tool calls in single response
- Proper tool choice handling ("auto", "none", specific tools)

### 4.3 Response Handling

```python
# Handle regular responses
if response_message.content:
    await cl.Message(content=response_message.content).send()

# Handle tool calls
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        # Execute tool and add result to history
        tool_result = await execute_mcp_tool(tool_call, all_mcp_tools)
        history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(tool_result),
        })
```

**Requirements:**

- Support for both text and tool call responses
- Proper message history management
- Tool result integration back into conversation

## 5. MCP Integration Requirements

### 5.1 Tool Discovery and Management

```python
async def on_mcp_connect(connection, session: ClientSession):
    # List available tools from MCP server
    result = await session.list_tools()

    # Process and store tools
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
```

**Requirements:**

- Automatic tool discovery upon connection
- Support for multiple MCP servers simultaneously
- Tool schema validation and conversion
- Dynamic tool availability updates

### 5.2 Tool Execution Framework

```python
async def execute_mcp_tool(tool_call, all_mcp_tools):
    """Execute MCP tool with proper error handling"""
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    async with cl.Step(name=tool_name, type="tool", show_input="json") as step:
        # Find correct MCP connection
        mcp_connection_name = find_mcp_for_tool(tool_name, all_mcp_tools)

        # Get MCP session
        mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_connection_name)

        # Execute with timeout
        result = await asyncio.wait_for(
            mcp_session.call_tool(tool_name, tool_args),
            timeout=60.0
        )

        return process_tool_result(result)
```

**Requirements:**

- Tool execution with visual feedback via `cl.Step`
- Timeout handling for long-running tools
- Error handling and user notification
- Result processing and formatting
- Connection management and tool routing

### 5.3 Tool Result Processing

```python
def process_tool_result(result):
    """Process MCP tool result into string format"""
    if isinstance(result, dict):
        return json.dumps(result)
    elif hasattr(result, "content") and result.content:
        return "".join(getattr(block, "text", "") for block in result.content)
    else:
        return str(result)
```

**Requirements:**

- Handle various MCP result formats
- Convert results to string for OpenAI message history
- Preserve structured data when possible
- Error result handling

## 6. Conversation Continuation Requirements

### 6.1 Multi-Turn Tool Interactions

```python
async def continue_conversation(history, aggregated_tools, max_iterations=5):
    """Handle multi-turn tool calling conversations"""
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Make API call
        response = await client.chat.completions.create(
            model=settings["model"],
            messages=history,
            tools=aggregated_tools,
            tool_choice="auto"
        )

        if response.choices[0].message.tool_calls:
            # Execute tools and continue
            # Add results to history
            continue
        else:
            # Final response - break loop
            break
```

**Requirements:**

- Support for multi-turn conversations with tool calls
- Configurable maximum iteration limits
- Proper conversation state management
- Loop termination conditions

### 6.2 Conversation State Management

```python
# Maintain conversation history in OpenAI format
history = [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
    {"role": "assistant", "content": "Final response"}
]
```

**Requirements:**

- Maintain complete conversation history
- Include tool calls and results in proper format
- Session persistence across page reloads
- History truncation for token limits

## 7. User Interface Requirements

### 7.1 Visual Feedback Systems

- **Connection Status**: Visual indicators for MCP server connections
- **Tool Availability**: Display of available tools from connected servers
- **Execution Progress**: Real-time feedback during tool execution
- **Error Handling**: Clear error messages with actionable guidance
- **Conversation Flow**: Clear distinction between AI responses and tool outputs

### 7.2 Interactive Elements

- **Chat Interface**: Standard conversational UI with message history
- **Tool Execution Display**: Expandable sections showing tool inputs/outputs
- **Connection Management**: UI for viewing and managing MCP connections
- **Settings Panel**: Configuration options for model parameters

## 8. Security and Operational Requirements

### 8.1 Security Considerations

- **API Key Management**: Secure storage and validation of OpenAI API keys
- **Input Validation**: Sanitization of user inputs and tool parameters
- **Error Handling**: Prevent information leakage through error messages
- **Session Isolation**: Ensure user sessions don't interfere with each other

### 8.2 Logging and Monitoring

```python
def setup_logging():
    """Configure comprehensive logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Console and file handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler('app.log')

    # Formatters with context
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
```

**Requirements:**

- Comprehensive logging of all operations
- Performance monitoring (API call duration, tool execution time)
- Error tracking and debugging information
- User activity logging (without PII)

### 8.3 Error Handling

- **Graceful Degradation**: Continue operation when tools fail
- **User-Friendly Messages**: Clear error communication without technical details
- **Retry Mechanisms**: Automatic retry for transient failures
- **Fallback Behaviors**: Default responses when tools are unavailable

## 9. Implementation Phases

### Phase 1: Core Framework

1. Set up Chainlit application structure
2. Implement OpenAI SDK integration
3. Basic conversation handling
4. Session management

### Phase 2: MCP Integration

1. MCP connection handling
2. Tool discovery and registration
3. Basic tool execution
4. Error handling

### Phase 3: Advanced Features

1. Multi-turn conversation support
2. Visual feedback improvements
3. Tool execution optimization
4. Advanced error handling

### Phase 4: Production Readiness

1. Comprehensive logging
2. Performance optimization
3. Security hardening
4. Documentation and testing

## 10. Testing Requirements

### 10.1 Unit Tests

- Tool discovery and registration
- OpenAI API integration
- Message history management
- Error handling scenarios

### 10.2 Integration Tests

- End-to-end conversation flows
- MCP server integration
- Tool execution with various result types
- Session persistence

### 10.3 User Acceptance Tests

- Natural conversation flows
- Tool execution feedback
- Error recovery scenarios
- Performance under load

## 11. Success Metrics

### 11.1 Technical Metrics

- **Tool Execution Success Rate**: >95% successful tool executions
- **Response Time**: <3 seconds for simple responses, <30 seconds for tool calls
- **Error Rate**: <5% unhandled errors
- **Session Stability**: >99% session persistence

### 11.2 User Experience Metrics

- **Conversation Completion Rate**: >90% successful task completion
- **User Satisfaction**: Positive feedback on tool execution clarity
- **Error Recovery**: Clear error messages leading to successful retries

## 12. Deployment and Configuration

### 12.1 Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install chainlit openai mcp python-dotenv

# Set environment variables
echo "OPENAI_API_KEY=sk-your-key-here" > .env
echo "DEFAULT_MODEL=gpt-4" >> .env
```

### 12.2 MCP Server Configuration

```json
{
  "mcpServers": {
    "server-name": {
      "command": "path/to/mcp/server",
      "args": ["--param", "value"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

### 12.3 Application Launch

```bash
# Run the Chainlit application
chainlit run app.py -w --port 8000
```

## 13. Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chainlit UI   │    │  OpenAI Client  │    │   MCP Servers   │
│                 │    │                 │    │                 │
│ • Chat Interface│    │ • Function Call │    │ • Tool Discovery│
│ • cl.Step UI    │    │ • Response Gen  │    │ • Tool Execution│
│ • Session Mgmt  │    │ • Tool Schema   │    │ • Result Format │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Main App Loop  │
                    │                 │
                    │ • Message Route │
                    │ • Tool Aggregate│
                    │ • Conversation  │
                    │ • Error Handle  │
                    └─────────────────┘
```

This comprehensive PRD provides the complete blueprint for building a robust Chainlit application with OpenAI SDK and MCP integration, ensuring proper tool calling capabilities and conversation continuity.
