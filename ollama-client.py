from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import (
    ToolCall, ToolCallResult
)

from llama_index.tools.mcp import McpToolSpec
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient

llm = Ollama(model="deepseek-r1", request_timeout=120.0)
Settings.llm = llm


SYSTEM_PROMPT = """

You are an AI assistant for Tool Calling.
Before helping, work with our tools to interact with our database.

"""


async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()

    agent = FunctionAgent(
        name="Agent",
        description="agent that interacts with our database",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT
    )

    return agent


async def handle_user_message(
    message_content: str,
    agent: FunctionAgent,
    agent_context: Context,
    verbose: bool = False
):
    handler = agent.run(message_content, ctx=agent_context)
    async for event in handler.stream_events():
        if verbose and type(event) == ToolCall:
            print(f"Calling tool {event.tool_name}")
        elif verbose and type(event) == ToolCallResult:
            print(f"Tool {event.tool_name} return {event.tool_output}")

    response = await handler
    return str(response)


mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool = McpToolSpec(client=mcp_client)

agent = await get_agent(mcp_tool)
context = Context(agent)


while True:
    msg = input("> ")
    if msg.lower() == "exit":
        break
    resp = await handle_user_message(msg, agent=agent, context=context)
    print("Agent: ", resp)
