from contextlib import AsyncExitStack
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import sys
from databricks.sdk import WorkspaceClient
from mcp import ClientSession, StdioServerParameters, stdio_client


class MCPClient:
    def __init__(self, llm_endpoint_name: str = "databricks-meta-llama-3-3-70b-instruct"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Initialize Databricks client and get OpenAI-compatible client
        self.workspace = WorkspaceClient()
        self.llm_endpoint_name = llm_endpoint_name
        self.openai_client = self.workspace.serving_endpoints.get_open_ai_client()
        print(f"Initialized OpenAI-compatible client for {llm_endpoint_name}")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # Display tool descriptions
        print("\nAvailable Tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

    def _convert_schema_to_openai_format(self, tool):
        """Convert MCP tool schema to OpenAI function format"""
        # Extract the input schema from the tool
        schema = tool.inputSchema

        # Create the function definition
        function_def = {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema
        }

        # Return in the OpenAI tool format
        return {
            "type": "function",
            "function": function_def
        }

    async def process_query(self, query: str) -> str:
        """Process a query using Meta Llama and available tools"""
        if not self.session:
            print("Error: Not connected to a server")
            return "Not connected to server."

        # Initialize conversation with user query
        messages = [
            {"role": "user", "content": query}
        ]

        # Get available tools and convert to OpenAI format
        response = await self.session.list_tools()
        available_tools = [self._convert_schema_to_openai_format(tool) for tool in response.tools]

        final_text = []

        try:
            # Make initial LLM API call
            print("Sending query to LLM with tools...")
            llm_response = self.openai_client.chat.completions.create(
                model=self.llm_endpoint_name,
                messages=messages,
                tools=available_tools,
                max_tokens=1000
            )

            # Get the assistant's response
            assistant_message = llm_response.choices[0].message

            # Add any content from the assistant to our output
            if assistant_message.content:
                final_text.append(assistant_message.content)

            # Check for tool calls
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    # Execute tool call
                    try:
                        final_text.append(f"\n[Calling tool: {tool_name} with arguments: {tool_args}]")
                        result = await self.session.call_tool(tool_name, tool_args)

                        # Extract the raw text from the result
                        raw_result = str(result.content)

                        # If it's a TextContent object, extract just the text part
                        if 'TextContent' in raw_result and "text='" in raw_result:
                            # Find the text content between quotes
                            text_start = raw_result.index("text='") + 6
                            text_end = raw_result.rindex("'")
                            if text_start < text_end:
                                cleaned_result = raw_result[text_start:text_end]
                                # Unescape newlines and tabs
                                cleaned_result = cleaned_result.replace('\\n', '\n').replace('\\t', '\t')
                                final_text.append(f"\nForecast for {tool_args.get('latitude', '')},{tool_args.get('longitude', '')}:\n{cleaned_result}")
                            else:
                                final_text.append(f"\nResult: {raw_result}")
                        else:
                            final_text.append(f"\nResult: {raw_result}")
                    except Exception as e:
                        final_text.append(f"\nError calling tool: {str(e)}")
        except Exception as e:
            final_text.append(f"\nError processing query: {str(e)}")


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Weather Client Started!")
        print("You can ask questions about weather. For example:")
        print("- What's the weather forecast for San Francisco?")
        print("- Are there any weather alerts in CA?")
        print("- Tell me the forecast for latitude 37.7749 and longitude -122.4194")
        print("(Type 'quit' to exit)")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                if query.strip():
                    print("\nProcessing query...")
                    response = await self.process_query(query)
                    print("\nResponse:")
                    print(response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        if self.session:
            self.session.close()
            self.session = None
        if self.exit_stack:
            self.exit_stack.close()
            self.exit_stack = None
        print("Cleanup completed.")


async def main():
    # Default to weather.py in the current directory if no path provided
    server_script_path = sys.argv[1] if len(sys.argv) > 1 else "mcp_weather_server.py"

    # Optional second argument for the LLM endpoint name
    llm_endpoint_name = sys.argv[2] if len(sys.argv) > 2 else "databricks-meta-llama-3-3-70b-instruct"

    client = MCPClient(llm_endpoint_name)
    try:
        print(f"Connecting to server: {server_script_path}")
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())