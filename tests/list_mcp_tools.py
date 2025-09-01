#!/usr/bin/env python3
"""
List available MCP tools using FastMCP
"""

import asyncio
from fastmcp import Client as FastMCPClient

async def list_mcp_tools():
    """List available tools on the MCP server."""
    try:
        client = FastMCPClient("mcp_server.py")
        async with client:
            tools = await client.list_tools()
            print("Available MCP Tools:")
            print("=" * 50)
            for i, tool in enumerate(tools, 1):
                print(f"{i}. {tool.name}")
                print(f"   Description: {tool.description}")
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    print(f"   Parameters: {tool.inputSchema}")
                print()
    except Exception as e:
        print(f"Error listing tools: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(list_mcp_tools())
