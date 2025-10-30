import asyncio, json
from fastmcp import Client
from prometheus_mcp_server.server import mcp

# async def main():
#     async with Client(mcp) as client:
#         res = await client.call_tool("list_metrics", {})
#         print(f"Total metrics: {len(res.data)}")
#         for metric in res.data:
#             if "oreo" in metric or "mlserve" in metric:
#                 print(metric)
# asyncio.run(main())


import asyncio
from fastmcp import Client

async def main():
    # Connect to your HTTP server (FastMCP uses JSON-RPC over /mcp)
    async with Client("http://localhost:8080/mcp") as client:
        await client.ping()
        result = await client.call_tool("health_check", {})
        print(result)

asyncio.run(main())