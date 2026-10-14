"""Talk to Rakhsh with the *official* MCP Python SDK (pip install mcp) — proves protocol compatibility.

    python examples/mcp_sdk_client.py
"""
import asyncio, json, os, pathlib, sys, tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = pathlib.Path(__file__).resolve().parents[1]


async def main():
    env = dict(os.environ, PYTHONPATH=str(ROOT), RAKHSH_HOME=tempfile.mkdtemp(prefix="rakhsh-mcp-"))
    params = StdioServerParameters(command=sys.executable, env=env,
                                   args=["-m", "rakhsh", "mcp", "--allow-dir", str(ROOT / "samples")])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            print("server:", init.server_info.name, "protocol", init.protocol_version)
            print("tools:", [t.name for t in (await s.list_tools()).tools])
            await s.call_tool("brain_ingest", {"brain": "lib", "path": str(ROOT / "samples" / "movies.csv")})
            res = await s.call_tool("brain_ask", {"brain": "lib", "question": "فیلم علمی‌تخیلی اکشن نولان سال ۲۰۱۰"})
            out = json.loads(res.content[0].text)
            print("answer:", out["answer"].get("record") or out["answer"]["text"][:200])
            print("level:", out["decision"]["level_name"], "confidence:", out["decision"]["confidence"])


asyncio.run(main())
