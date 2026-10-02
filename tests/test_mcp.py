import asyncio

from mcp import Client

from mini_supermemory.mcp_server import build_server
from tests.test_linking import ADIDAS, PUMA, sneaker_llm  # noqa: F401
from tests.test_retrieval import SNEAKERS


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def test_mcp_tools_share_the_engine(engine, sneaker_llm):  # noqa: F811
    server = build_server(lambda: engine, "khan")

    async def run():
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == {"memory", "recall", "context"}
            for msg in SNEAKERS:
                saved = _text(await client.call_tool("memory", {"content": msg}))
                assert saved.startswith("Saved")
            recall = _text(await client.call_tool("recall", {"query": "shoe preferences"}))
            ctx = _text(await client.call_tool("context", {}))
            forgot = _text(await client.call_tool("memory", {"content": "switching to Puma sneakers",
                                                             "action": "forget"}))
            return recall, ctx, forgot

    recall, ctx, forgot = asyncio.run(run())
    assert PUMA in recall and ADIDAS not in recall
    assert PUMA in ctx and ADIDAS not in ctx
    assert forgot == f"Forgot: {PUMA}"
    # Same store: data saved through MCP is visible to the engine/API/UI.
    assert len(engine.list_documents("khan")) == 3
