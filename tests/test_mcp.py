import json, unittest
from rakhsh.adapters.mcp import MCPServer
from tests._util import trained_registry, SAMPLES


class TestMCPProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = MCPServer(trained_registry(), allowed_dirs=[str(SAMPLES)])
        cls.init = cls.s.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                            "clientInfo": {"name": "claude", "version": "1"}}})

    def rpc(self, method, params=None, id=1):
        return self.s.handle({"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}})

    def call(self, name, **args):
        return self.rpc("tools/call", {"name": name, "arguments": args})["result"]

    def test_initialize(self):
        r = self.init["result"]
        self.assertEqual(r["serverInfo"]["name"], "rakhsh")
        self.assertIn("tools", r["capabilities"])
        self.assertEqual(self.s.actor, "agent:claude")          # actor comes from clientInfo

    def test_notification_has_no_reply(self):
        self.assertIsNone(self.s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_tools_list_schema(self):
        tools = self.rpc("tools/list")["result"]["tools"]
        names = {t["name"] for t in tools}
        for n in ("brain_ask", "brain_ingest", "brain_feedback", "brain_teach", "decider_decide"):
            self.assertIn(n, names)
        self.assertTrue(all(t["inputSchema"]["type"] == "object" for t in tools))

    def test_ask_tool(self):
        r = self.call("brain_ask", brain="lib", question="نویسنده بوف کور")
        self.assertFalse(r.get("isError"))
        self.assertIn("message_id", json.loads(r["content"][0]["text"]))

    def test_tool_errors_are_results_not_crashes(self):
        r = self.call("brain_ask", brain="no-such-brain-xyz", question="x")
        self.assertTrue(r["isError"])

    def test_path_sandbox(self):
        r = self.call("brain_ingest", brain="lib", path="/etc/passwd")
        self.assertTrue(r["isError"])

    def test_unknown_method(self):
        self.assertEqual(self.rpc("nope/nope")["error"]["code"], -32601)

    def test_resources_and_prompts(self):
        uris = [x["uri"] for x in self.rpc("resources/list")["result"]["resources"]]
        self.assertIn("rakhsh://overview", uris)
        txt = self.rpc("resources/read", {"uri": "rakhsh://overview"})["result"]["contents"][0]["text"]
        self.assertIn("brains", json.loads(txt))
        names = [p["name"] for p in self.rpc("prompts/list")["result"]["prompts"]]
        self.assertIn("rakhsh_router", names)


if __name__ == "__main__":
    unittest.main()
