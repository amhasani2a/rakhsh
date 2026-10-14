import json, threading, unittest, urllib.request, urllib.error
from http.server import ThreadingHTTPServer
from rakhsh.adapters.http import App, make_handler
from tests._util import trained_registry


class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(trained_registry(), token="s3cret")))
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()

    def req(self, path, body=None, token="s3cret", headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        if token: h["Authorization"] = f"Bearer {token}"
        data = None if body is None else json.dumps(body).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + path, data, h), timeout=30) as r:
                return r.status, json.loads(r.read()), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}"), dict(e.headers)

    def test_ui_served_without_token(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn("Rakhsh", r.read().decode())

    def test_token_required(self):
        self.assertEqual(self.req("/api/overview", token=None)[0], 401)
        self.assertEqual(self.req("/api/overview")[0], 200)

    def test_rest_ask(self):
        code, r, _ = self.req("/api/b/lib/ask", {"q": "نویسنده بوف کور"})
        self.assertEqual(code, 200); self.assertIn("answer", r)

    def test_mcp_over_http(self):
        code, r, h = self.req("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                       "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                                  "clientInfo": {"name": "t", "version": "1"}}})
        self.assertEqual(code, 200); self.assertIn("result", r)
        sid = {k.lower(): v for k, v in h.items()}.get("mcp-session-id")
        self.assertTrue(sid)
        code, r, _ = self.req("/mcp", {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers={"Mcp-Session-Id": sid})
        self.assertGreater(len(r["result"]["tools"]), 5)

    def test_foreign_origin_rejected(self):
        code, _, _ = self.req("/api/overview", headers={"Origin": "https://evil.example"})
        self.assertEqual(code, 403)


if __name__ == "__main__":
    unittest.main()
