"""Zero-dependency HTTP adapter: web chat UI + REST API + MCP (Streamable HTTP) on one port.

    GET  /                               web UI
    GET  /api/overview                   brains + deciders
    GET  /api/b/<brain>/state
    POST /api/b/<brain>/{upload,text,ask,search,probe,feedback,teach,sleep,remove,reset,snapshot,restore,delete}
    POST /api/d/<decider>/{create,teach,decide}
    POST /mcp                            MCP JSON-RPC (Mcp-Session-Id header per client)

Security: binds to 127.0.0.1 by default; ``--token`` enables Bearer auth (required when binding
to other interfaces); browser Origin is checked on every API call to prevent CSRF / DNS-rebinding.
Actor = header ``X-Rakhsh-Actor: agent:<name>`` (trust from policy) or ``human`` for the UI.
"""
from __future__ import annotations

import base64
import hmac
import json
import logging
import pathlib
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from ..errors import NotFound, NotReady, RakhshError
from ..registry import Registry
from .mcp import MCPServer, _jsonable

log = logging.getLogger("rakhsh.http")
WEB = pathlib.Path(__file__).resolve().parent / "web" / "index.html"
MAX_BODY = 200 * 1024 * 1024


class App:
    def __init__(self, registry: Registry, token: str | None = None, allowed_dirs: list[str] | None = None):
        self.reg, self.token, self.allowed_dirs = registry, token, allowed_dirs
        self.sessions: dict[str, MCPServer] = {}; self._lock = threading.Lock()

    # ------------------------------------------------------------------ REST
    def brain_action(self, name: str, action: str, body: dict, actor: str):
        reg = self.reg
        if action == "state":
            return reg.brain(name).state()
        if action == "delete":
            reg.delete_brain(name); return {"ok": True}
        if action == "snapshot":
            return {"id": reg.snapshot(name, body.get("label")), "all": reg.snapshots(name)}
        if action == "restore":
            reg.restore(name, body["id"]); return {"ok": True}
        b = reg.brain(name)
        if action == "upload":
            return {"ok": True, "source": b.ingest(body["name"], base64.b64decode(body["b64"]), actor=actor)}
        if action == "text":
            return {"ok": True, "source": b.ingest_text(body.get("name") or "متن", body["text"], actor=actor)}
        if action == "ask":
            return b.ask(body["q"], body.get("k"))
        if action == "search":
            r = b.search(body["q"], body.get("k")); return {"id": r["id"], "ranking": r["ranking"], "meta": r["meta"]}
        if action == "probe":
            return b.probe(body.get("q", ""))
        if action == "feedback":
            return b.feedback(int(body["id"]), body["kind"], body.get("chunk"), actor=actor)
        if action == "teach":
            return b.teach(body["question"], body.get("answer"), body.get("chunk"), actor=actor)
        if action == "sleep":
            return b.sleep(actor=actor)
        if action == "remove":
            b.remove_source(body["name"], actor=actor); return {"ok": True}
        if action == "reset":
            b.reset(actor=actor); return {"ok": True}
        raise NotFound(f"unknown action {action}")

    def decider_action(self, name: str, action: str, body: dict, actor: str):
        if action == "create":
            return self.reg.create_decider(name, body["labels"], body.get("mode", "categorical"), body.get("description", "")).info()
        d = self.reg.decider(name)
        if action == "teach":
            return d.teach(body["examples"], actor=actor)
        if action == "decide":
            return d.decide(body["text"])
        if action == "info":
            return d.info()
        raise NotFound(f"unknown action {action}")

    def mcp(self, msg, session: str | None):
        with self._lock:
            new = None
            is_init = isinstance(msg, dict) and msg.get("method") == "initialize"
            if is_init or session not in self.sessions:
                srv = MCPServer(self.reg, self.allowed_dirs)
                if is_init:
                    new = uuid.uuid4().hex; self.sessions[new] = srv
            else:
                srv = self.sessions[session]
        if isinstance(msg, list):
            out = [r for r in (srv.handle(m) for m in msg) if r is not None] or None
        else:
            out = srv.handle(msg)
        return out, new


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "rakhsh"

        def log_message(self, fmt, *a):
            log.debug(fmt, *a)

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8", headers=None):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def _json(self, obj, code=200, headers=None):
            self._send(code, json.dumps(obj, ensure_ascii=False, default=_jsonable).encode(), headers=headers)

        def _origin_ok(self) -> bool:
            """Browsers send Origin on cross-site requests: only the local UI may call us (anti CSRF / DNS-rebinding)."""
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1", "::1"):
                self._json({"error": "forbidden origin"}, 403); return False
            return True

        def _authorized(self) -> bool:
            if not self._origin_ok():
                return False
            if not app.token:
                return True
            got = self.headers.get("Authorization", "")
            ok = hmac.compare_digest(got, f"Bearer {app.token}")
            if not ok:
                self._json({"error": "unauthorized"}, 401)
            return ok

        def _actor(self) -> str:
            a = self.headers.get("X-Rakhsh-Actor", "").strip()
            return a if a.startswith("agent:") else "human"

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, WEB.read_bytes(), "text/html; charset=utf-8")
            if not self._authorized():
                return
            if path == "/api/overview":
                return self._json(app.reg.overview())
            if path == "/mcp":
                return self._json({"error": "SSE stream not supported; POST JSON-RPC"}, 405, {"Allow": "POST"})
            parts = [unquote(p) for p in path.strip("/").split("/")]
            if len(parts) == 4 and parts[:2] == ["api", "b"] and parts[3] == "state":
                return self._guard(lambda: app.brain_action(parts[2], "state", {}, "human"))
            self._json({"error": "not found"}, 404)

        def do_DELETE(self):
            if urlparse(self.path).path == "/mcp":
                app.sessions.pop(self.headers.get("Mcp-Session-Id", ""), None); return self._send(204, b"")
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._authorized():
                return
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._json({"error": "payload too large"}, 413)
            raw = self.rfile.read(n) or b"{}"
            path = urlparse(self.path).path
            if path == "/mcp":
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    return self._json({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}, 400)
                out, sid = app.mcp(msg, self.headers.get("Mcp-Session-Id"))
                hdr = {"Mcp-Session-Id": sid} if sid else None
                return self._send(202, b"") if out is None else self._json(out, headers=hdr)
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return self._json({"error": "invalid JSON"}, 400)
            parts = [unquote(p) for p in path.strip("/").split("/")]
            if len(parts) == 4 and parts[0] == "api" and parts[1] in ("b", "d"):
                fn = app.brain_action if parts[1] == "b" else app.decider_action
                return self._guard(lambda: fn(parts[2], parts[3], body, self._actor()))
            self._json({"error": "not found"}, 404)

        def _guard(self, fn):
            try:
                self._json(fn())
            except NotFound as e:
                self._json({"error": str(e)}, 404)
            except NotReady as e:
                self._json({"error": str(e)}, 409)
            except (RakhshError, KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:
                log.error(traceback.format_exc()); self._json({"error": f"internal error: {e}"}, 500)
    return H


def serve(registry: Registry, host="127.0.0.1", port=8765, token=None, open_browser=True, allowed_dirs=None):
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise SystemExit("refusing to bind a non-local interface without --token")
    app = App(registry, token, allowed_dirs)
    srv = ThreadingHTTPServer((host, port), make_handler(app)); srv.daemon_threads = True
    url = f"http://{host}:{port}"
    print(f"🐎 Rakhsh {url}   ·  MCP: {url}/mcp   ·  home: {registry.home}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nsaving…")
    finally:
        registry.save_all(); srv.server_close()
