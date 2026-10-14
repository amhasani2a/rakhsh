"""Model Context Protocol server for Rakhsh (zero-dependency JSON-RPC 2.0).

Transports
  * stdio            – ``rakhsh mcp``  (Claude Desktop, Cursor, VS Code, any MCP client)
  * Streamable HTTP  – ``POST /mcp`` on ``rakhsh serve`` (JSON responses)

Capabilities: tools (brains, deciders, training, collaboration), resources (status, audit),
prompts (how to use Rakhsh as a router / as a teacher).

Every call is attributed to an *actor* = ``agent:<clientInfo.name>``; its trust comes from the
server's TrustPolicy (``--trust claude=0.8``) and can never be self-declared by the agent.
"""
from __future__ import annotations

import base64
import json
import logging
import pathlib
import sys
import traceback
from typing import Any, Callable

from .._version import __version__
from ..errors import RakhshError
from ..ingest import supported_extensions
from ..registry import Registry

log = logging.getLogger("rakhsh.mcp")
PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

S_BRAIN = {"type": "string", "description": "Brain name (a separate knowledge space). Default 'default'.", "default": "default"}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


TOOLS: list[dict] = [
    {"name": "rakhsh_overview", "title": "List brains & deciders",
     "description": "List all brains (knowledge spaces) and deciders (decision layers) with their status.",
     "inputSchema": _obj({}), "annotations": {"readOnlyHint": True}},
    {"name": "brain_ingest", "title": "Add knowledge",
     "description": "Add knowledge to a brain from a local file path, raw text, or base64 content. Supported: "
                    + ", ".join(supported_extensions()) + ". Newly added data is searchable immediately; the brain "
                    "self-trains in the background (check brain_status).",
     "inputSchema": _obj({"brain": S_BRAIN, "path": {"type": "string", "description": "Local file path (must be inside an allowed directory)."},
                          "text": {"type": "string", "description": "Raw text to add (e.g. pasted notes)."},
                          "name": {"type": "string", "description": "Source name (required with text/content_base64)."},
                          "content_base64": {"type": "string", "description": "File bytes, base64-encoded."}})},
    {"name": "brain_status", "title": "Brain status",
     "description": "Training state, sources, sizes, learning statistics of a brain.",
     "inputSchema": _obj({"brain": S_BRAIN}), "annotations": {"readOnlyHint": True}},
    {"name": "brain_ask", "title": "Ask (quoted answer)",
     "description": "Answer a question by QUOTING the brain's own data (extractive, with source + locator). "
                    "Returns message_id for brain_feedback, the reflex decision trace (level, spikes, confidence) "
                    "and alternatives with chunk ids. `unsure=true` means the data probably doesn't contain the answer.",
     "inputSchema": _obj({"brain": S_BRAIN, "question": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 50}}, ["question"])},
    {"name": "brain_search", "title": "Search",
     "description": "Ranked retrieval only (no answer extraction): chunk ids, scores, sources, short snippets.",
     "inputSchema": _obj({"brain": S_BRAIN, "query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 50}}, ["query"]),
     "annotations": {"readOnlyHint": True}},
    {"name": "brain_probe", "title": "Reflex decision (no retrieval)",
     "description": "Sub-millisecond decision layer: which spike neurons fire for a text and how deep the hazard gate "
                    "wants to search (0 reflex/memory, 1 keyword, 2 deep). Use it to decide whether retrieval or an "
                    "LLM call is worth it.",
     "inputSchema": _obj({"brain": S_BRAIN, "text": {"type": "string"}}, ["text"]), "annotations": {"readOnlyHint": True}},
    {"name": "brain_feedback", "title": "Feedback on an answer",
     "description": "Teach the brain from an answer it gave: verdict 'up' (right), 'down' (wrong) or 'pick' with the "
                    "correct chunk id from the alternatives. Weighted by your trust level.",
     "inputSchema": _obj({"brain": S_BRAIN, "message_id": {"type": "integer"},
                          "verdict": {"type": "string", "enum": ["up", "down", "pick"]},
                          "chunk": {"type": "integer"}}, ["message_id", "verdict"])},
    {"name": "brain_teach", "title": "Teach question → answer",
     "description": "Directly teach that `question` should be answered by `answer` text (matched to existing data, or "
                    "added as new knowledge) or by a specific `chunk` id. Ideal for teacher→student distillation.",
     "inputSchema": _obj({"brain": S_BRAIN, "question": {"type": "string"}, "answer": {"type": "string"},
                          "chunk": {"type": "integer"}}, ["question"])},
    {"name": "brain_sleep", "title": "Consolidate (sleep)",
     "description": "Replay trusted experience into slow weights, refit the gate and decay fast synapses.",
     "inputSchema": _obj({"brain": S_BRAIN})},
    {"name": "brain_snapshot", "title": "Snapshot / restore",
     "description": "action='create' saves a restorable snapshot; 'list' lists them; 'restore' rolls back to `id`.",
     "inputSchema": _obj({"brain": S_BRAIN, "action": {"type": "string", "enum": ["create", "list", "restore"]},
                          "label": {"type": "string"}, "id": {"type": "string"}}, ["action"])},
    {"name": "brain_remove_source", "title": "Remove a source",
     "description": "Remove one source (file) and all its chunks from a brain.",
     "inputSchema": _obj({"brain": S_BRAIN, "source": {"type": "string"}}, ["source"]), "annotations": {"destructiveHint": True}},
    {"name": "decider_create", "title": "Create a decision layer",
     "description": "Create a few-shot decider with your own labels. mode='categorical' (alternatives, e.g. tools) or "
                    "'ordinal' (escalation ladder, e.g. no-llm < small-llm < large-llm; escalates when unsure).",
     "inputSchema": _obj({"name": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                          "mode": {"type": "string", "enum": ["categorical", "ordinal"], "default": "categorical"},
                          "description": {"type": "string"}}, ["name", "labels"])},
    {"name": "decider_teach", "title": "Teach a decider",
     "description": "Add labelled examples [{text, label}] to a decider (instant episodic recall + a few gradient steps).",
     "inputSchema": _obj({"name": {"type": "string"}, "examples": {"type": "array", "minItems": 1, "items": {
         "type": "object", "properties": {"text": {"type": "string"}, "label": {"type": "string"}}, "required": ["text", "label"]}}},
         ["name", "examples"])},
    {"name": "decider_decide", "title": "Decide",
     "description": "Single-forward decision: label, confidence, probabilities, abstained flag and word-level evidence. "
                    "If abstained, consult a stronger model and teach the decider the result.",
     "inputSchema": _obj({"name": {"type": "string"}, "text": {"type": "string"}}, ["name", "text"]), "annotations": {"readOnlyHint": True}},
]

PROMPTS = [
    {"name": "rakhsh_router", "description": "Use a Rakhsh decider as a cheap first-pass router before expensive calls.",
     "arguments": [{"name": "decider", "description": "decider name", "required": True}]},
    {"name": "rakhsh_teacher", "description": "Act as a teacher: answer, then distil your answer into a Rakhsh brain.",
     "arguments": [{"name": "brain", "description": "brain name", "required": True}]},
]


class MCPServer:
    def __init__(self, registry: Registry, allowed_dirs: list[str] | None = None, default_actor: str = "agent:unknown"):
        self.reg = registry
        self.allowed = [pathlib.Path(p).expanduser().resolve() for p in (allowed_dirs or [str(pathlib.Path.cwd())])]
        self.actor = default_actor
        self.handlers: dict[str, Callable[[dict], Any]] = {
            "initialize": self._initialize, "ping": lambda p: {},
            "tools/list": lambda p: {"tools": TOOLS}, "tools/call": self._call,
            "resources/list": self._resources, "resources/read": self._read,
            "resources/templates/list": lambda p: {"resourceTemplates": [
                {"uriTemplate": "rakhsh://brain/{name}", "name": "brain status", "mimeType": "application/json"}]},
            "prompts/list": lambda p: {"prompts": PROMPTS}, "prompts/get": self._prompt,
        }

    # ------------------------------------------------------------------ JSON-RPC
    def handle(self, msg: dict) -> dict | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        mid, method = msg.get("id"), msg.get("method")
        if method is None:                                   # a response from the client – ignore
            return None
        if mid is None:                                      # notification (e.g. notifications/initialized)
            return None
        fn = self.handlers.get(method)
        if fn is None:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        try:
            return {"jsonrpc": "2.0", "id": mid, "result": fn(msg.get("params") or {})}
        except _InvalidParams as e:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": str(e)}}
        except Exception as e:
            log.error("internal error: %s", traceback.format_exc())
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}

    def _initialize(self, p: dict) -> dict:
        client = (p.get("clientInfo") or {}).get("name") or "unknown"
        self.actor = "agent:" + "".join(ch for ch in client.lower() if ch.isalnum() or ch in "-_")[:40]
        ver = p.get("protocolVersion")
        return {"protocolVersion": ver if ver in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}, "prompts": {"listChanged": False}},
                "serverInfo": {"name": "rakhsh", "title": "Rakhsh · رخش", "version": __version__},
                "instructions": ("Rakhsh is a reflex retrieval & decision core. Brains answer by quoting the user's own data "
                                 "(always cite source+locator). Give feedback (brain_feedback / brain_teach) so it learns; "
                                 "use deciders as sub-millisecond routers and teach them when they abstain.")}

    # ------------------------------------------------------------------ tools
    def _call(self, p: dict) -> dict:
        name, args = p.get("name"), p.get("arguments") or {}
        if name not in {t["name"] for t in TOOLS}:
            raise _InvalidParams(f"unknown tool: {name}")
        try:
            result = getattr(self, "t_" + name)(args)
            text = json.dumps(result, ensure_ascii=False, indent=1, default=_jsonable)
            return {"content": [{"type": "text", "text": text}], "structuredContent": json.loads(text),
                    "isError": False}
        except (RakhshError, KeyError, ValueError, FileNotFoundError, PermissionError) as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}

    def t_rakhsh_overview(self, a):
        return self.reg.overview()

    def t_brain_ingest(self, a):
        b = self.reg.brain(a.get("brain", "default"))
        if a.get("path"):
            path = pathlib.Path(a["path"]).expanduser().resolve()
            if not any(path.is_relative_to(d) for d in self.allowed):
                raise PermissionError(f"path outside allowed directories {[str(d) for d in self.allowed]} (start with --allow-dir)")
            info = b.ingest(a.get("name") or path.name, path.read_bytes(), actor=self.actor)
        elif a.get("content_base64"):
            info = b.ingest(_need(a, "name"), base64.b64decode(a["content_base64"]), actor=self.actor)
        elif a.get("text"):
            info = b.ingest_text(a.get("name") or "text", a["text"], actor=self.actor)
        else:
            raise _InvalidParams("give one of: path, text, content_base64")
        return {"ok": True, "source": info, "status": b.status}

    def t_brain_status(self, a):
        return self.reg.brain(a.get("brain", "default"), create=False).state()

    def t_brain_ask(self, a):
        r = self.reg.brain(a.get("brain", "default"), create=False).ask(_need(a, "question"), a.get("k"))
        ans = r["answer"]
        return {"message_id": r["id"],
                "answer": {k: ans[k] for k in ("text", "source", "locator", "title", "chunk", "kind")} | ({"record": ans["record"]} if "record" in ans else {}),
                "alternatives": [{k: h[k] for k in ("chunk", "text", "source", "locator", "score")} for h in r["hits"]],
                "decision": {k: r["meta"][k] for k in ("level", "level_name", "path", "k", "confidence", "unsure", "ms")}
                | {"spikes": [s[0] + ("" if s[2] else "*") for s in r["meta"]["spikes"][:12]]}}

    def t_brain_search(self, a):
        b = self.reg.brain(a.get("brain", "default"), create=False)
        r = b.search(_need(a, "query"), a.get("k"))
        return {"message_id": r["id"], "decision": {k: r["meta"][k] for k in ("level", "level_name", "path", "confidence")},
                "hits": [{"chunk": i, "score": round(s, 4), "source": b.chunks[i].source, "locator": b.chunks[i].locator,
                          "snippet": b.chunks[i].text[:240]} for i, s in r["ranking"][:r["k"]]]}

    def t_brain_probe(self, a):
        return self.reg.brain(a.get("brain", "default"), create=False).probe(_need(a, "text"))

    def t_brain_feedback(self, a):
        return self.reg.brain(a.get("brain", "default"), create=False).feedback(
            _need(a, "message_id"), _need(a, "verdict"), a.get("chunk"), actor=self.actor)

    def t_brain_teach(self, a):
        return self.reg.brain(a.get("brain", "default"), create=False).teach(
            _need(a, "question"), a.get("answer"), a.get("chunk"), actor=self.actor)

    def t_brain_sleep(self, a):
        return self.reg.brain(a.get("brain", "default"), create=False).sleep(actor=self.actor)

    def t_brain_snapshot(self, a):
        n = a.get("brain", "default"); act = _need(a, "action")
        if act == "create":
            return {"id": self.reg.snapshot(n, a.get("label"))}
        if act == "list":
            return {"snapshots": self.reg.snapshots(n)}
        self.reg.restore(n, _need(a, "id")); return {"ok": True, "restored": a["id"]}

    def t_brain_remove_source(self, a):
        self.reg.brain(a.get("brain", "default"), create=False).remove_source(_need(a, "source"), actor=self.actor)
        return {"ok": True}

    def t_decider_create(self, a):
        return self.reg.create_decider(_need(a, "name"), _need(a, "labels"), a.get("mode", "categorical"),
                                       a.get("description", "")).info()

    def t_decider_teach(self, a):
        return self.reg.decider(_need(a, "name")).teach(_need(a, "examples"), actor=self.actor)

    def t_decider_decide(self, a):
        return self.reg.decider(_need(a, "name")).decide(_need(a, "text"))

    # ------------------------------------------------------------------ resources / prompts
    def _resources(self, p):
        res = [{"uri": "rakhsh://overview", "name": "overview", "mimeType": "application/json"},
               {"uri": "rakhsh://audit", "name": "audit log (last 100 events)", "mimeType": "application/json"}]
        res += [{"uri": f"rakhsh://brain/{n}", "name": f"brain {n}", "mimeType": "application/json"} for n in self.reg.brain_names()]
        return {"resources": res}

    def _read(self, p):
        uri = _need(p, "uri")
        if uri == "rakhsh://overview":
            data = self.reg.overview()
        elif uri == "rakhsh://audit":
            data = self.reg.audit.tail(100)
        elif uri.startswith("rakhsh://brain/"):
            data = self.reg.brain(uri.rsplit("/", 1)[-1], create=False).state()
        else:
            raise _InvalidParams(f"unknown resource {uri}")
        return {"contents": [{"uri": uri, "mimeType": "application/json",
                              "text": json.dumps(data, ensure_ascii=False, indent=1, default=_jsonable)}]}

    def _prompt(self, p):
        name, args = _need(p, "name"), p.get("arguments") or {}
        if name == "rakhsh_router":
            d = args.get("decider", "router")
            text = (f"Before each expensive step, call decider_decide(name='{d}', text=<the request>). Follow its label. "
                    f"If `abstained` is true, decide yourself, act, then call decider_teach with your choice so the "
                    f"router learns. Never skip teaching after an abstention.")
        elif name == "rakhsh_teacher":
            b = args.get("brain", "default")
            text = (f"Answer questions using brain_ask(brain='{b}') first and cite source+locator. If the quoted answer "
                    f"is right, call brain_feedback(verdict='up'); if another alternative is right use verdict='pick' "
                    f"with its chunk; if none, find the answer and call brain_teach(question, answer).")
        else:
            raise _InvalidParams(f"unknown prompt {name}")
        return {"description": name, "messages": [{"role": "user", "content": {"type": "text", "text": text}}]}


class _InvalidParams(Exception):
    pass


def _need(a: dict, key: str):
    if key not in a or a[key] in (None, ""):
        raise _InvalidParams(f"missing argument: {key}")
    return a[key]


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def run_stdio(server: MCPServer) -> None:
    """Newline-delimited JSON-RPC over stdin/stdout. Logs go to stderr only."""
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            if isinstance(msg, list):                     # batch (2025-03-26)
                resp = [r for r in (server.handle(m) for m in msg) if r is not None] or None
            else:
                resp = server.handle(msg)
        if resp is not None:
            out.write(json.dumps(resp, ensure_ascii=False, default=_jsonable) + "\n"); out.flush()
    server.reg.save_all()
