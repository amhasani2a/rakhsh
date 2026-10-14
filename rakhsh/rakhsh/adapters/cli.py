"""`rakhsh` command-line interface."""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

from .._version import __version__
from ..config import TrustPolicy

DEFAULT_HOME = os.environ.get("RAKHSH_HOME", str(pathlib.Path.home() / ".rakhsh"))


def _registry(args):
    from ..registry import Registry
    agents = {}
    for t in args.trust or []:
        k, _, v = t.partition("=")
        agents[k.strip()] = float(v)
    return Registry(args.home, trust=TrustPolicy(default_agent=args.agent_trust, agents=agents))


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rakhsh", description="رخش · Rakhsh — reflex retrieval & decision core")
    ap.add_argument("--version", action="version", version=f"rakhsh {__version__}")
    ap.add_argument("--home", default=DEFAULT_HOME, help="state directory (env RAKHSH_HOME), default ~/.rakhsh")
    ap.add_argument("--trust", action="append", metavar="AGENT=0.8", help="trust level of an MCP/HTTP agent (repeatable)")
    ap.add_argument("--agent-trust", type=float, default=0.5, help="default trust of unknown agents")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="web chat UI + REST API + MCP over HTTP")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", default=os.environ.get("RAKHSH_TOKEN")); s.add_argument("--no-browser", action="store_true")
    s.add_argument("--allow-dir", action="append", help="directories MCP clients may ingest from (default: cwd)")
    m = sub.add_parser("mcp", help="MCP server over stdio (for Claude Desktop, Cursor, …)")
    m.add_argument("--allow-dir", action="append", help="directories MCP clients may ingest from (default: cwd)")

    i = sub.add_parser("ingest", help="add files to a brain and train"); i.add_argument("brain"); i.add_argument("files", nargs="+")
    q = sub.add_parser("ask", help="ask a brain"); q.add_argument("brain"); q.add_argument("question", nargs="+")
    q.add_argument("--json", action="store_true")
    sub.add_parser("list", help="list brains and deciders")
    d = sub.add_parser("decider", help="manage deciders")
    dsub = d.add_subparsers(dest="dcmd", required=True)
    dc = dsub.add_parser("create"); dc.add_argument("name"); dc.add_argument("--labels", required=True, help="comma separated")
    dc.add_argument("--mode", choices=["categorical", "ordinal"], default="categorical"); dc.add_argument("--description", default="")
    dt = dsub.add_parser("teach"); dt.add_argument("name"); dt.add_argument("--label", required=True); dt.add_argument("texts", nargs="+")
    dr = dsub.add_parser("run"); dr.add_argument("name"); dr.add_argument("text", nargs="+")
    b = sub.add_parser("bench", help="reproduce the research benchmark"); b.add_argument("--data", default="data"); b.add_argument("--steps", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.cmd == "serve":
        from .http import serve
        return serve(_registry(args), args.host, args.port, args.token, not args.no_browser, args.allow_dir)
    if args.cmd == "mcp":
        from .mcp import MCPServer, run_stdio
        return run_stdio(MCPServer(_registry(args), args.allow_dir))
    if args.cmd == "bench":
        from ..research.bench import run
        return print(run(args.data, args.steps)["markdown"])
    reg = _registry(args)
    if args.cmd == "list":
        return _print(reg.overview())
    if args.cmd == "ingest":
        br = reg.brain(args.brain)
        for f in args.files:
            p = pathlib.Path(f); _print(br.ingest(p.name, p.read_bytes(), train=False))
        br.train(); return print(br.status["message"])
    if args.cmd == "ask":
        r = reg.brain(args.brain, create=False).ask(" ".join(args.question))
        if args.json:
            return _print(r)
        a, m = r["answer"], r["meta"]
        print(f"[{m['level_name']} · {m['path']} · conf {m['confidence']:.2f} · {m['ms']['forward']}ms]")
        print(f"📍 {a['source']} › {a['locator']}\n{a['text']}")
        return None
    if args.cmd == "decider":
        if args.dcmd == "create":
            return _print(reg.create_decider(args.name, [x.strip() for x in args.labels.split(",")], args.mode, args.description).info())
        dd = reg.decider(args.name)
        if args.dcmd == "teach":
            return _print(dd.teach([{"text": t, "label": args.label} for t in args.texts]))
        return _print(dd.decide(" ".join(args.text)))


if __name__ == "__main__":
    main()
