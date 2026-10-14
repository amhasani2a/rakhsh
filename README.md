<div align="center">

# 🐎 Rakhsh · رخش

**A reflex-based, single-forward retrieval & decision core that learns from *your* data and *your* feedback.**
Pure Python + NumPy · no GPU · no cloud · MCP server · REST API · web chat UI

[فارسی](README.fa.md) · [Course (10 sessions)](docs/course/README.md) · [Architecture](docs/ARCHITECTURE.md)

</div>

> In the Shahnameh, while Rostam slept, a lion attacked — and **Rakhsh**, his horse, killed it *before Rostam woke up*.
> That is a reflex: the right action, instantly, without waking the slow "thinking" brain.
> This project gives your data a Rakhsh.

---

## 1. What is it?

Most AI search stacks are *agents in a loop*: plan → call a tool → read → re-plan → call again. Each step costs a
large-model call. Rakhsh does the opposite: **one forward pass of a tiny network** produces, *at the same time*:

| head | what it gives | analogy |
|---|---|---|
| **dense** | a meaning vector (semantic search) | "what is this about?" |
| **sparse "spikes"** | a few firing word-neurons, incl. words you *didn't* type (query expansion) | "which keywords light up?" |
| **hazard gate** | a probability of stopping at each depth: `reflex → spike → deep` | "how hard do I need to think?" |

Plus it **keeps learning after deployment**: every 👍/👎/"this one" builds fast Hebbian synapses and *episodic reflex
neurons* (a confirmed question is answered next time from memory — `path = reflex-arc` — without searching at all), and
`sleep()` consolidates experience into the slow weights.

It answers **extractively** — it *quotes* your data with source + locator (chapter/paragraph, table/row). It never
invents text. That makes it ideal as:

* a **private knowledge base** over books, notes, CSV/Excel/SQLite/JSON, PDF/Word/HTML;
* a **memory + retrieval tool for LLM agents** (Claude, Cursor, any MCP client) — cheaper than another LLM call;
* a **micro decision layer** (`Decider`): route prompts to no/small/large LLM, triage tickets, classify intents —
  taught with a handful of examples, decides in **< 1 ms**, explains *which words* drove the decision.

## 2. Architecture

```
            ┌──────────────── adapters (thin, replaceable) ────────────────┐
            │  Web UI      REST /api/b/<brain>/<action>     MCP (stdio|HTTP) │   CLI   SDK
            └───────────────────────────────┬──────────────────────────────┘
                                            │  actor = "human" | "agent:<name>"  →  TrustPolicy
            ┌───────────────────────────────▼──────────────────────────────┐
            │ Registry   (brains/, deciders/, audit.jsonl, snapshots)        │
            │   Brain  ── ingest → chunk → self-supervised training           │
            │     │        ask / search / probe / feedback / teach / sleep    │
            │   Decider ── few-shot labels → prototypes + tiny net → decide   │
            └───────────────────────────────┬──────────────────────────────┘
            ┌───────────────────────────────▼──────────────────────────────┐
            │ core/  (pure NumPy, no framework)                              │
            │  text.py      normalise · tokens · hashed word+char n-grams     │
            │  autograd.py  reverse-mode autodiff (gradient-checked)          │
            │  model.py     ReflexNet: shared trunk → dense | spikes | gate    │
            │  index.py     inverted index · dense index · BM25 · fusion     │
            │  selfsup.py   pseudo-queries from your own text (no labels)     │
            │  plasticity   Hebbian FastSynapses · ReflexMemory (episodic)   │
            │  answer.py    best-sentence extraction + highlights             │
            └──────────────────────────────────────────────────────────────┘
            storage.py: JSON + NPZ only (no pickle → loading a brain can't run code)
```

**One query, one forward pass:**

```
query ─► featurise ─► trunk ─┬─► dense head ───────────────► cosine over chunk vectors
                             ├─► spike head: log1p(ReLU) ──► prune (jump threshold) ─► inverted index
                             └─► hazard gate ─► P(stop at level 0,1,2) ─► K and fusion weights
   ReflexMemory (episodic neurons) is checked first → if a trusted memory fires: answer instantly (reflex-arc)
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 3. Install

```bash
git clone <this repo> rakhsh && cd rakhsh
pip install -e ".[formats]"        # numpy + optional PDF/Word/Excel readers
# or with no install at all:  PYTHONPATH=. python -m rakhsh --help
```

One-click: `./start.sh` (macOS/Linux) or `start.bat` (Windows) — creates a venv and opens the UI.

Requires Python ≥ 3.10 and NumPy. Nothing else is mandatory.

## 4. Quick start

### 4.1 Web UI
```bash
rakhsh serve                 # → http://127.0.0.1:8765
```
Drop files (book, CSV, SQLite, Excel, PDF, Word, HTML, JSON) or paste text → the brain trains itself → chat.
Each answer shows the gate level, firing spikes (orange = words you didn't type), confidence, latency, and
👍 / 👎 / "this one" buttons. Switch or create brains from the sidebar. "Sleep" consolidates memory.

### 4.2 Python SDK
```python
from rakhsh import Registry

reg = Registry("~/.rakhsh")
brain = reg.brain("cinema")
brain.ingest("cinema_book.txt", open("samples/cinema_book.txt", "rb").read())
brain.wait()                                       # background self-supervised training

r = brain.ask("a film about a father with Alzheimer's and a court case")
print(r["answer"]["text"], r["answer"]["locator"], r["meta"]["level_name"])

brain.feedback(r["id"], "up")                      # 👍  next time: reflex-arc
brain.teach("tonight's pick?", answer="Watch 'A Separation' tonight.")   # brand-new knowledge
brain.save()
```
More: [`examples/sdk_quickstart.py`](examples/sdk_quickstart.py), [`examples/llm_router_teacher_student.py`](examples/llm_router_teacher_student.py).

### 4.3 CLI
```bash
rakhsh ingest books samples/cinema_book.txt samples/movies.csv
rakhsh ask books "who directed A Separation"
rakhsh list
rakhsh decider create router --labels no-llm,small-llm,large-llm --mode ordinal
rakhsh decider teach router --label no-llm "hi" "thanks"
rakhsh decider run router "translate this paragraph"
rakhsh bench                      # reproduce the research benchmark
```
Global flags: `--home DIR` (or `RAKHSH_HOME`), `--trust claude-ai=0.8`, `--agent-trust 0.5`, `-v`.

### 4.4 MCP (Claude Desktop, Cursor, any MCP client)

**stdio** — `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "rakhsh": {
      "command": "rakhsh",
      "args": ["--trust", "claude-ai=0.8", "mcp", "--allow-dir", "/Users/YOU/Documents"],
      "env": { "RAKHSH_HOME": "/Users/YOU/.rakhsh" }
    }
  }
}
```
**Streamable HTTP** — run `rakhsh serve` and point the client at `http://127.0.0.1:8765/mcp`
(see [`examples/cursor_mcp.json`](examples/cursor_mcp.json)). Compatibility is tested with the official
MCP Python SDK: [`examples/mcp_sdk_client.py`](examples/mcp_sdk_client.py).

| tool | purpose |
|---|---|
| `rakhsh_overview` | all brains + deciders, sizes, status |
| `brain_ingest` | add a file path (inside `--allow-dir`), raw text, or base64 bytes |
| `brain_status` | training state, sources, learning stats |
| `brain_ask` | quoted answer + source/locator + decision trace + alternatives + `message_id` |
| `brain_search` | ranked hits only |
| `brain_probe` | decision only: which spikes fire, which level the gate picks |
| `brain_feedback` | `up` / `down` / `pick` a chunk for a `message_id` |
| `brain_teach` | teach *question → answer* (new knowledge if the answer isn't in the data) |
| `brain_sleep` | consolidate episodic memory into slow weights |
| `brain_snapshot` | versioned snapshot of a brain |
| `brain_remove_source` | forget a source |
| `decider_create` / `decider_teach` / `decider_decide` | the micro decision layer |

Resources: `rakhsh://overview`, `rakhsh://audit`, `rakhsh://brain/{name}`.
Prompts: `rakhsh_router` (use Rakhsh before calling an LLM), `rakhsh_teacher` (an LLM teaches a Decider).

### 4.5 REST
```
GET  /api/overview
GET  /api/b/<brain>/state
POST /api/b/<brain>/{upload|text|ask|search|probe|feedback|teach|sleep|remove|reset|snapshot|restore|delete}
POST /api/d/<decider>/{create|teach|decide|info}
POST /mcp                              (JSON-RPC, Mcp-Session-Id)
headers: Authorization: Bearer <token>   (when started with --token)
         X-Rakhsh-Actor: agent:<name>    (otherwise the caller is "human")
```
```bash
curl -s localhost:8765/api/b/books/ask -H 'Content-Type: application/json' -d '{"q":"who directed A Separation"}'
```

## 5. Decision layer (`Decider`)

```python
router = reg.create_decider("llm-router", ["no-llm", "small-llm", "large-llm"], mode="ordinal")
router.teach([{"text": "hi", "label": "no-llm"}, {"text": "summarise this email", "label": "small-llm"},
              {"text": "design a distributed system", "label": "large-llm"}], actor="agent:teacher")
router.decide("translate this paragraph")
# {'label': 'small-llm', 'confidence': 0.98, 'abstained': False, 'evidence': [['ترجمه', 0.17], ...], 'ms': 0.4}
```
* **categorical** or **ordinal** (ordered labels). In ordinal mode, when unsure it *escalates* to the safer (higher) label.
* prototype memory (works from 1 example per label) blended with a tiny neural net that improves with more data.
* `abstained=True` below `abstain_below` confidence — the caller should ask a human/LLM, then `teach`.
* `evidence` = occlusion attribution: how much each word contributed.

**Teacher → student:** a large LLM labels a few cases once (via `decider_teach`); the Decider then makes every
future decision locally in microseconds. This is distillation as a workflow.

## 6. Trust, collaboration and safety

* Every call has an **actor**: `human` (UI/CLI/SDK) or `agent:<name>` (MCP `clientInfo.name`, or `X-Rakhsh-Actor`).
* `TrustPolicy`: human = 1.0, unknown agent = 0.5 (`--agent-trust`), per-agent `--trust name=0.8`.
  * feedback strength is scaled by trust;
  * an episodic reflex neuron fires only after **accumulated trust ≥ 1.0** (one human, or two agreeing 0.8 agents);
  * slow (gradient) learning only from actors with trust ≥ 0.75; a 0.1 "spam" agent can't rewrite the brain.
* **Audit log** (`audit.jsonl`): who ingested / taught / gave feedback, when.
* **Snapshots**: `brain_snapshot` / `restore` — roll back a bad teaching session.
* **No pickle**: state is JSON + NPZ (`allow_pickle=False`), so loading a shared brain cannot execute code.
* HTTP binds to `127.0.0.1`; non-local binds require `--token`; browser `Origin` is checked (anti-CSRF/DNS-rebinding);
  MCP file access is restricted to `--allow-dir` directories; body size is capped.

## 7. Storage layout
```
~/.rakhsh/
  audit.jsonl                                  who did what, when (append-only)
  brains/<name>/  meta.json      config, sources, stats, status
                  chunks.jsonl   your data, chunked (source + locator)
                  vocab.json     spike-neuron vocabulary
                  model.npz      ReflexNet weights (NumPy arrays only)
                  plastic.json   Hebbian synapses + episodic reflex neurons (with trust strength)
                  snapshots/<id>/  versioned copies of the files above
  deciders/<name>/ decider.json  labels, examples, stats
                   net.npz       tiny decision network
```

## 8. Benchmarks (honest, small)

Research set: 36 Persian film documents, 108 held-out queries in 3 difficulty levels
(navigational / keyword / abstract-paraphrase). Reproduce with `rakhsh bench`. Numbers vary ±0.02 between runs/seeds.

| system | R@1 | R@5 | MRR@10 | abstract-query R@1 |
|---|---|---|---|---|
| BM25 (classic baseline) | 0.870 | 0.944 | 0.898 | 0.611 |
| Rakhsh sparse spikes only | **0.917** | **0.963** | **0.936** | **0.750** |
| Rakhsh dense only | 0.898 | 0.954 | 0.921 | 0.722 |
| Rakhsh adaptive (gate) | 0.898 | 0.963 | 0.924 | 0.694 |

Gate accuracy 0.80 · average K = 4.7 · ~17 active spikes of ~620 · p50 latency 0.44 ms (BM25: 0.03 ms).

Scale test (single CPU core): a ~250 000-word book trains in ~95 s; `ask` ≈ 3.5 ms; a feedback update ≈ 1 s
(in the background). Earlier runs on the same set reached adaptive R@1 = 0.917 / abstract 0.750.

**What this does *not* prove:** the dataset is tiny and synthetic-ish; there is no comparison with large pretrained
embedding models (which would likely win on open-domain semantics). The claim is narrower: *a tiny, self-trained,
explainable, CPU-only model can beat BM25 on your own data and keeps improving from feedback.*

## 9. Limitations
* **No pretrained world knowledge.** It knows only what you give it. Synonyms it has never seen in your data won't match.
* **Extractive only.** It quotes; it doesn't write. Pair it with an LLM (via MCP) when you need prose.
* Training is CPU NumPy: fine up to a few hundred thousand words per brain; millions of documents need sharding.
* The gate is trained from self-supervised pseudo-queries — real feedback makes it much better.

## 10. Project layout & tests
```
rakhsh/core/        math & models (no I/O)          rakhsh/adapters/   cli · http · mcp · web UI
rakhsh/ingest/      loaders + chunking               rakhsh/research/   original research agent + bench
rakhsh/brain.py     learning knowledge brain         rakhsh/decide.py   micro decision layer
rakhsh/registry.py  many brains, audit, snapshots    rakhsh/storage.py  safe persistence
tests/              37 unit/integration tests        docs/course/       10-session course (Persian)
```
```bash
python -m unittest discover -s tests -t .     # ~15 s
```

## 11. Roadmap
* streaming SSE for MCP-over-HTTP · multi-brain federated search · int8 quantised trunk
* optional plug-in for pretrained embeddings (keep the gate/spikes/plasticity) · ONNX export of ReflexNet
* active learning: the brain asks *you* when `unsure=true`

## 12. Learn how it works
[`docs/course/`](docs/course/README.md) — a 10-session course (in Persian) that builds every idea in Rakhsh from
zero: vectors → tokenisation → BM25 → a neuron → autograd → contrastive learning → spikes → the hazard gate →
plasticity → systems engineering. Each session has runnable code, exercises and a challenge; solutions included.

