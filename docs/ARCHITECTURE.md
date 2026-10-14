# Rakhsh — Architecture

## Design principles
1. **One forward pass decides everything.** Meaning, keywords and "how hard to search" come from a single fused matmul.
2. **Core has no I/O.** `rakhsh/core` is pure NumPy math; it can be tested, ported (ONNX/Rust) or replaced in isolation.
3. **Adapters are thin.** Web UI, REST, MCP and CLI all call the same `Registry` → `Brain` / `Decider` methods.
4. **Learning never stops, but is gated by trust.** Fast plasticity is instant; slow learning needs trusted signals.
5. **Safe by default.** Local bind, optional token, Origin check, directory allow-list, no pickle, atomic writes, audit log.

## Layers
| layer | files | responsibility |
|---|---|---|
| core | `core/text.py` | normalisation (Arabic→Persian letters, digits, ZWNJ), tokens, stop-words, hashed word + char-n-gram features |
| | `core/autograd.py`, `core/nn.py` | ~200-line reverse-mode autodiff (`Tensor`, `matmul`, `tanh`, `relu`, `log1p`, `cross_entropy`, `hazard_nll`), `Linear`, Adam |
| | `core/model.py` | `ReflexNet`: trunk `tanh(xW1+b1)` → fused head `[W_dense | W_sparse | W_hazard]` |
| | `core/index.py` | inverted index for spikes, dense matrix index, BM25 baseline, score fusion, RRF |
| | `core/selfsup.py` | `PseudoQueries`: invents questions from your own chunks (spans, keyword bags, damaged copies, row fields) |
| | `core/plasticity.py` | `FastSynapses` (Hebbian), `ReflexMemory` (episodic neurons with trust strength) |
| | `core/answer.py` | extractive answer: best sentences + highlight segments |
| domain | `ingest/` | loaders (txt, md, pdf, docx, html, csv/tsv, xlsx, json/jsonl, sqlite) → `Chunk(source, locator, title, text)`; magic-byte detection; `register_loader` |
| | `brain.py` | lifecycle: ingest → train (background thread, atomic swap) → ask/search/probe → feedback/teach → sleep |
| | `decide.py` | `Decider`: hashed features → prototypes + small MLP; categorical/ordinal; abstain; occlusion evidence |
| | `registry.py` | many brains + deciders under one home; audit log; snapshots |
| | `storage.py` | JSON + NPZ persistence, atomic `os.replace`, snapshots |
| adapters | `adapters/mcp.py` | zero-dependency MCP (JSON-RPC 2.0): tools, resources, prompts; stdio transport |
| | `adapters/http.py` | stdlib `ThreadingHTTPServer`: web UI, REST, MCP Streamable-HTTP (`Mcp-Session-Id`) |
| | `adapters/cli.py` | `rakhsh serve | mcp | ingest | ask | list | decider | bench` |

## ReflexNet forward pass
```
x  = hashed features(query)                 (B, F)   sparse bag of word + char n-grams
h  = tanh(x·W1 + b1)                        (B, H)   shared trunk
z  = h·[W_d | W_s | W_g] + b                (B, D + V + L-1)   ONE matmul
dense  = normalize(z[:, :D])                         semantic vector
spikes = log1p(relu(z[:, D:D+V])) ; prune < 15% of max ; keep ≤ 32 ; + lexical skip for typed words
hazard = sigmoid(z[:, D+V:])  →  P(stop at l) = h_l · Π_{k<l}(1-h_k)  →  first l with h_l > 0.5 "fires"
```

## Training (self-supervised, no labels)
* pseudo-queries are generated from the user's chunks; the positive is the source chunk;
* loss = InfoNCE(dense) + InfoNCE(spikes · lexical doc vectors) + λ·FLOPS(spikes) (keeps spikes sparse);
* the **gate is labelled by outcome**: if the cheap spike index already finds the right chunk → label 1, else 2; level 0
  is reserved for the episodic reflex arc; it is fitted with a class-balanced discrete-hazard NLL;
* training runs on a deep copy in a background thread and is swapped in atomically; new data is indexed incrementally
  with the current model so it is searchable immediately.

## Learning after deployment
| signal | fast (instant) | slow |
|---|---|---|
| 👍 `up` | Hebbian: active spikes → chunk (+); episodic neuron strength += trust | replay buffer; gradient steps if trust ≥ 0.75; gate relabel |
| 👎 `down` | anti-Hebbian (−); weaken episodic neuron | replay with lower weight |
| `pick` chunk | (+) for picked, (−) for shown | replay |
| `teach` | adds a chunk if needed, episodic neuron | background retrain |
| `sleep()` | decay fast synapses | replay all episodes into slow weights, refit gate |

An episodic neuron answers at level 0 (`path = reflex-arc`) only when cosine ≥ 0.93 **and** accumulated trust ≥ 1.0.

## Concurrency & durability
* every public `Brain` method holds an `RLock`; queries during training use the previous model;
* the first query after creating a brain waits for the first model (outside the lock);
* writes are atomic (`*.tmp` → `os.replace`); `Registry.save_all()` on shutdown.

## Extending
* new file type: `rakhsh.register_loader(["ext"], fn(data: bytes, name: str, max_words: int) -> list[Chunk])`
* new transport: call `Registry` methods; pass `actor="agent:<name>"`
* new decision task: `Registry.create_decider(name, labels, mode)`
