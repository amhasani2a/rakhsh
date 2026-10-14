# Changelog

## 1.0.0 — Rakhsh
* Renamed and re-engineered from the *Reflex-based Search Agent* research prototype.
* Layered package: `core` (pure math) · `ingest` · `brain` · `decide` · `registry` · `storage` · `adapters`.
* MCP server (stdio + Streamable HTTP) with tools, resources and prompts; verified with the official MCP SDK.
* REST API + multi-brain web UI; token auth, Origin check, directory allow-list.
* `Decider`: few-shot categorical/ordinal micro decision layer with abstention and evidence.
* Trust-aware collaborative learning (human / agents), audit log, snapshots, pickle-free storage.
* 37 tests; reproducible benchmark (`rakhsh bench`); 10-session course in Persian.

## 0.x — Reflex Search / ReflexBrain (research)
* Single-forward fused heads (dense + sparse spikes + hazard gate), Persian film benchmark, universal ingestion,
  Hebbian + episodic plasticity, sleep consolidation, local chat UI.
