# SMRITI — Secure Memory Runtime for AI agents
Core: provenance labels, write gate, read gate (autoescaping), action check, consolidation
with label inheritance, Immune Auditor (counterfactual replay + rollback), MemPoison-Bench.
Rules:
- Fail closed: if a check errors, reject the write / read only high-trust memory.
- Low-trust memory may inform, never instruct (render as quoted data).
- Summaries inherit the MIN trust of their inputs.
- Models: NVIDIA open models (Nemotron, NemoGuard, NeMo Retriever).
- LLM endpoint and Postgres come from config/.env (defaults: http://localhost:8000/v1, localhost:5432).
- Pass/fail decided by deterministic oracle code, never an LLM.
- Unit tests must run on CPU without the model.
