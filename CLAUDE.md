# SMRITI — Secure Memory Runtime for AI agents (AEGIS 2026, PS4)
Pillars: provenance labels, write gate, read gate (autoescaping), immune auditor, Inception Arena (GRPO attacker).
Rules:
- Fail closed: if a check errors, reject the write / read only high-trust memory.
- Low-trust memory may inform, never instruct (render as quoted data).
- Summaries inherit the MIN trust of their inputs.
- Models: NVIDIA open models only (Nemotron, NemoGuard, NeMo Retriever).
- LLM endpoint: http://localhost:8000/v1 (OpenAI-compatible). Postgres: localhost:5432.
- Pass/fail decided by deterministic oracle code, never an LLM.
- Unit tests must run on CPU without the model.
