# SMRITI — Development Environment

Facts only. This describes the setup as it exists now. Do not change infrastructure without asking.

## Machines

| Machine | Role | Notes |
|---|---|---|
| Laptop (WSL Ubuntu) | Write code, run unit tests, run Claude Code | Repo at `~/smriti`, venv at `~/smriti/.venv` |
| GitHub `msja1729/smriti` (private) | Source of truth | Laptop pushes; A100 only pulls (read-only deploy key) |
| Brev A100 `smriti-a100` | Runs models, DB, GPU jobs, Jupyter | 1× A100 80GB PCIe, 12 CPUs, 120 GB RAM |

Claude Code runs on the laptop only. It must never SSH into the A100, run docker there, or read secrets files.

## Services (on the A100, bound to 127.0.0.1 only)

| Service | Container | Port | Details |
|---|---|---|---|
| Agent LLM | `agent-llm` | 8000 | NVIDIA NIM serving `nvidia/nvidia-nemotron-nano-9b-v2`, OpenAI-compatible API at `/v1` |
| Memory store | `smriti-db` | 5432 | Postgres 16 + pgvector 0.8.7, user `postgres` |

From the laptop, these are reachable only while a tunnel is open:

```bash
brev port-forward smriti-a100 -p 8000:8000 -p 5432:5432
```

Then use `http://localhost:8000/v1` and `localhost:5432` exactly as if they were local.

## Calling the model

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="nvidia/nvidia-nemotron-nano-9b-v2",
    messages=[{"role": "system", "content": "/no_think"},
              {"role": "user", "content": "..."}],
    max_tokens=256,
)
```

- `/no_think` in the system message: answer directly, no reasoning trace. Omit it to get reasoning.
- Tool calling is enabled (OpenAI `tools=` format).
- Limits: max context 32,768 tokens; max 32 concurrent requests.

## Model server limits (do not exceed)

- GPU memory cap for the model: 50% (~34 GB used). ~46 GB free for GRPO and classifiers.
- Server config lives in `scripts/start_agent_llm.sh`. Required flags include `HWLOC_COMPONENTS=-x86` (VM crash fix) and `NIM_MAX_NUM_SEQS=32` (memory fix).

## Project rules

- Models: NVIDIA open models only (Nemotron, NemoGuard, NeMo Retriever embeddings). No Qwen, Llama, DeBERTa, OpenAI, etc. inside SMRITI.
- Package code lives in `src/smriti/`; import as `from smriti.gateway import ...`.
- `tests/unit/` must run on CPU with no model and no database (use fakes).
- `tests/integration/` may use the model and DB through the tunnel; skip cleanly if `localhost:8000` is unreachable.
- Pass/fail decisions are made by deterministic code, never by an LLM.
- Secrets never go in the repo. `.env` files are gitignored.

## Where things run

| Task | Where |
|---|---|
| Writing code, unit tests | Laptop |
| Tests needing model/DB | Laptop via tunnel |
| GRPO training, full benchmark | A100, inside tmux (run by the human) |
| Notebooks, demo | Jupyter on the A100, kernel "SMRITI (venv)" |

## Python environments

- Laptop: `~/smriti/.venv` (Python 3.12). Activate with `source .venv/bin/activate`.
- A100: `~/venvs/smriti`, auto-activated on login. Project installed editable (`pip install -e`).
- Installed so far: `openai`, `pytest`, `psycopg[binary]`, `pgvector`, `ipykernel` (A100). Heavy training libs (vllm, trl) are not installed yet.