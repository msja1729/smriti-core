# SMRITI — Environment

What SMRITI needs to run. Unit tests need none of it.

## Services

| Service | Default address | What it is |
|---|---|---|
| Agent LLM | `http://localhost:8000/v1` | An OpenAI-compatible endpoint serving `nvidia/nvidia-nemotron-nano-9b-v2` (e.g. NVIDIA NIM on a GPU). Tool calling must be enabled |
| Memory store | `localhost:5432` | PostgreSQL 16 with the pgvector extension |
| Hosted models | `https://integrate.api.nvidia.com/v1` | NVIDIA API Catalog: Safety Guard judge and Nemotron embeddings |

If the GPU and database run on a remote machine, forward ports 8000 and 5432 to your
laptop (e.g. an SSH tunnel) and use the addresses above unchanged.

## Secrets (`.env` in the repo root, never committed)

```
NVIDIA_API_KEY=nvapi-...                 # from build.nvidia.com
SMRITI_PG_DSN="host=localhost port=5432 user=postgres dbname=postgres password=..."
```

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

`/no_think` in the system message makes Nemotron Nano answer directly without a reasoning trace.

## Python

Python 3.12, a virtual environment in `.venv`, and `pip install -e ".[dev]"`.
Lean 4 (via `elan`) is optional: only for `lean/` and the proof checks.

## Rules

- Package code lives in `src/smriti/`; import as `from smriti.gateway import ...`.
- `tests/unit/` runs on CPU with no model and no database (fakes).
- `tests/integration/` uses the model and database and skips cleanly if they are unreachable.
- Pass/fail decisions are made by deterministic code, never by an LLM.
- Secrets never go in the repo.
