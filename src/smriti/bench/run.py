"""Run the benchmark from one YAML config (SPEC §15, §18).

    python -m smriti.bench.run --config src/smriti/bench/configs/demo.yaml [--limit N]

Writes results/<run_id>/{config.yaml, episodes.jsonl, metrics.json}. Re-running the same
config resumes: episodes already in episodes.jsonl without an error are skipped.
"""
import argparse
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from smriti.attacks import ATTACKS
from smriti.bench.arms import ARMS
from smriti.bench.episodes import Components, EpisodeSettings, EpisodeSpec, run_episode
from smriti.bench.metrics import aggregate


@dataclass(frozen=True)
class BenchConfig:
    run_id: str
    seeds: list[int]
    personas: list[str]
    attacks: list[str]
    arms: list[str]
    agent_model: str
    agent_base_url: str
    judge_model: str
    embed_model: str
    hosted_base_url: str = "https://integrate.api.nvidia.com/v1"
    ablations: list[str] = ()
    clean_episodes: bool = True
    clean_sessions_between: int = 3
    turns_per_clean_session: int = 2
    consolidate_every: int = 10
    quota_low_writes_per_session: int = 20
    workers: int = 8
    results_dir: str = "results"
    cache_dir: str = ".cache/smriti"

    def __post_init__(self):
        unknown = [a for a in self.attacks if a not in ATTACKS]
        unknown += [a for a in [*self.arms, *self.ablations] if a not in ARMS]
        if unknown:
            raise ValueError(f"unknown attacks/arms: {unknown}")
        if not re.fullmatch(r"[a-z0-9_]+", self.run_id):
            raise ValueError("run_id must be lowercase letters, digits and _")

    @property
    def settings(self) -> EpisodeSettings:
        return EpisodeSettings(self.clean_sessions_between, self.turns_per_clean_session,
                               self.consolidate_every, self.quota_low_writes_per_session)


def load_config(path: str | Path) -> BenchConfig:
    raw = yaml.safe_load(Path(path).read_text())
    known = {f.name for f in fields(BenchConfig)}
    extra = set(raw) - known
    if extra:
        raise ValueError(f"unknown config keys: {sorted(extra)}")
    return BenchConfig(**raw)


def episode_grid(cfg: BenchConfig) -> list[EpisodeSpec]:
    arms = [*cfg.arms, *cfg.ablations]
    attacks = [*cfg.attacks, *(["clean"] if cfg.clean_episodes else [])]
    return [EpisodeSpec(p, a, arm, s)
            for p in cfg.personas for a in attacks for arm in arms for s in cfg.seeds]


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal KEY=VALUE loader; never overrides variables already set."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*", line)
        if m and not line.lstrip().startswith("#"):
            os.environ.setdefault(m.group(1), m.group(2).strip('"').strip("'"))


class _LockedPii:
    """Presidio/spaCy objects are shared across worker threads; serialize access."""

    def __init__(self, inner):
        self.inner, self.lock = inner, threading.Lock()

    def redact(self, text):
        with self.lock:
            return self.inner.redact(text)


def build_components(cfg: BenchConfig) -> Components:
    from smriti.models.cache import DiskCache
    from smriti.models.hosted import (
        CachedEmbedder, CachedJudge, HostedConfig, NvidiaEmbedder, SafetyGuardJudge)
    from smriti.models.nim import NimChat, NimConfig, NimSummarizer
    from smriti.models.pii import PresidioRedactor

    cache = DiskCache(cfg.cache_dir)
    llm = NimChat(NimConfig(base_url=cfg.agent_base_url, model=cfg.agent_model))
    judge = CachedJudge(SafetyGuardJudge(HostedConfig(cfg.hosted_base_url, cfg.judge_model)),
                        cache, cfg.judge_model)
    embedder = CachedEmbedder(NvidiaEmbedder(HostedConfig(cfg.hosted_base_url, cfg.embed_model)),
                              cache, cfg.embed_model)
    return Components(llm, NimSummarizer(llm), embedder, judge, _LockedPii(PresidioRedactor()))


def pg_store_factory(dsn: str, run_id: str, embed_dim: int):
    from smriti.store.store import PgStore

    def make():
        store = PgStore(dsn, schema=f"ep_{run_id}_{uuid.uuid4().hex[:8]}", embed_dim=embed_dim)
        store.init_schema()
        return store
    return make


def run(cfg: BenchConfig, parts: Components, make_store, limit: int | None = None,
        out_dir: Path | None = None, log=print) -> dict:
    out_dir = out_dir or Path(cfg.results_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    done = {}
    if episodes_path.exists():
        for line in episodes_path.read_text().splitlines():
            r = json.loads(line)
            if r["error"] is None:
                done[r["key"]] = r

    grid = episode_grid(cfg)[:limit]
    todo = [s for s in grid if s.key not in done]
    log(f"{len(grid)} episodes in grid, {len(done)} already done, {len(todo)} to run")
    lock = threading.Lock()
    started = time.time()

    def one(spec: EpisodeSpec) -> dict:
        store = make_store()
        try:
            return run_episode(spec, cfg.settings, parts, store).to_json()
        finally:
            for cleanup in ("drop_schema", "close"):
                if hasattr(store, cleanup):
                    getattr(store, cleanup)()

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(one, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with lock, episodes_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            status = r["error"] or f"attack={r['attack_succeeded']} task={r['task_succeeded']}"
            log(f"[{i}/{len(todo)} {time.time() - started:6.0f}s] {r['key']}: {status}")

    # Last line per episode wins (done episodes are never rerun, so this is the latest attempt).
    grid_keys = {s.key for s in grid}
    latest: dict[str, dict] = {}
    for line in episodes_path.read_text().splitlines():
        r = json.loads(line)
        if r["key"] in grid_keys:
            latest[r["key"]] = r
    results = list(latest.values())

    metrics = aggregate(results)
    metrics["models"] = {"agent": cfg.agent_model, "judge": cfg.judge_model,
                         "embed": cfg.embed_model}
    metrics["episodes"] = len(results)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="run only the first N episodes of the grid")
    ap.add_argument("--dry-run", action="store_true", help="print the grid size and exit")
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = load_config(args.config)
    if args.dry_run:
        print(f"{len(episode_grid(cfg))} episodes")
        return
    out_dir = Path(cfg.results_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(Path(args.config).read_text())

    parts = build_components(cfg)
    embed_dim = len(parts.embedder.embed(["dimension probe"], "query")[0])
    dsn = os.environ.get("SMRITI_PG_DSN", "host=localhost port=5432 user=postgres dbname=postgres")
    metrics = run(cfg, parts, pg_store_factory(dsn, cfg.run_id, embed_dim), args.limit, out_dir)
    print(json.dumps({"asr": metrics["asr"]}, indent=2))
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
