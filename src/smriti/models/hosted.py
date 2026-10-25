"""Hosted NVIDIA models on build.nvidia.com (OpenAI-compatible): the Safety Guard judge and
the Nemotron embedder, plus caching wrappers so benchmark reruns cost nothing.

Retries with exponential backoff on 429/5xx/timeouts come from the OpenAI client
(`max_retries`). Anything still failing raises, and the gateway fails closed.
"""
import json
import os
from dataclasses import dataclass
from typing import Protocol

from smriti.models.cache import DiskCache


class Judge(Protocol):
    def is_unsafe(self, text: str) -> bool: ...


class Embedder(Protocol):
    # input_type: "passage" for stored memories, "query" for searches (asymmetric models).
    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]: ...


@dataclass(frozen=True)
class HostedConfig:
    base_url: str
    model: str
    api_key_env: str = "NVIDIA_API_KEY"
    timeout_s: float = 60.0
    max_retries: int = 6


def _client(config: HostedConfig):
    from openai import OpenAI

    key = os.environ.get(config.api_key_env)
    if not key:
        raise RuntimeError(f"{config.api_key_env} is not set (put it in .env)")
    return OpenAI(base_url=config.base_url, api_key=key, timeout=config.timeout_s,
                  max_retries=config.max_retries)


class SafetyGuardJudge:
    """Llama-3.1-Nemotron-Safety-Guard: replies JSON like {"User Safety": "unsafe", ...}."""

    def __init__(self, config: HostedConfig, client=None):
        self.config = config
        self.client = client or _client(config)

    def is_unsafe(self, text: str) -> bool:
        r = self.client.chat.completions.create(
            model=self.config.model, messages=[{"role": "user", "content": text}],
            max_tokens=128, temperature=0)
        verdict = json.loads(r.choices[0].message.content)["User Safety"].strip().lower()
        if verdict not in ("safe", "unsafe"):
            raise ValueError(f"unexpected judge verdict {verdict!r}")
        return verdict == "unsafe"


class NvidiaEmbedder:
    def __init__(self, config: HostedConfig, batch_size: int = 32, client=None):
        self.config = config
        self.batch_size = batch_size
        self.client = client or _client(config)

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            r = self.client.embeddings.create(
                model=self.config.model, input=texts[i:i + self.batch_size],
                extra_body={"input_type": input_type, "truncate": "END"})
            out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        return out


class CachedJudge:
    def __init__(self, inner: Judge, cache: DiskCache, model: str):
        self.inner, self.cache, self.namespace = inner, cache, f"judge:{model}"

    def is_unsafe(self, text: str) -> bool:
        hit = self.cache.get(self.namespace, text)
        if hit is not None:
            return hit
        verdict = bool(self.inner.is_unsafe(text))
        self.cache.put(self.namespace, text, verdict)
        return verdict


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: DiskCache, model: str):
        self.inner, self.cache, self.model = inner, cache, model

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        namespace = f"embed:{self.model}:{input_type}"
        out: list[list[float] | None] = [self.cache.get(namespace, t) for t in texts]
        misses = [i for i, v in enumerate(out) if v is None]
        if misses:
            fresh = self.inner.embed([texts[i] for i in misses], input_type)
            for i, vec in zip(misses, fresh, strict=True):
                self.cache.put(namespace, texts[i], vec)
                out[i] = vec
        return out  # type: ignore[return-value]
