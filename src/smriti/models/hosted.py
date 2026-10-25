"""Interfaces for the hosted NVIDIA judge and embedder, plus caching wrappers.

The real build.nvidia.com clients are added in M4; until then tests use fakes.
"""
from typing import Protocol

from smriti.models.cache import DiskCache


class Judge(Protocol):
    def is_unsafe(self, text: str) -> bool: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


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
        self.inner, self.cache, self.namespace = inner, cache, f"embed:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [self.cache.get(self.namespace, t) for t in texts]
        misses = [i for i, v in enumerate(out) if v is None]
        if misses:
            fresh = self.inner.embed([texts[i] for i in misses])
            for i, vec in zip(misses, fresh, strict=True):
                self.cache.put(self.namespace, texts[i], vec)
                out[i] = vec
        return out  # type: ignore[return-value]
