"""Hosted clients against a fake OpenAI client (no network)."""
from types import SimpleNamespace as NS

import pytest

from fakes import FakeEmbedder
from smriti.models.cache import DiskCache
from smriti.models.hosted import CachedEmbedder, HostedConfig, NvidiaEmbedder, SafetyGuardJudge

CFG = HostedConfig(base_url="http://x/v1", model="m")


class FakeClient:
    def __init__(self, guard_reply=None):
        self.guard_reply = guard_reply
        self.embed_requests = []
        self.chat = NS(completions=NS(create=self._chat))
        self.embeddings = NS(create=self._embed)

    def _chat(self, **kw):
        return NS(choices=[NS(message=NS(content=self.guard_reply))])

    def _embed(self, **kw):
        self.embed_requests.append(kw)
        n = len(kw["input"])
        # Return out of order to check index sorting.
        return NS(data=[NS(index=i, embedding=[float(i), 1.0]) for i in reversed(range(n))])


@pytest.mark.parametrize("reply,unsafe", [
    ('{"User Safety": "safe"} ', False),
    ('{"User Safety": "unsafe", "Safety Categories": "Fraud/Deception"}', True),
])
def test_guard_parses_verdict(reply, unsafe):
    assert SafetyGuardJudge(CFG, FakeClient(reply)).is_unsafe("t") is unsafe


@pytest.mark.parametrize("reply", ["not json", '{"other": 1}', '{"User Safety": "maybe"}'])
def test_guard_garbage_raises(reply):
    # The gateway turns this into a rejected write (fail closed).
    with pytest.raises(Exception):
        SafetyGuardJudge(CFG, FakeClient(reply)).is_unsafe("t")


def test_embedder_batches_orders_and_passes_input_type():
    client = FakeClient()
    vecs = NvidiaEmbedder(CFG, batch_size=2, client=client).embed(["a", "b", "c"], "query")
    assert vecs == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert [len(r["input"]) for r in client.embed_requests] == [2, 1]
    assert all(r["extra_body"]["input_type"] == "query" for r in client.embed_requests)


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        NvidiaEmbedder(CFG)


def test_cache_separates_query_and_passage(tmp_path):
    inner = FakeEmbedder()
    emb = CachedEmbedder(inner, DiskCache(tmp_path), "m")
    emb.embed(["a"], "passage")
    emb.embed(["a"], "query")
    emb.embed(["a"], "query")
    assert inner.input_types == ["passage", "query"]
