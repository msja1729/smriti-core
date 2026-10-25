from fakes import FakeEmbedder, FakeJudge
from smriti.models.cache import DiskCache
from smriti.models.hosted import CachedEmbedder, CachedJudge


def test_disk_cache_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    assert c.get("ns", "k") is None
    c.put("ns", "k", {"a": [1, 2]})
    assert DiskCache(tmp_path).get("ns", "k") == {"a": [1, 2]}
    assert c.get("other", "k") is None


def test_cached_judge_calls_once_across_instances(tmp_path):
    inner = FakeJudge(unsafe=True)
    assert CachedJudge(inner, DiskCache(tmp_path), "m").is_unsafe("t")
    assert CachedJudge(inner, DiskCache(tmp_path), "m").is_unsafe("t")
    assert inner.seen == ["t"]


def test_cached_embedder_only_embeds_misses(tmp_path):
    inner = FakeEmbedder()
    emb = CachedEmbedder(inner, DiskCache(tmp_path), "m")
    first = emb.embed(["a", "b"])
    calls = inner.calls
    assert emb.embed(["b", "a", "c"])[:2] == [first[1], first[0]]
    assert inner.calls == calls + 1          # one batch for the single miss "c"
    assert emb.embed([]) == []
