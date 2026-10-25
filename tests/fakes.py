"""Deterministic stand-ins for model-backed components (no GPU, no network)."""
import hashlib
import re

from smriti.models.pii import PiiResult

DIM = 64


class FakeEmbedder:
    """Bag-of-words hashed into DIM buckets: texts sharing words are near each other."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedder down")
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM] += 1.0
            vec[0] += 1e-3                      # never the zero vector
            out.append(vec)
        return out


class FakeJudge:
    def __init__(self, unsafe: bool = False, fail: bool = False):
        self.unsafe = unsafe
        self.fail = fail
        self.seen: list[str] = []

    def is_unsafe(self, text: str) -> bool:
        self.seen.append(text)
        if self.fail:
            raise RuntimeError("judge down")
        return self.unsafe


class FakePii:
    """Redacts phone numbers; reports emails without redacting (mirrors the default policy)."""

    PHONE = re.compile(r"\+?\d[\d -]{7,}\d")
    EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")

    def __init__(self, fail: bool = False):
        self.fail = fail

    def redact(self, text: str) -> PiiResult:
        if self.fail:
            raise RuntimeError("pii down")
        found = set()
        if self.PHONE.search(text):
            found.add("PHONE_NUMBER")
            text = self.PHONE.sub("[PHONE]", text)
        if self.EMAIL.search(text):
            found.add("EMAIL_ADDRESS")
        return PiiResult(text, frozenset(found))


class FakeSummarizer:
    def __init__(self, output: str | None = None, fail: bool = False):
        self.output = output
        self.fail = fail

    def summarize(self, texts: list[str]) -> str:
        if self.fail:
            raise RuntimeError("summarizer down")
        return self.output if self.output is not None else " / ".join(texts)
