"""Deterministic stand-ins for model-backed components (no GPU, no network)."""
import re

from smriti.models.hash_embed import HashEmbedder
from smriti.models.nim import LlmReply
from smriti.models.pii import PiiResult


class FakeEmbedder(HashEmbedder):
    """HashEmbedder that counts calls and can be made to fail."""

    def __init__(self, fail: bool = False):
        super().__init__(dim=64)
        self.fail = fail
        self.calls = 0
        self.input_types: list[str] = []

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        self.calls += 1
        self.input_types.append(input_type)
        if self.fail:
            raise RuntimeError("embedder down")
        return super().embed(texts, input_type)


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


class ScriptedChat:
    """Returns the given replies in order; records the messages it was sent."""

    def __init__(self, replies: list[LlmReply | Exception]):
        self.replies = list(replies)
        self.sent: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> LlmReply:
        self.sent.append([dict(m) for m in messages])
        reply = self.replies.pop(0) if self.replies else LlmReply("done", ())
        if isinstance(reply, Exception):
            raise reply
        return reply
