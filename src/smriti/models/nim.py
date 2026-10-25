"""Nemotron on NIM (OpenAI-compatible). Validates tool calls and retries once with the
error (SPEC §17); a second bad reply raises ToolCallError, which the agent treats as a
failed turn."""
import json
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class NimConfig:
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 60.0
    no_think: bool = True          # "/no_think" in the system message: no reasoning trace


@dataclass(frozen=True)
class LlmToolCall:
    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class LlmReply:
    content: str | None
    tool_calls: tuple[LlmToolCall, ...]


class ToolCallError(Exception): ...


class _BadToolCall(Exception): ...


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LlmReply: ...


class NimChat:
    def __init__(self, config: NimConfig, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.base_url, timeout=config.timeout_s,
                            api_key=os.environ.get("NIM_API_KEY", "not-needed"))
        self.config = config
        self.client = client

    def chat(self, messages: list[dict], tools: list[dict]) -> LlmReply:
        messages = self._with_no_think(messages)
        names = {t["function"]["name"] for t in tools}
        try:
            return self._call(messages, tools, names)
        except _BadToolCall as e:
            retry = messages + [{"role": "user", "content": (
                f"Your last tool call was invalid: {e}. "
                "Call one of the available tools again with valid JSON arguments.")}]
            try:
                return self._call(retry, tools, names)
            except _BadToolCall as e2:
                raise ToolCallError(str(e2)) from e2

    def _with_no_think(self, messages: list[dict]) -> list[dict]:
        if not self.config.no_think:
            return list(messages)
        if messages and messages[0]["role"] == "system":
            first = {**messages[0], "content": "/no_think\n" + messages[0]["content"]}
            return [first, *messages[1:]]
        return [{"role": "system", "content": "/no_think"}, *messages]

    def _call(self, messages: list[dict], tools: list[dict], names: set[str]) -> LlmReply:
        kwargs = dict(model=self.config.model, messages=messages,
                      temperature=self.config.temperature, max_tokens=self.config.max_tokens)
        if tools:
            kwargs["tools"] = tools
        msg = self.client.chat.completions.create(**kwargs).choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            name = tc.function.name
            if name not in names:
                raise _BadToolCall(f"unknown tool {name!r}")
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                raise _BadToolCall(f"arguments are not valid JSON ({e.msg})") from e
            if not isinstance(args, dict):
                raise _BadToolCall("arguments must be a JSON object")
            calls.append(LlmToolCall(tc.id, name, args))
        return LlmReply(msg.content, tuple(calls))


SUMMARY_PROMPT = (
    "Summarize the following notes into a short, factual summary. "
    "Keep names, dates, amounts and addresses exactly as written.")


class NimSummarizer:
    def __init__(self, chat: ChatModel):
        self.chat = chat

    def summarize(self, texts: list[str]) -> str:
        notes = "\n".join(f"- {t}" for t in texts)
        reply = self.chat.chat([{"role": "system", "content": SUMMARY_PROMPT},
                                {"role": "user", "content": notes}], [])
        return (reply.content or "").strip()
