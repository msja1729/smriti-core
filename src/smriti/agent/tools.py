"""Mock tools (SPEC §10). They return fixtures, never touch the real world, and are the
only writers of tool_log."""
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from smriti.gateway.action_check import ActionDecision
from smriti.gateway.gateway import Gateway
from smriti.gateway.labels import ToolCall
from smriti.gateway.read_gate import RenderedContext

# Tools whose output becomes memory, and the provenance source it is written under.
READ_SOURCES: dict[str, str] = {"read_web": "tool:web", "read_email": "tool:email"}


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_STR = {"type": "string"}
_LIST = {"type": "array", "items": {"type": "string"}}

SCHEMAS: dict[str, dict] = {
    "read_web": _fn("read_web", "Fetch a web page.", {"url": _STR}, ["url"]),
    "read_email": _fn("read_email", "Read an email by id.", {"id": _STR}, ["id"]),
    "send_email": _fn("send_email", "Send an email.",
                      {"to": _LIST, "cc": _LIST, "subject": _STR, "body": _STR},
                      ["to", "subject", "body"]),
    "pay": _fn("pay", "Pay a payee by bank transfer.",
               {"payee": _STR, "account": _STR, "amount": {"type": "number"}},
               ["payee", "account", "amount"]),
    "deploy": _fn("deploy", "Deploy a git repository at a ref.",
                  {"repo": _STR, "ref": _STR}, ["repo", "ref"]),
    "calendar_add": _fn("calendar_add", "Add a calendar event.",
                        {"title": _STR, "when": _STR, "attendees": _LIST}, ["title", "when"]),
}


@dataclass(frozen=True)
class Fixtures:
    web: dict[str, str] = field(default_factory=dict)
    emails: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutcome:
    decision: str                  # "executed" | "confirm_denied" | "rejected_fail_closed"
    text: str                      # what the model sees as the tool result


def _addresses(value) -> list[str]:
    if value is None:
        return []
    items = re.split(r"[,;]", value) if isinstance(value, str) else value
    return [str(v).strip().lower() for v in items if str(v).strip()]


def _normalize(name: str, args: dict) -> dict:
    args = dict(args)
    missing = [k for k in SCHEMAS[name]["function"]["parameters"]["required"] if k not in args]
    if missing:
        raise ValueError(f"missing required argument(s): {', '.join(missing)}")
    if name == "send_email":
        args["to"] = _addresses(args.get("to"))
        args["cc"] = _addresses(args.get("cc"))
    if name == "calendar_add":
        args["attendees"] = _addresses(args.get("attendees"))
    return args


class ToolRunner:
    def __init__(
        self,
        gateway: Gateway,
        fixtures: Fixtures,
        confirm: Callable[[str, dict], bool],
        tools: tuple[str, ...] | list[str],
    ):
        self.gateway = gateway
        self.fixtures = fixtures
        self.confirm = confirm
        self.tools = tuple(tools)

    def schemas(self) -> list[dict]:
        return [SCHEMAS[t] for t in self.tools]

    def run(self, name: str, args: dict, ctx: RenderedContext, *,
            episode_id: str, session_id: str, step: int,
            turn_seq: int | None = None) -> ToolOutcome:
        def log(decision: str, logged_args: dict) -> None:
            self.gateway.store.log_tool_call(ToolCall(
                episode_id, session_id, step, name, logged_args, bool(ctx.tainted), decision,
                list(ctx.memory_ids), turn_seq))

        if name not in self.tools:
            log("rejected_fail_closed", dict(args))
            return ToolOutcome("rejected_fail_closed", f"Error: unknown tool {name!r}.")
        try:
            args = _normalize(name, args)
        except ValueError as e:
            log("rejected_fail_closed", dict(args))
            return ToolOutcome("rejected_fail_closed", f"Error: {e}.")

        if self.gateway.check_action(name, args, ctx) is ActionDecision.REQUIRE_CONFIRMATION:
            try:
                approved = bool(self.confirm(name, args))
            except Exception:
                approved = False   # fail closed
            if not approved:
                log("confirm_denied", args)
                return ToolOutcome("confirm_denied", "The user denied this action.")

        log("executed", args)
        return ToolOutcome("executed", self._execute(name, args))

    def _execute(self, name: str, a: dict) -> str:
        if name == "read_web":
            return self.fixtures.web.get(a["url"], f"404: {a['url']} not found.")
        if name == "read_email":
            return self.fixtures.emails.get(str(a["id"]), f"No email with id {a['id']}.")
        if name == "send_email":
            return f"Email sent to {', '.join(a['to'])}."
        if name == "pay":
            return f"Paid {a['amount']} to {a['payee']}."
        if name == "deploy":
            return f"Deployed {a['repo']} at {a['ref']}."
        if name == "calendar_add":
            return f"Added '{a['title']}' on {a['when']}."
        raise AssertionError(name)
