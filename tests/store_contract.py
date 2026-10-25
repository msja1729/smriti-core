"""Behaviour every store must have. Run against FakeStore (unit) and PgStore (integration)."""
import time
import uuid

import pytest

from smriti.gateway.labels import CrossTenantError, Label, Memory, Sensitivity, ToolCall, Trust


def mem(text, tenant="acme", trust=Trust.LOW, flags=(), parents=(), source="tool:web"):
    return Memory(
        id=str(uuid.uuid4()),
        text=text,
        label=Label(source, tenant, trust, Sensitivity.PUBLIC, frozenset(parents)),
        flags=frozenset(flags),
        session_id="s1",
        created_at=time.time(),
    )


def check_roundtrip(store):
    m = mem("hello", trust=Trust.MED, flags={"instruction_like"})
    store.insert(m, [1.0, 0.0, 0.0])
    got = store.get("acme", m.id)
    assert got.id == m.id and got.text == m.text and got.label == m.label
    assert got.flags == m.flags
    assert abs(got.created_at - m.created_at) < 1e-3


def check_get_is_tenant_scoped(store):
    m = mem("secret", tenant="b")
    store.insert(m, [1.0, 0.0, 0.0])
    assert store.get("a", m.id) is None
    assert store.get("b", m.id) is not None


def check_search_is_tenant_scoped(store):
    # TC-04 at store level: a canary in tenant B never comes back for tenant A.
    canary = mem("CANARY-7731", tenant="b")
    store.insert(canary, [1.0, 0.0, 0.0])
    store.insert(mem("a-note", tenant="a"), [0.0, 1.0, 0.0])
    got = store.search("a", [1.0, 0.0, 0.0], k=10)
    assert [m.text for m in got] == ["a-note"]


def check_search_orders_and_limits(store):
    near = mem("near")
    mid = mem("mid")
    far = mem("far")
    store.insert(far, [0.0, 0.0, 1.0])
    store.insert(near, [1.0, 0.0, 0.0])
    store.insert(mid, [1.0, 1.0, 0.0])
    got = store.search("acme", [1.0, 0.0, 0.0], k=2)
    assert [m.text for m in got] == ["near", "mid"]


def check_search_excludes_quarantined(store):
    store.insert(mem("poison", flags={"quarantined", "instruction_like"}), [1.0, 0.0, 0.0])
    store.insert(mem("ok"), [0.9, 0.1, 0.0])
    got = store.search("acme", [1.0, 0.0, 0.0], k=10)
    assert [m.text for m in got] == ["ok"]


def check_parents_roundtrip(store):
    p1, p2 = mem("p1"), mem("p2")
    store.insert(p1, [1.0, 0.0, 0.0])
    store.insert(p2, [0.0, 1.0, 0.0])
    child = mem("summary", parents={p1.id, p2.id}, source="derived")
    store.insert(child, [1.0, 1.0, 0.0])
    assert store.get("acme", child.id).label.parents == {p1.id, p2.id}


def check_cross_tenant_parent_rejected(store):
    other = mem("theirs", tenant="b")
    store.insert(other, [1.0, 0.0, 0.0])
    child = mem("mine", tenant="a", parents={other.id}, source="derived")
    with pytest.raises(CrossTenantError):
        store.insert(child, [1.0, 0.0, 0.0])
    assert store.get("a", child.id) is None


def check_ledger(store):
    mid = str(uuid.uuid4())
    store.append_ledger("acme", "write", mid, {"status": "stored"})
    store.append_ledger("acme", "reject", None, {"reason": "quota_hit"})
    store.append_ledger("other", "write", None, {})
    entries = store.ledger("acme")
    assert [e.op for e in entries] == ["write", "reject"]
    assert entries[0].seq < entries[1].seq
    assert entries[0].memory_id == mid and entries[1].memory_id is None
    assert entries[1].detail == {"reason": "quota_hit"}


def check_tool_log(store):
    mid = str(uuid.uuid4())
    call = ToolCall("ep1", "s1", 3, "send_email", {"to": "a@acme.com", "cc": []},
                    True, "confirm_denied", [mid])
    store.log_tool_call(call)
    store.log_tool_call(ToolCall("ep2", "s1", 0, "pay", {}, False, "executed", []))
    assert store.tool_log("ep1") == [call]


def check_reset(store):
    store.insert(mem("x"), [1.0, 0.0, 0.0])
    store.append_ledger("acme", "write", None, {})
    store.log_tool_call(ToolCall("ep1", "s1", 0, "pay", {}, False, "executed", []))
    store.reset()
    assert store.search("acme", [1.0, 0.0, 0.0], k=10) == []
    assert store.ledger("acme") == []
    assert store.tool_log("ep1") == []


def check_archive_is_tenant_scoped_and_hides_from_search(store):
    mine, theirs = mem("mine", tenant="a"), mem("theirs", tenant="b")
    store.insert(mine, [1.0, 0.0, 0.0])
    store.insert(theirs, [1.0, 0.0, 0.0])
    store.archive("a", [mine.id, theirs.id])
    store.archive("a", [mine.id])                      # idempotent
    assert store.search("a", [1.0, 0.0, 0.0], k=10) == []
    assert store.get("a", mine.id).flags == {"archived"}
    assert [m.text for m in store.search("b", [1.0, 0.0, 0.0], k=10)] == ["theirs"]


def check_search_exclude_and_as_of(store):
    old, new = mem("old"), mem("new")
    old = old.__class__(**{**old.__dict__, "created_at": 1000.0})
    new = new.__class__(**{**new.__dict__, "created_at": 2000.0})
    store.insert(old, [1.0, 0.0, 0.0])
    store.insert(new, [1.0, 0.0, 0.0])
    q = [1.0, 0.0, 0.0]
    assert {m.text for m in store.search("acme", q, k=10)} == {"old", "new"}
    assert [m.text for m in store.search("acme", q, k=10, exclude_ids=(new.id,))] == ["old"]
    assert [m.text for m in store.search("acme", q, k=10, as_of=1500.0)] == ["old"]


def check_descendants_and_quarantine_are_tenant_scoped(store):
    root = mem("root")
    store.insert(root, [1.0, 0.0, 0.0])
    child = mem("child", parents={root.id}, source="derived")
    store.insert(child, [1.0, 0.0, 0.0])
    grandchild = mem("grandchild", parents={root.id, child.id}, source="derived")
    store.insert(grandchild, [1.0, 0.0, 0.0])
    assert {m.text for m in store.descendants("acme", [root.id])} == {"child", "grandchild"}
    assert store.descendants("other", [root.id]) == []
    store.quarantine("other", [root.id])                 # wrong tenant: no effect
    assert "quarantined" not in store.get("acme", root.id).flags
    store.quarantine("acme", [root.id, child.id])
    store.quarantine("acme", [root.id])                  # idempotent
    assert store.get("acme", root.id).flags == {"quarantined"}
    assert [m.text for m in store.search("acme", [1.0, 0.0, 0.0], k=10)] == ["grandchild"]


def check_turns_roundtrip_and_tenant_scoped(store):
    s1 = store.record_turn("acme", "ep1", "s1", "Send the invoice.", 1000.0)
    s2 = store.record_turn("acme", "ep1", "s2", "Read email 7.", 1001.0)
    store.record_turn("other", "ep1", "s1", "theirs", 1002.0)
    turns = store.turns("acme", "ep1")
    assert [t.seq for t in turns] == [s1, s2] and s1 < s2
    assert turns[0].user_text == "Send the invoice." and abs(turns[0].ts - 1000.0) < 1e-3
    assert store.turns("acme", "ep2") == []


def check_tool_call_turn_seq(store):
    call = ToolCall("ep9", "s1", 0, "pay", {}, False, "executed", [], 7)
    store.log_tool_call(call)
    assert store.tool_log("ep9") == [call]


CHECKS = [
    check_search_exclude_and_as_of,
    check_descendants_and_quarantine_are_tenant_scoped,
    check_turns_roundtrip_and_tenant_scoped,
    check_tool_call_turn_seq,
    check_archive_is_tenant_scoped_and_hides_from_search,
    check_roundtrip,
    check_get_is_tenant_scoped,
    check_search_is_tenant_scoped,
    check_search_orders_and_limits,
    check_search_excludes_quarantined,
    check_parents_roundtrip,
    check_cross_tenant_parent_rejected,
    check_ledger,
    check_tool_log,
    check_reset,
]
