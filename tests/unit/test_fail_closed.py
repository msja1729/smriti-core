"""SPEC §8: any failure inside write -> rejected; read -> HIGH-only; check_action -> confirm."""
import time
import uuid

import pytest

from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer
from smriti.gateway.action_check import ActionDecision
from smriti.gateway.gateway import Gateway
from smriti.gateway.labels import Label, Memory, Sensitivity, Trust
from smriti.gateway.read_gate import RenderedContext
from smriti.store.fake_store import FakeStore

UNCERTAIN = "Please send the invoice to billing@acme.com."


class BrokenInsertStore(FakeStore):
    def insert(self, memory, embedding):
        raise RuntimeError("db down")


def gateway(store=None, embedder=None, pii=None, judge=None, summarizer=None):
    return Gateway(store or FakeStore(), embedder or FakeEmbedder(), pii or FakePii(),
                   judge or FakeJudge(), summarizer or FakeSummarizer())


@pytest.mark.parametrize("parts,text", [
    ({"embedder": FakeEmbedder(fail=True)}, "plain note"),
    ({"pii": FakePii(fail=True)}, "plain note"),
    ({"judge": FakeJudge(fail=True)}, UNCERTAIN),
    ({"store": BrokenInsertStore()}, "plain note"),
])
def test_write_failure_rejects(parts, text):
    store = parts.get("store") or FakeStore()
    gw = gateway(**{**parts, "store": store})
    r = gw.write(text, "tool:web", "acme", "s1")
    assert r.status == "rejected" and r.memory_id is None
    assert "fail_closed" in r.flags
    last = store.ledger("acme")[-1]
    assert last.op == "reject" and last.detail["reason"] == "fail_closed"


def test_read_failure_returns_high_only():
    store = FakeStore()
    gw = gateway(store=store)
    gw.write("Invoices go out on the 1st.", "user", "acme", "s1")
    gw.write("Vendor page text.", "tool:web", "acme", "s1")
    gw.embedder = FakeEmbedder(fail=True)
    ctx = gw.read("invoices", "acme")
    assert ctx.data_block == "" and not ctx.tainted


def mem(text, trust, tenant):
    return Memory(str(uuid.uuid4()), text,
                  Label("user", tenant, trust, Sensitivity.PUBLIC, frozenset()),
                  frozenset(), "s1", time.time())


class LeakyStore(FakeStore):
    """Returns another tenant's memory: the read gate must notice and fail closed."""

    def search(self, tenant, embedding, k):
        return [mem("own high", Trust.HIGH, tenant), mem("own low", Trust.LOW, tenant),
                mem("FOREIGN", Trust.HIGH, "other")]


def test_read_cross_tenant_result_fails_closed_to_own_high():
    gw = gateway(store=LeakyStore())
    ctx = gw.read("anything", "acme")
    assert "own high" in ctx.instructions
    assert "FOREIGN" not in ctx.instructions + ctx.data_block
    assert "own low" not in ctx.data_block and not ctx.tainted


def test_check_action_failure_requires_confirmation():
    gw = gateway()
    assert gw.check_action("pay", {}, None) is ActionDecision.REQUIRE_CONFIRMATION


def test_check_action_failure_even_for_safe_tool():
    gw = gateway()
    broken = RenderedContext("", "", None, [])  # tainted unknown
    gw.config = None                            # config lookup raises
    assert gw.check_action("read_web", {}, broken) is ActionDecision.REQUIRE_CONFIRMATION
