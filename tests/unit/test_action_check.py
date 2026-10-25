import pytest

from fakes import FakeEmbedder, FakeJudge, FakePii, FakeSummarizer
from smriti.gateway.action_check import ActionDecision, check_action
from smriti.gateway.gateway import Gateway, GatewayConfig
from smriti.gateway.read_gate import RenderedContext
from smriti.store.fake_store import FakeStore

RISKY = GatewayConfig().risky_tools


@pytest.mark.parametrize("tool", sorted(RISKY))
def test_tainted_risky_requires_confirmation(tool):
    assert check_action(tool, tainted=True, risky_tools=RISKY) is ActionDecision.REQUIRE_CONFIRMATION


@pytest.mark.parametrize("tool,tainted", [
    ("read_web", True), ("calendar_add", True), ("send_email", False), ("pay", False)])
def test_other_cases_allowed(tool, tainted):
    assert check_action(tool, tainted=tainted, risky_tools=RISKY) is ActionDecision.ALLOW


def test_default_risky_tools():
    assert RISKY == {"send_email", "pay", "deploy", "share_file"}


def ctx(tainted):
    return RenderedContext("", "", tainted, [])


def test_gateway_check_action_and_switch():
    gw = Gateway(FakeStore(), FakeEmbedder(), FakePii(), FakeJudge(), FakeSummarizer())
    assert gw.check_action("pay", {}, ctx(True)) is ActionDecision.REQUIRE_CONFIRMATION
    off = Gateway(FakeStore(), FakeEmbedder(), FakePii(), FakeJudge(), FakeSummarizer(),
                  GatewayConfig(action_check=False))
    assert off.check_action("pay", {}, ctx(True)) is ActionDecision.ALLOW
