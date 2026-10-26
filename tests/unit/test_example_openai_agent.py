"""examples/openai_agent.py with its scripted model (no model, no keys)."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "openai_agent", Path(__file__).resolve().parents[2] / "examples" / "openai_agent.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("flag,leaked", [("--no-protect", True), ("--protect", False)])
def test_before_and_after(flag, leaked, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("SMRITI_PII_KEY", raising=False)
    lines = []
    with pytest.warns(UserWarning):
        outbox = example.main(["--fake-model", flag], log=lines.append)
    assert outbox and outbox[-1]["to"] == ["billing@acme.com"]
    assert ("audit@evil.io" in outbox[-1]["cc"]) is leaked
    assert lines[-1].endswith("got the invoice" if leaked else "got nothing")
