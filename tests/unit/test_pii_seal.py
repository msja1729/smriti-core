"""Reversible sealing with Presidio's encrypt/decrypt (no spaCy, no network)."""
import pytest

from fakes import FakePii
from smriti.models.pii import SEALED, PiiSealer, SealError, unseal_or_fail

KEY = bytes(range(32))
PHONE = "+1 415 555 0132"


@pytest.fixture(scope="module")
def sealer():
    return PiiSealer(KEY)


def test_seal_hides_value_and_round_trips(sealer):
    sealed = sealer.seal(PHONE, "PHONE_NUMBER", "finance")
    assert SEALED.fullmatch(sealed) and "555" not in sealed
    text = f"My phone is {sealed}, call me."
    out = sealer.unseal(text, "finance")
    assert out.text == f"My phone is {PHONE}, call me."
    assert out.types == ("PHONE_NUMBER",)


def test_same_value_seals_differently_each_time(sealer):
    assert sealer.seal(PHONE, "PHONE_NUMBER", "t") != sealer.seal(PHONE, "PHONE_NUMBER", "t")


def test_several_values_unseal_together(sealer):
    a = sealer.seal("Priya", "PERSON", "t")
    b = sealer.seal(PHONE, "PHONE_NUMBER", "t")
    out = sealer.unseal(f"{a} at {b}", "t")
    assert out.text == f"Priya at {PHONE}" and out.types == ("PERSON", "PHONE_NUMBER")


def test_text_without_sealed_values_is_unchanged(sealer):
    assert sealer.unseal("nothing [here] to open", "t").text == "nothing [here] to open"


def test_other_tenant_cannot_unseal(sealer):
    # The check prefix makes this reliable, not just likely: try many values.
    for i in range(50):
        sealed = sealer.seal(f"{PHONE}{i}", "PHONE_NUMBER", "finance")
        with pytest.raises(SealError):
            sealer.unseal(sealed, "devops")


def test_other_master_key_cannot_unseal(sealer):
    sealed = sealer.seal(PHONE, "PHONE_NUMBER", "t")
    with pytest.raises(SealError):
        PiiSealer(bytes(32)).unseal(sealed, "t")


@pytest.mark.parametrize("change", [
    lambda c: c[:-6] + ("A" if c[-6] != "A" else "B") + c[-5:],   # one character altered
    lambda c: c[:-4],                                             # truncated
    lambda c: c + "AAAA",                                         # extended
])
def test_altered_ciphertext_is_rejected(sealer, change):
    sealed = sealer.seal(PHONE, "PHONE_NUMBER", "t")
    m = SEALED.fullmatch(sealed)
    bad = f"[{m.group(1)}:{change(m.group(2))}]"
    with pytest.raises(SealError):
        sealer.unseal(bad, "t")


def test_short_master_key_rejected():
    with pytest.raises(ValueError):
        PiiSealer(b"too short")


def test_from_env(monkeypatch):
    monkeypatch.delenv("SMRITI_PII_KEY", raising=False)
    assert PiiSealer.from_env() is None
    monkeypatch.setenv("SMRITI_PII_KEY", KEY.hex())
    assert PiiSealer.from_env().key_for("t") == PiiSealer(KEY).key_for("t")
    monkeypatch.setenv("SMRITI_PII_KEY", "not hex")
    with pytest.raises(ValueError):
        PiiSealer.from_env()


def test_tenant_keys_differ(sealer):
    assert sealer.key_for("a") != sealer.key_for("b") and len(sealer.key_for("a")) == 32


def test_unseal_without_key_or_tenant_fails_closed(sealer):
    sealed = sealer.seal(PHONE, "PHONE_NUMBER", "t")
    with pytest.raises(SealError):
        unseal_or_fail(None, sealed, "t")
    with pytest.raises(SealError):
        unseal_or_fail(sealer, sealed, None)
    assert unseal_or_fail(None, "plain text", None).text == "plain text"


def test_fake_pii_seals_only_with_sealer_and_tenant(sealer):
    assert FakePii().redact(f"call {PHONE}", "t").text == "call [PHONE]"
    assert FakePii(sealer=sealer).redact(f"call {PHONE}").text == "call [PHONE]"
    out = FakePii(sealer=sealer).redact(f"call {PHONE}", "t")
    assert "555" not in out.text and out.found == {"PHONE_NUMBER"}
    assert sealer.unseal(out.text, "t").text == f"call {PHONE}"


def test_fake_pii_leaves_sealed_values_alone(sealer):
    pii = FakePii(sealer=sealer)
    once = pii.redact(f"call {PHONE}", "t").text
    assert pii.redact(once, "t").text == once


def test_ordinary_bracketed_text_is_not_a_sealed_value(sealer):
    for text in ("[NOTE:see below]", "[TODO:call back]", "[A:" + "x" * 20 + "]"):
        assert not SEALED.search(text)
        assert sealer.unseal(text, "t").text == text
