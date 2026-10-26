"""Real Presidio on CPU (no network). Default policy: redact phone etc., keep email/IBAN."""
import pytest

from fakes import has_phone

pytest.importorskip("presidio_analyzer")

from smriti.models.pii import PresidioRedactor  # noqa: E402


@pytest.fixture(scope="module")
def pii():
    return PresidioRedactor()


def test_phone_redacted(pii):
    r = pii.redact("Call Bob at +1 415-555-0132 tomorrow.")
    assert "[PHONE]" in r.text and not has_phone(r.text)
    assert "PHONE_NUMBER" in r.found


def test_email_kept_but_found(pii):
    r = pii.redact("Acme billing is billing@acme.com.")
    assert "billing@acme.com" in r.text
    assert "EMAIL_ADDRESS" in r.found


def test_iban_kept_but_found(pii):
    r = pii.redact("Pay IBAN GB29NWBK60161331926819.")
    assert "GB29NWBK60161331926819" in r.text
    assert "IBAN_CODE" in r.found


@pytest.mark.parametrize("text", [
    "Send the October invoice to Acme.",
    "Meeting with the Acme team moved to Thursday at 3pm.",
    "The main branch passed CI at 10:42.",
])
def test_business_text_untouched(pii, text):
    r = pii.redact(text)
    assert r.text == text and r.found == frozenset()


# -- sealing (reversible) -----------------------------------------------------------------

@pytest.fixture(scope="module")
def sealing():
    from smriti.models.pii import PiiSealer
    return PresidioRedactor(sealer=PiiSealer(bytes(range(32))))


def test_without_sealer_phone_and_name_are_redacted_as_before(pii):
    r = pii.redact("Sam (sam@example.com) is my cousin, +1 415-555-0132.", "personal")
    assert "[PERSON]" in r.text and "[PHONE]" in r.text and "sam@example.com" in r.text


def test_sealer_encrypts_phone_and_name(sealing):
    r = sealing.redact("Sam (sam@example.com) is my cousin, +1 415-555-0132.", "personal")
    assert "Sam " not in r.text and not has_phone(r.text) and "sam@example.com" in r.text
    assert "[PERSON:" in r.text and "[PHONE_NUMBER:" in r.text
    assert {"PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"} <= r.found
    out = sealing.unseal(r.text, "personal")
    assert out.text == "Sam (sam@example.com) is my cousin, +1 415-555-0132."


def test_sealer_without_tenant_redacts(sealing):
    r = sealing.redact("Call +1 415-555-0132.")
    assert "[PHONE]" in r.text and "[PHONE_NUMBER:" not in r.text


def test_never_needed_identifiers_stay_permanently_redacted(sealing):
    r = sealing.redact("My card is 4111 1111 1111 1111.", "personal")
    assert r.text == "My card is [CARD]."


def test_already_sealed_text_is_not_sealed_again(sealing):
    once = sealing.redact("Call Sam at +1 415-555-0132.", "personal").text
    twice = sealing.redact(f"Summary: {once}", "personal").text
    assert twice == f"Summary: {once}"
