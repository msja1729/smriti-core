"""Real Presidio on CPU (no network). Default policy: redact phone etc., keep email/IBAN."""
import pytest

pytest.importorskip("presidio_analyzer")

from smriti.models.pii import PresidioRedactor  # noqa: E402


@pytest.fixture(scope="module")
def pii():
    return PresidioRedactor()


def test_phone_redacted(pii):
    r = pii.redact("Call Bob at +1 415-555-0132 tomorrow.")
    assert "[PHONE]" in r.text and "555" not in r.text
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
