"""PII detection + redaction with Presidio (SPEC §9.1 step 5).

Policy (decided 2026-10-25, sealing added 2026-10-26): redact identifiers the agent never
needs; seal (reversibly encrypt) those a task may need again; keep contact details (email,
IBAN) the legitimate tasks rely on. Any detection marks the memory as PII.

Sealing uses Presidio's own `encrypt` / `decrypt` operators (AES-CBC), wrapped so the
ciphertext carries a findable marker, `[PHONE_NUMBER:<ciphertext>]`, and encrypted under a
per-tenant key. Without a sealer, sealed types are redacted permanently instead.
"""
import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from typing import Protocol

from presidio_anonymizer.operators import Decrypt, Encrypt

# Presidio entity -> placeholder written into the stored text (permanent).
DEFAULT_REDACT: dict[str, str] = {
    "CREDIT_CARD": "[CARD]",
    "US_SSN": "[SSN]",
    "US_BANK_NUMBER": "[BANK_ACCOUNT]",
}
# Presidio entity -> placeholder used only when no sealer is configured.
DEFAULT_SEAL: dict[str, str] = {
    "PHONE_NUMBER": "[PHONE]",
    "PERSON": "[PERSON]",
}
DEFAULT_FLAG_ONLY: tuple[str, ...] = ("EMAIL_ADDRESS", "IBAN_CODE")

KEY_ENV = "SMRITI_PII_KEY"
# A sealed value: "[ENTITY_TYPE:" + Presidio's urlsafe-base64 ciphertext + "]". None of these
# characters are escaped by the read gate, so the marker reaches the model unchanged.
SEALED = re.compile(r"\[([A-Z][A-Z_]*):([A-Za-z0-9_-]+={0,2})\]")
# Encrypted together with the value and checked on decryption: AES-CBC has no integrity
# check, so a wrong (other tenant's) key can still decrypt to garbage now and then.
_CHECK = "smriti1:"


@dataclass(frozen=True)
class PiiResult:
    text: str                      # redacted and/or sealed
    found: frozenset[str]          # entity types detected (redacted, sealed or flag-only)


@dataclass(frozen=True)
class Unsealed:
    text: str                      # with every sealed value decrypted
    types: tuple[str, ...]         # entity types decrypted, in order


class SealError(Exception):
    """A sealed value could not be decrypted (wrong tenant, altered, or no key)."""


class PiiRedactor(Protocol):
    def redact(self, text: str, tenant: str | None = None) -> PiiResult: ...
    def unseal(self, text: str, tenant: str | None) -> Unsealed: ...


class _Seal(Encrypt):
    """Presidio `encrypt`, with a check prefix inside and an entity marker around it."""

    def operate(self, text: str = None, params: dict = None) -> str:
        return f"[{params['entity_type']}:{super().operate(_CHECK + text, params)}]"

    def operator_name(self) -> str:
        return "smriti_seal"


class _Unseal(Decrypt):
    """Presidio `decrypt` of one `[TYPE:ciphertext]` marker; rejects a failed check."""

    def operate(self, text: str = None, params: dict = None) -> str:
        m = SEALED.fullmatch(text)
        if not m:
            raise SealError("not a sealed value")
        try:
            plain = super().operate(m.group(2), params)
        except Exception as e:
            raise SealError("cannot decrypt") from e
        if not plain.startswith(_CHECK):
            raise SealError("cannot decrypt")
        return plain[len(_CHECK):]

    def operator_name(self) -> str:
        return "smriti_unseal"


class PiiSealer:
    """Reversible masking with Presidio's encrypt/decrypt and one AES-256 key per tenant,
    derived from a master key, so one tenant's sealed values never open in another."""

    def __init__(self, master_key: bytes):
        if len(master_key) < 32:
            raise ValueError("master key must be at least 32 bytes")
        from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine

        self._master = bytes(master_key)
        self._anonymizer = AnonymizerEngine()
        self._anonymizer.add_anonymizer(_Seal)
        self._deanonymizer = DeanonymizeEngine()
        self._deanonymizer.add_deanonymizer(_Unseal)

    @classmethod
    def from_env(cls, var: str = KEY_ENV) -> "PiiSealer | None":
        """`SMRITI_PII_KEY` as 64 hex characters; unset means no sealing (redact instead)."""
        raw = os.environ.get(var, "").strip()
        if not raw:
            return None
        try:
            return cls(bytes.fromhex(raw))
        except ValueError as e:
            raise ValueError(f"{var} must be 64 hex characters") from e

    def key_for(self, tenant: str) -> bytes:
        return hmac.new(self._master, b"smriti-pii:" + tenant.encode(), hashlib.sha256).digest()

    def register(self, anonymizer) -> None:
        anonymizer.add_anonymizer(_Seal)

    def operator(self, tenant: str):
        from presidio_anonymizer.entities import OperatorConfig
        return OperatorConfig("smriti_seal", {"key": self.key_for(tenant)})

    def seal(self, value: str, entity_type: str, tenant: str) -> str:
        """Seal one whole value (tests and callers outside Presidio's analyzer)."""
        from presidio_anonymizer.entities import RecognizerResult
        out = self._anonymizer.anonymize(
            text=value, analyzer_results=[RecognizerResult(entity_type, 0, len(value), 1.0)],
            operators={entity_type: self.operator(tenant)})
        return out.text

    def unseal(self, text: str, tenant: str) -> Unsealed:
        from presidio_anonymizer.entities import OperatorConfig, OperatorResult

        spans = [OperatorResult(m.start(), m.end(), m.group(1), m.group(0), "smriti_seal")
                 for m in SEALED.finditer(text)]
        if not spans:
            return Unsealed(text, ())
        out = self._deanonymizer.deanonymize(
            text=text, entities=spans,
            operators={"DEFAULT": OperatorConfig("smriti_unseal", {"key": self.key_for(tenant)})})
        return Unsealed(out.text, tuple(s.entity_type for s in spans))


def unseal_or_fail(sealer: "PiiSealer | None", text: str, tenant: str | None) -> Unsealed:
    """Shared by redactors: sealed values without a sealer or tenant fail closed."""
    if not SEALED.search(text):
        return Unsealed(text, ())
    if sealer is None or tenant is None:
        raise SealError("sealed value but no key or tenant")
    return sealer.unseal(text, tenant)


class PresidioRedactor:
    def __init__(
        self,
        redact: dict[str, str] = DEFAULT_REDACT,
        seal: dict[str, str] = DEFAULT_SEAL,
        flag_only: tuple[str, ...] = DEFAULT_FLAG_ONLY,
        min_score: float = 0.4,
        spacy_model: str = "en_core_web_sm",
        sealer: PiiSealer | None = None,
    ):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        nlp = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }).create_engine()
        self._analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        self._anonymizer = AnonymizerEngine()
        self.sealer = sealer
        if sealer is not None:
            sealer.register(self._anonymizer)
        self._redact = dict(redact)
        self._seal = dict(seal)
        self._entities = list(redact) + list(seal) + list(flag_only)
        self._min_score = min_score
        self.redact("warm up")             # first call loads models (~seconds)

    def redact(self, text: str, tenant: str | None = None) -> PiiResult:
        """Redact, and seal when a sealer and tenant are given (else seal types are redacted)."""
        from presidio_anonymizer.entities import OperatorConfig

        results = self._analyzer.analyze(
            text=text, language="en", entities=self._entities, score_threshold=self._min_score)
        # Already-sealed values (e.g. a summary of sealed notes) are left exactly as they are.
        sealed = [m.span() for m in SEALED.finditer(text)]
        results = [r for r in results
                   if not any(r.start < end and start < r.end for start, end in sealed)]
        found = frozenset(r.entity_type for r in results)
        to_change = [r for r in results if r.entity_type in self._redact or r.entity_type in self._seal]
        if not to_change:
            return PiiResult(text, found)
        operators = {e: OperatorConfig("replace", {"new_value": v})
                     for e, v in {**self._redact, **self._seal}.items()}
        if self.sealer is not None and tenant is not None:
            operators.update({e: self.sealer.operator(tenant) for e in self._seal})
        out = self._anonymizer.anonymize(text=text, analyzer_results=to_change, operators=operators)
        return PiiResult(out.text, found)

    def unseal(self, text: str, tenant: str | None) -> Unsealed:
        return unseal_or_fail(self.sealer, text, tenant)
