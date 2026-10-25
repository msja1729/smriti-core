"""PII detection + redaction with Presidio (SPEC §9.1 step 5).

Policy (decided 2026-10-25): redact identifiers the agent never needs; keep contact
details (email, IBAN) the legitimate tasks rely on, but still mark the memory as PII.
"""
from dataclasses import dataclass
from typing import Protocol

# Presidio entity -> placeholder written into the stored text.
DEFAULT_REDACT: dict[str, str] = {
    "PHONE_NUMBER": "[PHONE]",
    "CREDIT_CARD": "[CARD]",
    "US_SSN": "[SSN]",
    "US_BANK_NUMBER": "[BANK_ACCOUNT]",
    "PERSON": "[PERSON]",
}
DEFAULT_FLAG_ONLY: tuple[str, ...] = ("EMAIL_ADDRESS", "IBAN_CODE")


@dataclass(frozen=True)
class PiiResult:
    text: str                      # redacted
    found: frozenset[str]          # entity types detected (redacted or flag-only)


class PiiRedactor(Protocol):
    def redact(self, text: str) -> PiiResult: ...


class PresidioRedactor:
    def __init__(
        self,
        redact: dict[str, str] = DEFAULT_REDACT,
        flag_only: tuple[str, ...] = DEFAULT_FLAG_ONLY,
        min_score: float = 0.4,
        spacy_model: str = "en_core_web_sm",
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
        self._redact = dict(redact)
        self._entities = list(redact) + list(flag_only)
        self._min_score = min_score
        self.redact("warm up")             # first call loads models (~seconds)

    def redact(self, text: str) -> PiiResult:
        from presidio_anonymizer.entities import OperatorConfig

        results = self._analyzer.analyze(
            text=text, language="en", entities=self._entities, score_threshold=self._min_score)
        found = frozenset(r.entity_type for r in results)
        to_redact = [r for r in results if r.entity_type in self._redact]
        if not to_redact:
            return PiiResult(text, found)
        operators = {e: OperatorConfig("replace", {"new_value": v}) for e, v in self._redact.items()}
        out = self._anonymizer.anonymize(text=text, analyzer_results=to_redact, operators=operators)
        return PiiResult(out.text, found)
