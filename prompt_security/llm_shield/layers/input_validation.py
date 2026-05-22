"""
llm_shield.layers.input_validation
====================================
Layer 1 — Input Validation.

Responsibility
--------------
Reject or sanitise raw user input *before* it ever reaches the LLM or the
prompt-construction stage.  This is the cheapest, fastest, and most reliable
line of defence: pure Python, zero network calls, < 1 ms for typical inputs.

Checks (in execution order)
----------------------------
1. **Encoding normalisation** — NFC unicode, null-byte stripping.
2. **Length guard** — character count + rough token estimate.
3. **Blocked substrings** — literal case-insensitive phrase blocklist.
4. **Injection patterns** — regex heuristics that catch structural attacks
   (role overrides, delimiter confusion, etc.).
5. **Schema validation** — Pydantic model ensuring the caller passed a
   well-formed ``UserInput`` object.

Public API
----------
``validate(raw: str, *, context: RequestContext) -> SanitisedInput``
    Run all checks; return a clean ``SanitisedInput`` or raise an
    ``InputValidationError`` subclass.

``InputValidator``
    Stateful class (holds compiled regexes); useful when you want to
    inject a custom config in tests.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from llm_shield.config import InputValidationSettings, settings
from llm_shield.exceptions import (
    BlockedSubstringError,
    EncodingError,
    InjectionPatternError,
    InputTooLongError,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Injection pattern library
# ---------------------------------------------------------------------------
# Each tuple is (name, pattern).  Keeping names lets us surface *which*
# pattern fired in the exception details and audit log.

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Role / persona overrides
    ("role_override_system", r"(?i)\b(?:system|assistant|user)\s*:\s*"),
    ("role_override_xml", r"(?i)<\s*(?:system|assistant|user)\s*>"),
    # Instruction override attempts
    ("ignore_instructions", r"(?i)ignor(?:e|ing)\s+(?:all\s+)?(?:previous|above|prior|the\s+above)"),
    ("override_instructions", r"(?i)(?:override|overwrite|replace|forget)\s+(?:your\s+)?(?:instructions?|rules?|guidelines?|constraints?)"),
    ("new_instructions", r"(?i)(?:new|updated?|revised?)\s+instructions?\s*[:\-]"),
    # Prompt delimiter confusion
    ("triple_backtick_injection", r"```\s*(?:system|assistant|prompt|instructions?)"),
    ("xml_cdata_injection", r"<!\[CDATA\["),
    ("delimiter_confusion", r"(?i)(?:###|---|\*\*\*)\s*(?:system|instructions?|rules?)"),
    # Persona / jailbreak triggers
    ("dan_mode", r"(?i)\bDAN\s*mode\b"),
    ("developer_mode", r"(?i)\bdeveloper\s*mode\b"),
    ("jailbreak_explicit", r"(?i)\bjailbreak\b"),
    ("evil_twin", r"(?i)(?:evil|unrestricted|uncensored)\s+(?:mode|version|ai|assistant)"),
    # Indirect injection hooks (content that tells a downstream LLM to act)
    ("indirect_injection_hook", r"(?i)(?:when\s+you\s+read\s+this|if\s+you\s+(?:see|find|read)\s+this)"),
    ("prompt_leak_attempt", r"(?i)(?:reveal|show|print|display|output|repeat)\s+(?:your\s+)?(?:system\s+prompt|instructions?|context|rules?)"),
    # Base64 / encoded payload heuristic (>=40 consecutive b64 chars is suspicious)
    ("base64_payload", r"[A-Za-z0-9+/]{40,}={0,2}"),
    # Excessive special characters (often used to confuse tokenisers)
    ("token_flooding", r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]{3,}"),
    # Fake tool-call injection
    ("fake_tool_call", r"(?i)<\s*tool(?:_call|_result|_use)?\s*>"),
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class UserInput(BaseModel):
    """
    The raw payload a caller sends to the pipeline.

    Attributes
    ----------
    content:
        The user's message / prompt text.
    user_id:
        Opaque caller identifier (used by RBAC in later layers).
    session_id:
        Optional session correlation token (for audit logs).
    metadata:
        Arbitrary caller-supplied metadata; passed through unchanged.
    """

    content: str = Field(..., min_length=1, description="User message text.")
    user_id: str = Field(..., min_length=1, description="Caller identity key.")
    session_id: str | None = Field(default=None, description="Session correlation ID.")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary caller metadata (not used in validation).",
    )

    @field_validator("content")
    @classmethod
    def content_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank or whitespace-only.")
        return v

    @field_validator("user_id")
    @classmethod
    def user_id_must_be_safe(cls, v: str) -> str:
        # Prevent user_id itself from being a vector (e.g. path traversal).
        if re.search(r'[/\\<>"\'`]', v):
            raise ValueError("user_id contains disallowed characters.")
        return v


@dataclass(frozen=True)
class SanitisedInput:
    """
    The result of a successful Layer 1 pass.

    Downstream layers receive this object, never the raw string.

    Attributes
    ----------
    original:
        The raw content as supplied by the caller (preserved for audit).
    sanitised:
        The cleaned / normalised content that will be forwarded.
    user_id:
        Caller identity, forwarded unchanged.
    session_id:
        Session ID, forwarded unchanged.
    char_length:
        Character length of the sanitised content.
    estimated_tokens:
        Rough token estimate (chars / 4).
    flags:
        Human-readable list of transformations applied (e.g. "null_bytes_stripped").
    metadata:
        Caller metadata, forwarded unchanged.
    """

    original: str
    sanitised: str
    user_id: str
    session_id: str | None
    char_length: int
    estimated_tokens: int
    flags: list[str]
    metadata: dict[str, str]


@dataclass
class ValidationResult:
    """Intermediate result accumulator used inside ``InputValidator``."""

    text: str
    flags: list[str] = field(default_factory=list)

    def add_flag(self, flag: str) -> None:
        self.flags.append(flag)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class InputValidator:
    """
    Stateful input validator.

    Compile regex patterns once at construction time and reuse across requests.
    For production use, instantiate once at module load (see ``_default_validator``
    at the bottom of this file) and call ``validate()`` on it.

    Parameters
    ----------
    cfg:
        ``InputValidationSettings`` instance.  Defaults to ``settings.input_validation``.
    """

    def __init__(self, cfg: InputValidationSettings | None = None) -> None:
        self._cfg = cfg or settings.input_validation
        self._compiled_patterns = self._compile_patterns()
        self._blocked_lower = [s.lower() for s in self._cfg.blocked_substrings]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self, user_input: UserInput) -> SanitisedInput:
        """
        Run all Layer 1 checks against *user_input*.

        Returns a ``SanitisedInput`` on success.
        Raises an ``InputValidationError`` subclass on failure.

        Parameters
        ----------
        user_input:
            A validated ``UserInput`` Pydantic model.

        Raises
        ------
        EncodingError
            Null bytes or non-UTF-8 sequences detected.
        InputTooLongError
            Character or token limit exceeded.
        BlockedSubstringError
            One or more literal blocked phrases found.
        InjectionPatternError
            One or more injection pattern regexes matched.
        """
        result = ValidationResult(text=user_input.content)

        # Step 1 — encoding / normalisation
        result = self._check_encoding(result)

        # Step 2 — length guard
        self._check_length(result.text)

        # Step 3 — blocked substrings (literal, fast)
        self._check_blocked_substrings(result.text)

        # Step 4 — injection patterns (regex)
        if self._cfg.enable_pattern_detection:
            self._check_injection_patterns(result.text)

        estimated_tokens = max(1, len(result.text) // 4)

        return SanitisedInput(
            original=user_input.content,
            sanitised=result.text,
            user_id=user_input.user_id,
            session_id=user_input.session_id,
            char_length=len(result.text),
            estimated_tokens=estimated_tokens,
            flags=result.flags,
            metadata=user_input.metadata,
        )

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_encoding(self, result: ValidationResult) -> ValidationResult:
        """Normalise unicode; strip null bytes; reject control-character floods."""
        text = result.text

        # Null-byte check first — they can mask subsequent patterns
        if "\x00" in text:
            if not self._cfg.strip_null_bytes:
                raise EncodingError("Input contains null bytes (\\x00).")
            text = text.replace("\x00", "")
            result.add_flag("null_bytes_stripped")

        # Reject non-UTF-8 bytes (shouldn't happen in Python str, but guard anyway)
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise EncodingError(f"Input contains non-UTF-8 bytes: {exc}") from exc

        # Unicode normalisation (NFC)
        if self._cfg.normalise_unicode:
            normalised = unicodedata.normalize("NFC", text)
            if normalised != text:
                result.add_flag("unicode_normalised")
            text = normalised

        # Reject strings that are *mostly* control characters (obfuscation attempt)
        control_chars = sum(1 for c in text if unicodedata.category(c) == "Cc" and c != "\n")
        if control_chars > 10 and (control_chars / max(len(text), 1)) > 0.05:
            raise EncodingError(
                f"Input contains an unusually high proportion of control characters "
                f"({control_chars}/{len(text)})."
            )

        result.text = text
        return result

    def _check_length(self, text: str) -> None:
        """Enforce character and rough token limits."""
        char_len = len(text)
        if char_len > self._cfg.max_input_length:
            raise InputTooLongError(
                actual_length=char_len,
                max_length=self._cfg.max_input_length,
            )

        estimated_tokens = char_len // 4
        if estimated_tokens > self._cfg.max_prompt_tokens:
            raise InputTooLongError(
                actual_length=char_len,
                max_length=self._cfg.max_prompt_tokens * 4,
            )

    def _check_blocked_substrings(self, text: str) -> None:
        """Scan for exact (case-insensitive) blocked phrases."""
        lower_text = text.lower()
        matched = [
            phrase
            for phrase in self._cfg.blocked_substrings
            if phrase.lower() in lower_text
        ]
        if matched:
            raise BlockedSubstringError(matched_substrings=matched)

    def _check_injection_patterns(self, text: str) -> None:
        """Run compiled injection-pattern regexes against the text."""
        fired: list[str] = []
        for name, pattern in self._compiled_patterns:
            if pattern.search(text):
                fired.append(name)

        if len(fired) >= self._cfg.pattern_detection_threshold:
            raise InjectionPatternError(matched_patterns=fired)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compile_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        """Compile injection patterns once; store (name, compiled) tuples."""
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for name, raw in _INJECTION_PATTERNS:
            try:
                compiled.append((name, re.compile(raw)))
            except re.error as exc:
                # A bad pattern in the list is a programming error, not a
                # runtime error — surface it loudly at startup.
                raise ValueError(
                    f"Invalid injection pattern '{name}': {exc}"
                ) from exc
        return compiled


# ---------------------------------------------------------------------------
# Convenience module-level helpers
# ---------------------------------------------------------------------------

#: Default validator instance — uses settings from config.
_default_validator = InputValidator()


def validate(user_input: UserInput) -> SanitisedInput:
    """
    Module-level convenience wrapper around ``InputValidator.validate``.

    Equivalent to ``InputValidator().validate(user_input)`` but reuses the
    module-level singleton (compiled regex cache shared across calls).

    Parameters
    ----------
    user_input:
        A validated ``UserInput`` Pydantic model.

    Returns
    -------
    SanitisedInput
        Clean input ready for Layer 2.

    Raises
    ------
    InputValidationError
        (or a subclass) if any check fails.
    """
    return _default_validator.validate(user_input)


def validate_raw(
    content: str,
    *,
    user_id: str = "anonymous",
    session_id: str | None = None,
) -> SanitisedInput:
    """
    Convenience wrapper: construct a ``UserInput`` and validate in one step.

    Useful in scripts and tests where you just have a raw string.

    Parameters
    ----------
    content:
        Raw user message.
    user_id:
        Caller identity (defaults to ``"anonymous"``).
    session_id:
        Optional session ID for audit correlation.
    """
    user_input = UserInput(content=content, user_id=user_id, session_id=session_id)
    return validate(user_input)