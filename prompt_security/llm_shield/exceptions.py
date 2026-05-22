"""
llm_shield.exceptions
=====================
Custom exception hierarchy for the llm-shield project.

Design goals
------------
* Every exception carries enough context to produce a useful structured log
  entry without the caller needing to remember extra fields.
* HTTP-friendly: each class exposes an ``http_status`` attribute so a future
  FastAPI/Flask wrapper can map exceptions to responses with zero boilerplate.
* Machine-readable: a ``code`` slug (e.g. ``"INPUT_TOO_LONG"``) lets the demo
  CLI and tests do ``except InputValidationError as e: if e.code == ...``
  without parsing human-readable messages.
* Layered: the hierarchy mirrors the four defence layers so log aggregation
  can filter by layer trivially.

Hierarchy
---------
LLMShieldError                          ← base; always catch this
├── InputValidationError  (Layer 1)
│   ├── InputTooLongError
│   ├── InjectionPatternError
│   ├── EncodingError
│   └── BlockedSubstringError
├── PromptValidationError (Layer 2)
│   ├── IntentViolationError
│   ├── CanaryLeakError
│   └── ScopeBoundaryError
├── ToolControlError      (Layer 3)
│   ├── ToolNotAllowedError
│   ├── ToolParameterError
│   ├── ToolQuotaExceededError
│   └── ToolOutputScrubError
├── RBACError             (Layer 4)
│   ├── AuthenticationError
│   ├── AuthorizationError
│   └── RoleNotFoundError
├── PipelineError                       ← orchestration-level problems
└── ConfigurationError                  ← bad settings / missing files
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class LLMShieldError(Exception):
    """
    Base exception for all llm-shield errors.

    Parameters
    ----------
    message:
        Human-readable description (shown in logs and CLI output).
    code:
        Upper-snake-case slug identifying the error variant, e.g.
        ``"INPUT_TOO_LONG"``.  Defaults to the class name.
    http_status:
        Suggested HTTP status code for API wrappers.
    details:
        Arbitrary extra context (serialised into the audit log).
    """

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or type(self).__name__.upper()
        if http_status is not None:
            self.http_status = http_status
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON logging."""
        return {
            "error_type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "http_status": self.http_status,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"details={self.details!r})"
        )


# ---------------------------------------------------------------------------
# Layer 1 — Input Validation
# ---------------------------------------------------------------------------


class InputValidationError(LLMShieldError):
    """Raised when raw user input fails any Layer 1 check."""

    http_status: int = HTTPStatus.BAD_REQUEST


class InputTooLongError(InputValidationError):
    """
    Input exceeds the configured character or token limit.

    Extra ``details`` keys
    ----------------------
    ``actual_length``   : int — measured length of the input.
    ``max_length``      : int — configured limit.
    """

    def __init__(self, actual_length: int, max_length: int) -> None:
        super().__init__(
            message=(
                f"Input length {actual_length} exceeds the maximum allowed "
                f"length of {max_length} characters."
            ),
            code="INPUT_TOO_LONG",
            details={"actual_length": actual_length, "max_length": max_length},
        )
        self.actual_length = actual_length
        self.max_length = max_length


class InjectionPatternError(InputValidationError):
    """
    Input matches one or more known injection patterns.

    Extra ``details`` keys
    ----------------------
    ``matched_patterns`` : list[str] — regex patterns that fired.
    ``hit_count``        : int — total number of pattern hits.
    """

    def __init__(self, matched_patterns: list[str]) -> None:
        super().__init__(
            message=(
                f"Input contains {len(matched_patterns)} injection pattern(s).  "
                "Request blocked."
            ),
            code="INJECTION_PATTERN_DETECTED",
            details={
                "matched_patterns": matched_patterns,
                "hit_count": len(matched_patterns),
            },
        )
        self.matched_patterns = matched_patterns


class BlockedSubstringError(InputValidationError):
    """
    Input contains a literal blocked substring.

    Extra ``details`` keys
    ----------------------
    ``matched_substrings`` : list[str] — the substrings that triggered the block.
    """

    def __init__(self, matched_substrings: list[str]) -> None:
        super().__init__(
            message=(
                f"Input contains {len(matched_substrings)} blocked phrase(s).  "
                "Request blocked."
            ),
            code="BLOCKED_SUBSTRING",
            details={"matched_substrings": matched_substrings},
        )
        self.matched_substrings = matched_substrings


class EncodingError(InputValidationError):
    """
    Input contains disallowed byte sequences or encoding anomalies.

    Extra ``details`` keys
    ----------------------
    ``reason`` : str — human-readable description of the encoding issue.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Input encoding check failed: {reason}",
            code="ENCODING_ERROR",
            details={"reason": reason},
        )
        self.reason = reason


# ---------------------------------------------------------------------------
# Layer 2 — Prompt Validation
# ---------------------------------------------------------------------------


class PromptValidationError(LLMShieldError):
    """Raised when a prompt fails any Layer 2 structural or semantic check."""

    http_status: int = HTTPStatus.UNPROCESSABLE_ENTITY


class IntentViolationError(PromptValidationError):
    """
    The LLM classifier flagged the prompt as malicious / out-of-scope.

    Extra ``details`` keys
    ----------------------
    ``intent_label``      : str  — classifier label, e.g. ``"prompt_injection"``.
    ``confidence``        : float — 0.0–1.0 classifier confidence.
    ``classifier_model``  : str  — model used for classification.
    """

    def __init__(
        self,
        intent_label: str,
        confidence: float,
        classifier_model: str,
    ) -> None:
        super().__init__(
            message=(
                f"Prompt intent classified as '{intent_label}' "
                f"(confidence {confidence:.0%}) by {classifier_model}.  "
                "Request blocked."
            ),
            code="INTENT_VIOLATION",
            details={
                "intent_label": intent_label,
                "confidence": confidence,
                "classifier_model": classifier_model,
            },
        )
        self.intent_label = intent_label
        self.confidence = confidence


class CanaryLeakError(PromptValidationError):
    """
    The model's response contains the injected canary token, indicating
    that the system prompt boundary has been breached.

    Extra ``details`` keys
    ----------------------
    ``canary_token`` : str — the token that was found in the output.
    """

    def __init__(self, canary_token: str) -> None:
        super().__init__(
            message=(
                f"Canary token '{canary_token}' detected in model output.  "
                "System prompt integrity may be compromised."
            ),
            code="CANARY_LEAK",
            http_status=HTTPStatus.FORBIDDEN,
            details={"canary_token": canary_token},
        )
        self.canary_token = canary_token


class ScopeBoundaryError(PromptValidationError):
    """
    The prompt contains invalid or mismatched scope boundary tags.

    Extra ``details`` keys
    ----------------------
    ``found_tags``   : list[str] — tags discovered in the prompt.
    ``allowed_tags`` : list[str] — tags permitted by config.
    """

    def __init__(self, found_tags: list[str], allowed_tags: list[str]) -> None:
        super().__init__(
            message=(
                f"Prompt contains unrecognised scope tags: {found_tags}.  "
                f"Allowed tags: {allowed_tags}."
            ),
            code="SCOPE_BOUNDARY_VIOLATION",
            details={"found_tags": found_tags, "allowed_tags": allowed_tags},
        )


# ---------------------------------------------------------------------------
# Layer 3 — Tool Control
# ---------------------------------------------------------------------------


class ToolControlError(LLMShieldError):
    """Raised when a tool invocation fails any Layer 3 check."""

    http_status: int = HTTPStatus.FORBIDDEN


class ToolNotAllowedError(ToolControlError):
    """
    The requested tool is not in the allowlist for the current context.

    Extra ``details`` keys
    ----------------------
    ``tool_name``   : str       — the requested tool.
    ``allowed_tools``: list[str] — tools permitted in this context.
    ``role``        : str       — caller's role (may overlap with RBAC).
    """

    def __init__(
        self,
        tool_name: str,
        allowed_tools: list[str],
        role: str,
    ) -> None:
        super().__init__(
            message=(
                f"Tool '{tool_name}' is not permitted for role '{role}'.  "
                f"Allowed tools: {allowed_tools}."
            ),
            code="TOOL_NOT_ALLOWED",
            details={
                "tool_name": tool_name,
                "allowed_tools": allowed_tools,
                "role": role,
            },
        )
        self.tool_name = tool_name


class ToolParameterError(ToolControlError):
    """
    A tool was called with invalid or dangerous parameters.

    Extra ``details`` keys
    ----------------------
    ``tool_name``  : str  — the tool that was called.
    ``param_name`` : str  — the offending parameter.
    ``reason``     : str  — why the parameter was rejected.
    """

    def __init__(self, tool_name: str, param_name: str, reason: str) -> None:
        super().__init__(
            message=(
                f"Tool '{tool_name}' received an invalid parameter "
                f"'{param_name}': {reason}."
            ),
            code="TOOL_PARAMETER_INVALID",
            http_status=HTTPStatus.BAD_REQUEST,
            details={
                "tool_name": tool_name,
                "param_name": param_name,
                "reason": reason,
            },
        )


class ToolQuotaExceededError(ToolControlError):
    """
    The request has already used its maximum allowed tool calls.

    Extra ``details`` keys
    ----------------------
    ``quota``  : int — configured maximum.
    ``used``   : int — calls already made in this request.
    """

    def __init__(self, quota: int, used: int) -> None:
        super().__init__(
            message=(
                f"Tool call quota exceeded: {used} calls made, "
                f"maximum is {quota}."
            ),
            code="TOOL_QUOTA_EXCEEDED",
            http_status=HTTPStatus.TOO_MANY_REQUESTS,
            details={"quota": quota, "used": used},
        )


class ToolOutputScrubError(ToolControlError):
    """
    A tool produced output that could not be safely scrubbed.
    This is an internal error — the response is suppressed entirely.

    Extra ``details`` keys
    ----------------------
    ``tool_name``   : str — the tool that produced the output.
    ``reason``      : str — scrubbing failure reason.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(
            message=(
                f"Tool '{tool_name}' output could not be scrubbed safely: {reason}.  "
                "Response suppressed."
            ),
            code="TOOL_OUTPUT_SCRUB_FAILED",
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
            details={"tool_name": tool_name, "reason": reason},
        )


# ---------------------------------------------------------------------------
# Layer 4 — RBAC
# ---------------------------------------------------------------------------


class RBACError(LLMShieldError):
    """Raised when a request fails any Layer 4 access-control check."""

    http_status: int = HTTPStatus.FORBIDDEN


class AuthenticationError(RBACError):
    """
    The caller could not be identified.

    Extra ``details`` keys
    ----------------------
    ``reason`` : str — why authentication failed.
    """

    def __init__(self, reason: str = "No valid identity provided.") -> None:
        super().__init__(
            message=f"Authentication failed: {reason}",
            code="AUTHENTICATION_FAILED",
            http_status=HTTPStatus.UNAUTHORIZED,
            details={"reason": reason},
        )


class AuthorizationError(RBACError):
    """
    The caller is authenticated but lacks permission for the requested action.

    Extra ``details`` keys
    ----------------------
    ``user_id``     : str       — caller identity.
    ``role``        : str       — caller's role.
    ``required``    : str       — permission / role that was needed.
    ``resource``    : str       — what was being accessed.
    """

    def __init__(
        self,
        user_id: str,
        role: str,
        required: str,
        resource: str,
    ) -> None:
        super().__init__(
            message=(
                f"User '{user_id}' (role: '{role}') is not authorised to access "
                f"'{resource}'.  Required permission: '{required}'."
            ),
            code="AUTHORIZATION_FAILED",
            details={
                "user_id": user_id,
                "role": role,
                "required": required,
                "resource": resource,
            },
        )


class RoleNotFoundError(RBACError):
    """
    A role name was referenced that does not exist in the manifest.

    Extra ``details`` keys
    ----------------------
    ``role``           : str       — unknown role name.
    ``available_roles`` : list[str] — roles that are defined.
    """

    def __init__(self, role: str, available_roles: list[str]) -> None:
        super().__init__(
            message=(
                f"Role '{role}' is not defined.  "
                f"Available roles: {available_roles}."
            ),
            code="ROLE_NOT_FOUND",
            http_status=HTTPStatus.BAD_REQUEST,
            details={"role": role, "available_roles": available_roles},
        )


# ---------------------------------------------------------------------------
# Pipeline & Config
# ---------------------------------------------------------------------------


class PipelineError(LLMShieldError):
    """
    Orchestration-level error: a layer returned an unexpected result or the
    pipeline state machine reached an invalid state.
    """

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str, *, stage: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="PIPELINE_ERROR",
            details={"stage": stage, **(details or {})},
        )
        self.stage = stage


class ConfigurationError(LLMShieldError):
    """
    A required configuration value is missing or invalid.
    Raised at startup / import time, not during request processing.
    """

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str, *, setting_key: str | None = None) -> None:
        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            details={"setting_key": setting_key} if setting_key else {},
        )