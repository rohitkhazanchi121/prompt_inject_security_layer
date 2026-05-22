"""
llm_shield.layers.tool_control
================================
Layer 3 — Tool Control.

Responsibility
--------------
Every tool invocation requested by the model must pass through this layer
before the actual tool handler runs.  Four sub-checks run in sequence:

1. **Quota enforcement**    — per-request call counter blocks runaway tool use.
2. **Allowlist check**      — tool must exist AND the caller's role must be
                              in its ``allowed_roles`` (read from the manifest
                              via the gateway).
3. **Parameter sandboxing** — required-field presence, string length caps, and
                              named danger guards (path traversal, email format, …).
4. **Output scrubbing**     — PII / secret patterns are redacted from the result.

Transport-agnostic design
--------------------------
``ToolController`` talks to tool backends through the ``ToolGateway`` protocol,
NOT directly to the in-process registry.  This is the critical architectural
point: the security checks in this file are **completely independent of how
tools are hosted**.

Today::

    ToolController(gateway=InProcessGateway())   ← default; wraps registry.invoke()

Tomorrow (real MCP server)::

    ToolController(gateway=MCPClientGateway(client))   ← one-line swap

The MCP server hosts every tool with zero access control — it just receives a
call and runs the handler.  ``tool_control.py`` is the gatekeeper that sits
*above* the transport layer and enforces: who can call what, with what params,
how many times, and with what data visible in the result.

Public API
----------
``ToolGateway``       — Protocol (structural interface) any backend must satisfy.
``InProcessGateway``  — Default gateway: delegates to the module-level registry.
``ToolController``    — The Layer 3 security gate; accepts any ``ToolGateway``.
``CallCounter``       — Per-request mutable quota tracker.
``invoke_checked``    — Module-level convenience wrapper (uses default gateway).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llm_shield.config import ToolControlSettings, settings
from llm_shield.exceptions import (
    ToolNotAllowedError,
    ToolOutputScrubError,
    ToolParameterError,
    ToolQuotaExceededError,
)
from llm_shield.mcp.server import ToolDescriptor, ToolInvocationResult, registry


# ---------------------------------------------------------------------------
# ToolGateway protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolGateway(Protocol):
    """
    Structural interface any tool backend must satisfy.

    ``ToolController`` depends only on this protocol — never on a concrete
    registry or MCP client class.  Swap the gateway to change the transport;
    the security logic is untouched.

    Methods
    -------
    get_descriptor(name)
        Return the ``ToolDescriptor`` for *name*, or ``None`` if unknown.
        Used by the allowlist check and parameter sandboxing.
    invoke(name, params)
        Execute the tool and return a ``ToolInvocationResult``.
        The gateway MUST NOT apply any access control — that's Layer 3's job.
    list_tool_names()
        Return all tool names this gateway knows about (for error messages).
    """

    def get_descriptor(self, name: str) -> ToolDescriptor | None: ...
    def invoke(self, name: str, params: dict[str, Any]) -> ToolInvocationResult: ...
    def list_tool_names(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# InProcessGateway — default implementation
# ---------------------------------------------------------------------------


class InProcessGateway(ToolGateway):
    """
    Gateway that dispatches tool calls to the in-process ``ToolRegistry``.

    Explicitly inherits from ``ToolGateway`` so that:
    - Intent is unambiguous — this IS a gateway, not just something that
      happens to have the right methods.
    - Static analysers (mypy, pyright) verify the interface is fully
      implemented; a missing or mistyped method is a type error, not a
      silent runtime duck-typing miss.
    - IDE "Find implementations" / "Go to definition" navigation works.

    To switch to a real MCP server, implement a second concrete subclass::

        class MCPClientGateway(ToolGateway):
            def __init__(self, client: mcp.ClientSession) -> None:
                self._client = client

            def get_descriptor(self, name: str) -> ToolDescriptor | None: ...
            def invoke(self, name: str, params: dict[str, Any]) -> ToolInvocationResult: ...
            def list_tool_names(self) -> list[str]: ...

        controller = ToolController(gateway=MCPClientGateway(session))

    Quota, allowlist, param sandboxing, and output scrubbing stay in
    ``ToolController`` regardless of which gateway subclass you use.
    """

    def get_descriptor(self, name: str) -> ToolDescriptor | None:
        return registry.get(name)

    def invoke(self, name: str, params: dict[str, Any]) -> ToolInvocationResult:
        return registry.invoke(name, params)

    def list_tool_names(self) -> list[str]:
        return registry.all_tool_names()


# ---------------------------------------------------------------------------
# Parameter guard rules
# ---------------------------------------------------------------------------

# Additional dangerous-value checks applied on top of type validation.
# Each entry: (tool_category_or_name, param_name, check_fn, reason)
_PARAM_GUARDS: list[tuple[str, str, Any, str]] = [
    # Path traversal guard for all filesystem tools
    ("filesystem", "path", lambda v: ".." not in str(v),          "Path traversal ('..') is not allowed."),
    ("filesystem", "path", lambda v: not str(v).startswith("/"),   "Absolute paths are not permitted."),
    # Email address guard
    ("email", "to", lambda v: re.match(r"[^@]+@[^@]+\.[^@]+", str(v)), "Recipient is not a valid email address."),
    # Prevent writing to log files (write_file only)
    ("write_file", "path", lambda v: not str(v).startswith("logs/"), "Writing to the logs/ directory is forbidden."),
    # Query limit sanity check
    ("query_db", "limit", lambda v: isinstance(v, (int, type(None))) and (v is None or 1 <= int(v) <= 100),
     "limit must be an integer between 1 and 100."),
]


# ---------------------------------------------------------------------------
# Call counter (per-request mutable state)
# ---------------------------------------------------------------------------


@dataclass
class CallCounter:
    """
    Tracks tool invocations within a single pipeline request.

    Create one per request and pass it to every ``ToolController.invoke_checked``
    call in that request's lifecycle.

    The quota is enforced regardless of transport — whether the tool call goes
    to the in-process registry or a remote MCP server, it counts.
    """

    quota: int = field(default_factory=lambda: settings.tool_control.max_tool_calls_per_request)
    used: int = 0
    calls: list[str] = field(default_factory=list)

    def increment(self, tool_name: str) -> None:
        if self.used >= self.quota:
            raise ToolQuotaExceededError(quota=self.quota, used=self.used)
        self.used += 1
        self.calls.append(tool_name)

    @property
    def remaining(self) -> int:
        return max(0, self.quota - self.used)


# ---------------------------------------------------------------------------
# ToolController — the Layer 3 security gate
# ---------------------------------------------------------------------------


class ToolController:
    """
    Layer 3 — Tool Control gate.

    Owns all access-control and safety logic for tool invocations.
    Deliberately knows nothing about transport — it talks only to a
    ``ToolGateway``.

    Parameters
    ----------
    cfg:
        ``ToolControlSettings``.  Defaults to ``settings.tool_control``.
    gateway:
        Any object satisfying the ``ToolGateway`` protocol.
        Defaults to ``InProcessGateway()`` (wraps the in-process registry).
        Pass ``MCPClientGateway(session)`` to route calls through a real
        MCP server without changing a single line of security logic.
    """

    def __init__(
        self,
        cfg: ToolControlSettings | None = None,
        gateway: ToolGateway | None = None,
    ) -> None:
        self._cfg = cfg or settings.tool_control
        self._gateway = gateway or InProcessGateway()
        self._scrub_patterns = self._compile_scrub_patterns()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke_checked(
        self,
        tool_name: str,
        params: dict[str, Any],
        *,
        role: str,
        call_counter: CallCounter,
    ) -> ToolInvocationResult:
        """
        Gate-keep and dispatch a single tool call.

        Security checks run in this order (cheapest first):

        1. Quota  — reject if this request already hit its call cap.
        2. Allowlist — reject if the tool is unknown or role not permitted.
        3. Params — reject if required params are missing or danger guards fire.
        4. Invoke — call the tool via the gateway (transport-agnostic).
        5. Scrub  — redact PII / secrets from the raw result.

        Parameters
        ----------
        tool_name:     Requested tool name.
        params:        Parameter dict from the model.
        role:          Caller's RBAC role (from Layer 4 authentication).
        call_counter:  Per-request quota tracker (mutated in place).

        Returns
        -------
        ToolInvocationResult — with ``scrubbed=True`` if output scrubbing ran.

        Raises
        ------
        ToolQuotaExceededError  — request hit the tool-call limit.
        ToolNotAllowedError     — tool unknown or role not permitted.
        ToolParameterError      — param missing, too long, or failed a danger guard.
        ToolOutputScrubError    — scrubbing failed (response suppressed).
        """
        # 1 — Quota (cheapest — before any gateway interaction)
        call_counter.increment(tool_name)

        # 2 — Allowlist (reads descriptor from gateway — still no tool call yet)
        descriptor = self._check_allowlist(tool_name, role)

        # 3 — Parameter sandboxing
        self._validate_params(tool_name, params, descriptor)

        # 4 — Invoke via gateway (this is the ONLY place a tool call exits Layer 3)
        result = self._gateway.invoke(tool_name, params)

        # 5 — Output scrubbing (runs on whatever the gateway returned)
        if self._cfg.enable_output_scrubbing:
            result = self._scrub_output(result)

        return result

    # ------------------------------------------------------------------
    # Allowlist check
    # ------------------------------------------------------------------

    def _check_allowlist(self, tool_name: str, role: str) -> ToolDescriptor:
        """
        Verify the tool exists and the role is permitted to use it.

        Returns the descriptor on success (so callers don't need to fetch
        it again for parameter validation).
        """
        descriptor = self._gateway.get_descriptor(tool_name)
        if descriptor is None:
            raise ToolNotAllowedError(
                tool_name=tool_name,
                allowed_tools=self._gateway.list_tool_names(),
                role=role,
            )
        if role not in descriptor.allowed_roles:
            # Build the list of tools this role CAN use (for the error message)
            permitted = [
                name for name in self._gateway.list_tool_names()
                if (d := self._gateway.get_descriptor(name)) and role in d.allowed_roles
            ]
            raise ToolNotAllowedError(
                tool_name=tool_name,
                allowed_tools=permitted,
                role=role,
            )
        return descriptor

    # ------------------------------------------------------------------
    # Parameter sandboxing
    # ------------------------------------------------------------------

    def _validate_params(
        self,
        tool_name: str,
        params: dict[str, Any],
        descriptor: ToolDescriptor,
    ) -> None:
        """
        Validate params against the tool's schema and danger guards.

        Schema comes from ``descriptor.parameters`` which is loaded from
        ``tools_manifest.json`` — the same file regardless of transport.
        """
        schema = descriptor.parameters

        # Required field presence
        for param_name, meta in schema.items():
            if meta.get("required") and param_name not in params:
                raise ToolParameterError(
                    tool_name=tool_name,
                    param_name=param_name,
                    reason="Required parameter is missing.",
                )

        # String length cap
        for param_name, value in params.items():
            if isinstance(value, str) and len(value) > 10_000:
                raise ToolParameterError(
                    tool_name=tool_name,
                    param_name=param_name,
                    reason=f"String value exceeds 10 000 character limit ({len(value)} chars).",
                )

        # Named danger guards
        tool_category = descriptor.category
        for guard_key, param_name, check_fn, reason in _PARAM_GUARDS:
            if guard_key not in (tool_category, tool_name):
                continue
            if param_name not in params:
                continue
            try:
                if not check_fn(params[param_name]):
                    raise ToolParameterError(
                        tool_name=tool_name,
                        param_name=param_name,
                        reason=reason,
                    )
            except ToolParameterError:
                raise
            except Exception as exc:
                raise ToolParameterError(
                    tool_name=tool_name,
                    param_name=param_name,
                    reason=f"Guard check raised: {exc}",
                ) from exc

    # ------------------------------------------------------------------
    # Output scrubbing
    # ------------------------------------------------------------------

    def _scrub_output(self, result: ToolInvocationResult) -> ToolInvocationResult:
        """Redact PII and secrets from a tool result — regardless of transport."""
        try:
            scrubbed_result = self._redact_dict(result.result)
            return ToolInvocationResult(
                tool_name=result.tool_name,
                success=result.success,
                result=scrubbed_result,
                error=result.error,
                duration_ms=result.duration_ms,
                scrubbed=True,
            )
        except Exception as exc:
            raise ToolOutputScrubError(
                tool_name=result.tool_name,
                reason=str(exc),
            ) from exc

    def _redact_value(self, value: str) -> str:
        for pattern in self._scrub_patterns:
            value = pattern.sub("[REDACTED]", value)
        return value

    def _redact_dict(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self._redact_value(obj)
        if isinstance(obj, dict):
            return {k: self._redact_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._redact_dict(item) for item in obj]
        return obj

    def _compile_scrub_patterns(self) -> list[re.Pattern[str]]:
        compiled = []
        for raw in self._cfg.sensitive_output_patterns:
            try:
                compiled.append(re.compile(raw))
            except re.error:
                pass
        return compiled


# ---------------------------------------------------------------------------
# Module-level singleton & convenience wrapper
# ---------------------------------------------------------------------------

#: Default controller using the in-process gateway.
#: To use a remote MCP server instead:
#:   from llm_shield.layers.tool_control import ToolController
#:   controller = ToolController(gateway=MCPClientGateway(session))
_default_controller = ToolController()


def invoke_checked(
    tool_name: str,
    params: dict[str, Any],
    *,
    role: str,
    call_counter: CallCounter,
) -> ToolInvocationResult:
    """Module-level convenience wrapper around ``ToolController.invoke_checked``."""
    return _default_controller.invoke_checked(
        tool_name, params, role=role, call_counter=call_counter
    )