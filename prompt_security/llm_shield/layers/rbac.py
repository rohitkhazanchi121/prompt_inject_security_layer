"""
llm_shield.layers.rbac
========================
Layer 4 — Role-Based Access Control (RBAC).

Responsibility
--------------
Every request must be authenticated (caller identity resolved) and authorised
(role checked against requested resources) before the pipeline proceeds.
This layer also owns the structured audit log.

Design
------
Roles are defined in ``data/users.json`` and the permission matrix lives in
``data/tools_manifest.json``.  The RBAC layer loads both at startup and keeps
them in memory.  No database calls.

Role hierarchy (lowest → highest privilege)
-------------------------------------------
``guest``  → read-only search and directory listing.
``user``   → guest + read file/db, get own profile.
``admin``  → user + write file/db, send email, delete file.

Checks
------
1. **Authentication** — look up ``user_id`` in the user store; confirm active.
2. **Authorisation** — verify the user's role is in the tool's ``allowed_roles``.
3. **Audit** — write a structured JSON-Lines entry for every request outcome.

Public API
----------
``RBACController``
    Stateful class; loads manifests once at init.

``RequestIdentity``
    Resolved caller context — downstream layers use this instead of raw user_id.

``authenticate(user_id) -> RequestIdentity``
    Module-level helper: resolve identity from the user store.

``authorise(identity, tool_name) -> None``
    Module-level helper: raise ``AuthorizationError`` if role not permitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_shield.config import RBACSettings, settings
from llm_shield.audit import AuditLogger, AuditEvent, Outcome
from llm_shield.exceptions import (
    AuthenticationError,
    AuthorizationError,
    RoleNotFoundError,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestIdentity:
    """
    Resolved caller identity.  Created by ``RBACController.authenticate``.

    Attributes
    ----------
    user_id:    Opaque caller key (from ``UserInput``).
    name:       Display name from the user store (or ``"Anonymous"``).
    role:       RBAC role string: ``"guest"``, ``"user"``, or ``"admin"``.
    email:      Caller's email address (may be empty for guests).
    active:     Whether the account is active.
    metadata:   Extra fields from the user record.
    is_guest:   True when the user_id was not found and the default role applies.
    """

    user_id: str
    name: str
    role: str
    email: str
    active: bool
    metadata: dict[str, Any]
    is_guest: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_at_least_user(self) -> bool:
        return self.role in ("user", "admin")


# ---------------------------------------------------------------------------
# Role permission matrix
# ---------------------------------------------------------------------------

# Canonical ordered list of roles (lowest → highest).
ROLE_ORDER: list[str] = ["guest", "user", "admin"]


# ---------------------------------------------------------------------------
# RBACController
# ---------------------------------------------------------------------------


class RBACController:
    """
    Layer 4 — RBAC gate and audit logger.

    Parameters
    ----------
    cfg:            ``RBACSettings``.  Defaults to ``settings.rbac``.
    audit_logger:   ``AuditLogger`` instance.  Defaults to the module singleton.
    """

    def __init__(
        self,
        cfg: RBACSettings | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._cfg = cfg or settings.rbac
        self._audit = audit_logger or _default_audit_logger()
        self._users: dict[str, dict[str, Any]] = {}
        self._tool_roles: dict[str, list[str]] = {}
        self._load_users()
        self._load_tool_manifest()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def authenticate(self, user_id: str) -> RequestIdentity:
        """
        Resolve a ``user_id`` to a ``RequestIdentity``.

        If the user is not found, returns a guest identity (fail-open).
        If the account is inactive, raises ``AuthenticationError``.

        Parameters
        ----------
        user_id: The opaque caller identity string from ``UserInput``.

        Raises
        ------
        AuthenticationError
            If the account exists but is marked inactive / suspended.
        """
        user = self._users.get(user_id)

        if user is None:
            # Unknown user → guest identity (fail-open with minimal permissions)
            return RequestIdentity(
                user_id=user_id,
                name="Anonymous",
                role=self._cfg.default_role,
                email="",
                active=True,
                metadata={},
                is_guest=True,
            )

        if not user.get("active", True):
            raise AuthenticationError(
                reason=f"Account '{user_id}' is suspended or inactive."
            )

        role = user.get("role", self._cfg.default_role)
        if role not in ROLE_ORDER:
            raise RoleNotFoundError(role=role, available_roles=ROLE_ORDER)

        return RequestIdentity(
            user_id=user_id,
            name=user.get("name", user_id),
            role=role,
            email=user.get("email", ""),
            active=True,
            metadata=user.get("metadata", {}),
            is_guest=False,
        )

    def authorise(self, identity: RequestIdentity, tool_name: str) -> None:
        """
        Check that ``identity.role`` is allowed to invoke ``tool_name``.

        Parameters
        ----------
        identity:  Resolved caller identity.
        tool_name: The tool being requested.

        Raises
        ------
        AuthorizationError
            If the role is not in the tool's ``allowed_roles``.
        """
        allowed = self._tool_roles.get(tool_name, [])
        if identity.role not in allowed:
            raise AuthorizationError(
                user_id=identity.user_id,
                role=identity.role,
                required=f"one of {allowed}",
                resource=tool_name,
            )

    def can_invoke(self, identity: RequestIdentity, tool_name: str) -> bool:
        """Return True if the identity is permitted to call tool_name."""
        allowed = self._tool_roles.get(tool_name, [])
        return identity.role in allowed

    def allowed_tools_for(self, role: str) -> list[str]:
        """Return all tool names accessible to a given role."""
        return sorted(
            name for name, roles in self._tool_roles.items() if role in roles
        )

    def audit(self, event: AuditEvent) -> None:
        """Write a structured audit log entry."""
        if self._cfg.enable_audit_log:
            self._audit.log(event)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_users(self) -> None:
        """Load users from data/users.json into an in-memory dict."""
        users_path = settings.users_file
        if not users_path.exists():
            return
        try:
            with open(users_path, encoding="utf-8") as fh:
                data = json.load(fh)
            for user in data.get("users", []):
                uid = user.get("user_id")
                if uid:
                    self._users[uid] = user
        except (json.JSONDecodeError, OSError):
            pass  # graceful degradation

    def _load_tool_manifest(self) -> None:
        """Load tool-role mappings from data/tools_manifest.json."""
        manifest_path = settings.tools_manifest_file
        if not manifest_path.exists():
            return
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                data = json.load(fh)
            for tool in data.get("tools", []):
                name = tool.get("name")
                roles = tool.get("allowed_roles", [])
                if name:
                    self._tool_roles[name] = roles
        except (json.JSONDecodeError, OSError):
            pass


# ---------------------------------------------------------------------------
# Audit logger factory (lazy singleton)
# ---------------------------------------------------------------------------

_audit_logger_instance: AuditLogger | None = None


def _default_audit_logger() -> AuditLogger:
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger(
            log_path=settings.rbac.audit_log_path,
            enabled=settings.rbac.enable_audit_log,
        )
    return _audit_logger_instance


# ---------------------------------------------------------------------------
# Module-level singleton & helpers
# ---------------------------------------------------------------------------

_default_controller: RBACController | None = None


def _get_controller() -> RBACController:
    global _default_controller
    if _default_controller is None:
        _default_controller = RBACController()
    return _default_controller


def authenticate(user_id: str) -> RequestIdentity:
    """Module-level convenience wrapper."""
    return _get_controller().authenticate(user_id)


def authorise(identity: RequestIdentity, tool_name: str) -> None:
    """Module-level convenience wrapper."""
    _get_controller().authorise(identity, tool_name)


def allowed_tools_for(role: str) -> list[str]:
    """Return all tools accessible to a role."""
    return _get_controller().allowed_tools_for(role)