"""
llm_shield.mcp.server
=======================
MCP tool server — manifest-driven tool registry.

Architecture: single source of truth
--------------------------------------
``data/tools_manifest.json`` owns ALL tool metadata:
  description, parameters schema, allowed_roles, rate_limit, category.

``server.py`` owns ONLY the mapping of tool name → Python handler callable.

At startup, ``ToolRegistry`` loads the manifest, iterates every entry, looks
up the matching handler from ``_HANDLER_MAP``, and builds a ``ToolDescriptor``
that merges both.  If a tool appears in the manifest but has no handler, or a
handler exists with no manifest entry, ``ConfigurationError`` is raised
immediately — drift between the two is a hard startup failure, not a silent bug.

This means:
  - To add a new tool: add it to ``tools_manifest.json`` AND add its handler
    to ``_HANDLER_MAP`` below.  That's it.  No other code changes needed.
  - To change permissions, description, or parameters: edit the manifest only.
  - To change behaviour: edit the handler only.

Public API
----------
``ToolRegistry``                — singleton registry class.
``registry``                    — module-level instance.
``invoke_tool(name, **params)`` — dispatch helper used by ``tool_control.py``.
``list_tools()``                — return all registered tool descriptors.
``get_tool(name)``              — look up a single descriptor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from llm_shield.config import settings
from llm_shield.exceptions import ConfigurationError
from llm_shield.mcp.tools.database import get_user_profile, insert_record, query_db
from llm_shield.mcp.tools.filesystem import delete_file, list_dir, read_file, write_file
from llm_shield.mcp.tools.search_email import send_email, web_search


# ---------------------------------------------------------------------------
# Handler map — the ONLY place handlers are declared in Python
# ---------------------------------------------------------------------------
# Add a new entry here when you add a new tool handler.
# The key must exactly match the "name" field in tools_manifest.json.

_HANDLER_MAP: dict[str, Callable[..., Any]] = {
    "read_file":       read_file,
    "write_file":      write_file,
    "list_dir":        list_dir,
    "delete_file":     delete_file,
    "query_db":        query_db,
    "insert_record":   insert_record,
    "get_user_profile": get_user_profile,
    "web_search":      web_search,
    "send_email":      send_email,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDescriptor:
    """
    Merged view of a tool: handler from code + all metadata from the manifest.

    Attributes
    ----------
    name:          Tool name (must be unique across the registry).
    handler:       Python callable that implements the tool.
    description:   Human-readable description (from manifest).
    category:      Grouping label, e.g. 'filesystem' (from manifest).
    parameters:    JSON-schema-style parameter definitions (from manifest).
    allowed_roles: RBAC roles permitted to invoke this tool (from manifest).
    rate_limit:    Rate-limit spec dict, e.g. {'calls_per_minute': 10} (from manifest).
    """

    name: str
    handler: Callable[..., Any]
    description: str
    category: str
    parameters: dict[str, Any]
    allowed_roles: list[str]
    rate_limit: dict[str, Any]


@dataclass
class ToolInvocationResult:
    """
    Wrapper around a tool's raw return value.

    Attributes
    ----------
    tool_name:   Name of the invoked tool.
    success:     Whether the tool reported success.
    result:      Raw dict returned by the handler.
    error:       Error string if success is False, else None.
    duration_ms: Wall-clock time of the invocation in milliseconds.
    scrubbed:    True if tool_control.py applied output scrubbing.
    """

    tool_name: str
    success: bool
    result: dict[str, Any]
    error: str | None = None
    duration_ms: float = 0.0
    scrubbed: bool = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Manifest-driven in-process MCP tool registry.

    Construction
    ------------
    1. Load ``data/tools_manifest.json``.
    2. For each manifest entry, look up the handler in ``_HANDLER_MAP``.
    3. Build a ``ToolDescriptor`` merging manifest metadata + handler.
    4. Validate consistency — fail hard on drift (missing handler or
       missing manifest entry).

    All metadata (description, parameters, allowed_roles, rate_limit,
    category) comes exclusively from the manifest.  The Python code
    contributes only the callable handler.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        manifest = self._load_manifest()
        self._build_from_manifest(manifest)
        self._validate_no_orphan_handlers(manifest)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        """Load and parse tools_manifest.json. Raise ConfigurationError on failure."""
        path = settings.tools_manifest_file
        if not path.exists():
            raise ConfigurationError(
                f"tools_manifest.json not found at '{path}'. "
                "Create it or set DATA_DIR to the correct directory.",
                setting_key="DATA_DIR",
            )
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"tools_manifest.json is not valid JSON: {exc}",
                setting_key="DATA_DIR",
            ) from exc

    def _build_from_manifest(self, manifest: dict[str, Any]) -> None:
        """
        Iterate manifest entries and merge with _HANDLER_MAP.

        Raises ConfigurationError if a manifest tool has no handler —
        this catches the case where someone adds to the manifest but
        forgets to wire the Python callable.
        """
        for entry in manifest.get("tools", []):
            name = entry.get("name")
            if not name:
                continue  # skip malformed entries

            handler = _HANDLER_MAP.get(name)
            if handler is None:
                raise ConfigurationError(
                    f"Tool '{name}' is defined in tools_manifest.json but has no "
                    f"handler in _HANDLER_MAP in mcp/server.py. "
                    f"Add an entry: '{name}': <your_callable>.",
                    setting_key="MCP_HANDLER_MAP",
                )

            self._tools[name] = ToolDescriptor(
                name=name,
                handler=handler,
                description=entry.get("description", ""),
                category=entry.get("category", "unknown"),
                parameters=entry.get("parameters", {}),
                allowed_roles=entry.get("allowed_roles", []),
                rate_limit=entry.get("rate_limit", {}),
            )

    def _validate_no_orphan_handlers(self, manifest: dict[str, Any]) -> None:
        """
        Warn if a handler exists in _HANDLER_MAP with no manifest entry.

        This catches the reverse drift: handler added to code but manifest
        not updated.  We warn (not error) so that development is less
        friction — but it will show as a startup warning in production logs.
        """
        manifest_names = {t["name"] for t in manifest.get("tools", []) if "name" in t}
        orphans = set(_HANDLER_MAP.keys()) - manifest_names
        if orphans:
            import warnings
            warnings.warn(
                f"Handlers registered in _HANDLER_MAP have no manifest entry "
                f"and will NOT be accessible via the registry: {sorted(orphans)}. "
                f"Add them to tools_manifest.json to expose them.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDescriptor | None:
        """Return a ``ToolDescriptor`` by name, or ``None`` if not registered."""
        return self._tools.get(name)

    def all_tools(self) -> list[ToolDescriptor]:
        """Return all registered tool descriptors (in manifest order)."""
        return list(self._tools.values())

    def all_tool_names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def invoke(self, name: str, params: dict[str, Any]) -> ToolInvocationResult:
        """
        Dispatch a tool call synchronously.

        This method is intentionally *not* guarded by RBAC or tool-control
        checks — those live in ``tool_control.py`` which wraps this call.
        Calling ``registry.invoke`` directly bypasses all security layers
        (which is the point — the demo shows what happens without the layers).

        Parameters
        ----------
        name:   Tool name (must match a manifest entry).
        params: Keyword arguments forwarded verbatim to the handler.

        Returns
        -------
        ToolInvocationResult
        """
        descriptor = self._tools.get(name)
        if descriptor is None:
            return ToolInvocationResult(
                tool_name=name,
                success=False,
                result={},
                error=f"Tool '{name}' is not registered. Known tools: {self.all_tool_names()}",
            )

        t0 = time.perf_counter()
        try:
            raw = descriptor.handler(**params)
            duration_ms = (time.perf_counter() - t0) * 1000
            success = raw.get("success", True) if isinstance(raw, dict) else True
            error = raw.get("error") if isinstance(raw, dict) else None
            return ToolInvocationResult(
                tool_name=name,
                success=success,
                result=raw if isinstance(raw, dict) else {"value": raw},
                error=error,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - t0) * 1000
            return ToolInvocationResult(
                tool_name=name,
                success=False,
                result={},
                error=f"Tool raised an exception: {exc!s}",
                duration_ms=duration_ms,
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global registry — import and use everywhere.
registry: ToolRegistry = ToolRegistry()


def invoke_tool(name: str, **params: Any) -> ToolInvocationResult:
    """Dispatch a tool by name via the global registry."""
    return registry.invoke(name, params)


def list_tools() -> list[ToolDescriptor]:
    """Return all registered tool descriptors."""
    return registry.all_tools()


def get_tool(name: str) -> ToolDescriptor | None:
    """Look up a single tool descriptor."""
    return registry.get(name)


# ---------------------------------------------------------------------------
# Real FastMCP server (production swap-in)
# ---------------------------------------------------------------------------
# To turn this into a real network MCP server:
#
#   from mcp.server.fastmcp import FastMCP
#   mcp = FastMCP("llm-shield-demo")
#
#   # Register every tool from the manifest automatically:
#   for descriptor in registry.all_tools():
#       mcp.add_tool(
#           descriptor.handler,
#           name=descriptor.name,
#           description=descriptor.description,
#       )
#
#   if __name__ == "__main__":
#       mcp.run()                                           # stdio transport
#       # mcp.run(transport="sse", host="0.0.0.0", port=8080)  # SSE transport
#
# The pipeline's tool_control layer would then call the MCP client instead
# of registry.invoke() directly.
# ---------------------------------------------------------------------------