"""
llm_shield.mcp.tools.filesystem
=================================
Dummy filesystem tools — no real disk I/O.

All "files" live in an in-process dict so the demo is fully self-contained
and safe to run anywhere.  The store is pre-seeded with realistic-looking
fixtures that make the demo scenarios meaningful.

Tools exposed
-------------
``read_file(path)``           — return file content or a not-found error.
``write_file(path, content)`` — overwrite / create a file in the store.
``list_dir(path)``            — list entries under a virtual directory prefix.
``delete_file(path)``         — remove a file from the store.

Security notes (demonstrated here for the showcase)
---------------------------------------------------
* Path traversal guard: any path containing ``..`` or starting with ``/`` is
  rejected before the tool body runs — the check lives in ``tool_control.py``
  but is also enforced here as defence-in-depth.
* Write access is gated by RBAC in ``tool_control.py``; these functions
  themselves are role-agnostic (the pipeline enforces permissions upstream).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# In-memory filesystem fixture
# ---------------------------------------------------------------------------

_FS_STORE: dict[str, str] = {
    "reports/q3_sales.txt": (
        "Q3 Sales Report\n"
        "===============\n"
        "Total Revenue:  $4,821,000\n"
        "Units Sold:     18,340\n"
        "Top Product:    Widget Pro X\n"
        "Region Leader:  Pacific Northwest\n"
        "YoY Growth:     +12.4%\n"
    ),
    "reports/q3_expenses.txt": (
        "Q3 Expense Report\n"
        "=================\n"
        "Salaries:       $1,200,000\n"
        "Infrastructure: $340,000\n"
        "Marketing:      $210,000\n"
        "Travel:         $88,000\n"
        "Total:          $1,838,000\n"
    ),
    "config/app_settings.json": json.dumps(
        {
            "app_name": "AcmeCRM",
            "version": "3.2.1",
            "max_connections": 100,
            "log_level": "INFO",
            "feature_flags": {
                "new_dashboard": True,
                "beta_search": False,
            },
        },
        indent=2,
    ),
    "docs/onboarding.md": (
        "# Employee Onboarding Guide\n\n"
        "Welcome to Acme Corp!  This guide covers your first week.\n\n"
        "## Day 1\n"
        "- Collect your access badge from reception (Building A, floor 1).\n"
        "- Set up your workstation using the IT setup script.\n"
        "- Complete mandatory security training (link in your welcome email).\n\n"
        "## Day 2\n"
        "- Meet your team lead for a 1:1 orientation.\n"
        "- Join #general and #your-team on Slack.\n"
    ),
    "logs/access.log": (
        "2024-09-01T08:12:33Z INFO  user-alice  login  success\n"
        "2024-09-01T08:45:01Z INFO  user-bob    query  accounts\n"
        "2024-09-01T09:03:17Z WARN  guest-001   query  admin_panel  DENIED\n"
        "2024-09-01T09:15:55Z INFO  admin-carol write  config       success\n"
    ),
}

# Metadata store: path → {created, modified, size}
_FS_META: dict[str, dict[str, Any]] = {
    path: {
        "created": "2024-09-01T00:00:00Z",
        "modified": "2024-09-01T00:00:00Z",
        "size_bytes": len(content),
    }
    for path, content in _FS_STORE.items()
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _validate_path(path: str) -> str | None:
    """
    Return an error string if the path is unsafe, else None.
    Defence-in-depth: tool_control.py also checks this.
    """
    if ".." in path:
        return "Path traversal detected: '..' is not allowed."
    if path.startswith("/"):
        return "Absolute paths are not permitted; use relative paths."
    if not path.strip():
        return "Path must not be empty."
    return None


def read_file(path: str) -> dict[str, Any]:
    """
    Read a file from the dummy filesystem.

    Parameters
    ----------
    path:
        Relative path within the virtual filesystem (e.g. ``"reports/q3_sales.txt"``).

    Returns
    -------
    dict with keys:
        ``success`` bool, ``path`` str, ``content`` str | None,
        ``error`` str | None, ``metadata`` dict | None.
    """
    if err := _validate_path(path):
        return {"success": False, "path": path, "content": None, "error": err, "metadata": None}

    if path not in _FS_STORE:
        return {
            "success": False,
            "path": path,
            "content": None,
            "error": f"File not found: '{path}'",
            "metadata": None,
        }

    return {
        "success": True,
        "path": path,
        "content": _FS_STORE[path],
        "error": None,
        "metadata": _FS_META.get(path),
    }


def write_file(path: str, content: str) -> dict[str, Any]:
    """
    Write content to the dummy filesystem.

    Parameters
    ----------
    path:    Relative path.
    content: String content to store.
    """
    if err := _validate_path(path):
        return {"success": False, "path": path, "error": err}

    now = datetime.now(timezone.utc).isoformat()
    existed = path in _FS_STORE
    _FS_STORE[path] = content
    _FS_META[path] = {
        "created": _FS_META.get(path, {}).get("created", now),
        "modified": now,
        "size_bytes": len(content),
    }
    return {
        "success": True,
        "path": path,
        "action": "updated" if existed else "created",
        "error": None,
    }


def list_dir(path: str) -> dict[str, Any]:
    """
    List virtual filesystem entries under a path prefix.

    Parameters
    ----------
    path:
        Directory prefix (e.g. ``"reports"`` or ``""`` for root).
    """
    if err := _validate_path(path or "."):
        return {"success": False, "path": path, "entries": [], "error": err}

    prefix = path.rstrip("/") + "/" if path else ""
    entries = [
        {
            "name": p[len(prefix):] if prefix else p,
            "full_path": p,
            "size_bytes": _FS_META[p]["size_bytes"],
            "modified": _FS_META[p]["modified"],
        }
        for p in sorted(_FS_STORE)
        if p.startswith(prefix) or not prefix
    ]

    return {"success": True, "path": path or "/", "entries": entries, "error": None}


def delete_file(path: str) -> dict[str, Any]:
    """
    Remove a file from the dummy filesystem.

    Parameters
    ----------
    path: Relative path to remove.
    """
    if err := _validate_path(path):
        return {"success": False, "path": path, "error": err}

    if path not in _FS_STORE:
        return {"success": False, "path": path, "error": f"File not found: '{path}'"}

    del _FS_STORE[path]
    _FS_META.pop(path, None)
    return {"success": True, "path": path, "action": "deleted", "error": None}