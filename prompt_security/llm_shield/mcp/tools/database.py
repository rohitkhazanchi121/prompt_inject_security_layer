"""
llm_shield.mcp.tools.database
================================
Dummy database tools — in-memory tables, no real DB required.

Tables
------
``accounts``    — customer accounts with balance and status.
``orders``      — purchase orders linked to accounts.
``products``    — product catalogue with pricing.
``employees``   — internal HR records (admin-only in RBAC).

Tools exposed
-------------
``query_db(table, filter)``         — filtered SELECT on a table.
``insert_record(table, record)``    — append a row to a table.
``get_user_profile(user_id)``       — shortcut to fetch one user record.

Security notes
--------------
* ``query_db`` accepts only column-equality filters, never raw SQL strings —
  a structural guard against SQL injection analogous to parameterised queries.
* ``employees`` table is listed here but gated to admin role by the RBAC
  layer (``tools_manifest.json``).  The tool itself doesn't check roles;
  the pipeline does.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# In-memory tables
# ---------------------------------------------------------------------------

_DB: dict[str, list[dict[str, Any]]] = {
    "accounts": [
        {"account_id": "ACC-001", "owner": "user-alice", "balance": 12_450.00,  "status": "active",    "tier": "gold"},
        {"account_id": "ACC-002", "owner": "user-bob",   "balance":  3_200.50,  "status": "active",    "tier": "silver"},
        {"account_id": "ACC-003", "owner": "guest-001",  "balance":    150.00,  "status": "active",    "tier": "basic"},
        {"account_id": "ACC-004", "owner": "user-bob",   "balance":      0.00,  "status": "suspended", "tier": "basic"},
    ],
    "orders": [
        {"order_id": "ORD-1001", "account_id": "ACC-001", "product_id": "PROD-A", "qty": 3,  "total": 299.97, "status": "shipped"},
        {"order_id": "ORD-1002", "account_id": "ACC-001", "product_id": "PROD-B", "qty": 1,  "total": 499.00, "status": "delivered"},
        {"order_id": "ORD-1003", "account_id": "ACC-002", "product_id": "PROD-A", "qty": 10, "total": 999.90, "status": "processing"},
        {"order_id": "ORD-1004", "account_id": "ACC-003", "product_id": "PROD-C", "qty": 2,  "total":  49.98, "status": "delivered"},
    ],
    "products": [
        {"product_id": "PROD-A", "name": "Widget Pro X",   "price": 99.99,  "stock": 342, "category": "hardware"},
        {"product_id": "PROD-B", "name": "SuperSaaS Plan", "price": 499.00, "stock": 999, "category": "software"},
        {"product_id": "PROD-C", "name": "Acme USB Hub",   "price": 24.99,  "stock": 85,  "category": "hardware"},
        {"product_id": "PROD-D", "name": "Cloud Storage 1TB", "price": 9.99, "stock": 999, "category": "software"},
    ],
    "employees": [
        {"emp_id": "EMP-001", "name": "Alice Nguyen", "role": "engineer",   "salary": 145_000, "department": "engineering"},
        {"emp_id": "EMP-002", "name": "Bob Patel",    "role": "sales_rep",  "salary":  95_000, "department": "sales"},
        {"emp_id": "EMP-003", "name": "Carol Osei",   "role": "platform",   "salary": 160_000, "department": "platform"},
        {"emp_id": "EMP-004", "name": "Dave Kim",     "role": "sec_eng",    "salary": 155_000, "department": "security"},
    ],
}

_KNOWN_TABLES = frozenset(_DB.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_filter(rows: list[dict[str, Any]], filter_: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Apply equality filters only.

    ``{"account_id": "ACC-001", "status": "active"}`` → rows that match
    both conditions.  No operators, no raw SQL — structural injection guard.
    """
    result = rows
    for key, value in filter_.items():
        result = [r for r in result if r.get(key) == value]
    return result


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def query_db(
    table: str,
    filter: dict[str, Any] | None = None,  # noqa: A002  (mirrors tool param name)
    limit: int = 50,
) -> dict[str, Any]:
    """
    Run a read-only query against a dummy database table.

    Parameters
    ----------
    table:
        One of: ``accounts``, ``orders``, ``products``, ``employees``.
    filter:
        Optional dict of ``{column: value}`` equality conditions.
    limit:
        Maximum rows to return (default 50, max 100).

    Returns
    -------
    dict with keys:
        ``success``, ``table``, ``rows``, ``row_count``, ``error``.
    """
    if table not in _KNOWN_TABLES:
        return {
            "success": False,
            "table": table,
            "rows": [],
            "row_count": 0,
            "error": f"Unknown table '{table}'. Known tables: {sorted(_KNOWN_TABLES)}",
        }

    limit = min(int(limit), 100)
    rows = copy.deepcopy(_DB[table])

    if filter:
        # Guard: only str/int/float/bool values are accepted in filters
        for k, v in filter.items():
            if not isinstance(v, (str, int, float, bool, type(None))):
                return {
                    "success": False,
                    "table": table,
                    "rows": [],
                    "row_count": 0,
                    "error": f"Filter value for '{k}' must be a scalar, not {type(v).__name__}.",
                }
        rows = _apply_filter(rows, filter)

    rows = rows[:limit]
    return {
        "success": True,
        "table": table,
        "rows": rows,
        "row_count": len(rows),
        "error": None,
    }


def insert_record(table: str, record: dict[str, Any]) -> dict[str, Any]:
    """
    Append a record to a dummy database table.

    Parameters
    ----------
    table:  Target table name.
    record: Dict of column→value pairs.  An ``id`` is generated if absent.

    Returns
    -------
    dict with keys: ``success``, ``table``, ``inserted_id``, ``error``.
    """
    if table not in _KNOWN_TABLES:
        return {
            "success": False,
            "table": table,
            "inserted_id": None,
            "error": f"Unknown table '{table}'.",
        }

    new_row = copy.deepcopy(record)
    # Auto-generate a primary key if the caller didn't supply one
    pk_field = f"{table.rstrip('s')}_id"  # accounts→account_id, orders→order_id
    if pk_field not in new_row:
        new_row[pk_field] = str(uuid.uuid4())[:8].upper()

    new_row["_created_at"] = datetime.now(timezone.utc).isoformat()
    _DB[table].append(new_row)

    return {
        "success": True,
        "table": table,
        "inserted_id": new_row.get(pk_field),
        "error": None,
    }


def get_user_profile(user_id: str) -> dict[str, Any]:
    """
    Fetch a single account record by ``owner`` field (maps to user_id).

    Parameters
    ----------
    user_id:
        The user identity string (e.g. ``"user-alice"``).

    Returns
    -------
    dict with keys: ``success``, ``profile``, ``orders``, ``error``.
    """
    if not user_id or not isinstance(user_id, str):
        return {"success": False, "profile": None, "orders": [], "error": "user_id must be a non-empty string."}

    accounts = _apply_filter(_DB["accounts"], {"owner": user_id})
    if not accounts:
        return {
            "success": False,
            "profile": None,
            "orders": [],
            "error": f"No account found for user_id '{user_id}'.",
        }

    account = copy.deepcopy(accounts[0])
    orders = _apply_filter(_DB["orders"], {"account_id": account["account_id"]})

    return {
        "success": True,
        "profile": account,
        "orders": copy.deepcopy(orders),
        "error": None,
    }