"""
llm_shield.mcp.tools.search
==============================
Dummy web-search tool — searches an in-process knowledge base loaded from
``data/sensitive_records.csv``.  No real HTTP calls are made.

Tool exposed
------------
``web_search(query, max_results)`` — keyword search over the KB.

llm_shield.mcp.tools.email_tool
================================
Dummy send-email tool — writes to the audit log only, never sends real mail.

Tool exposed
------------
``send_email(to, subject, body)`` — log-only email simulation.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from llm_shield.config import settings


# ===========================================================================
# Search tool
# ===========================================================================

# ---------------------------------------------------------------------------
# Knowledge-base seeded with inline records so the tool works even if the
# CSV hasn't been generated yet.  The CSV loader supplements this with
# whatever is in data/sensitive_records.csv.
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: list[dict[str, str]] = [
    {
        "id": "KB-001",
        "title": "Return & Refund Policy",
        "body": (
            "Acme Corp accepts returns within 30 days of purchase. "
            "Items must be in original packaging. Refunds are processed within 5 business days."
        ),
        "category": "policy",
        "tags": "return refund policy customer",
    },
    {
        "id": "KB-002",
        "title": "Widget Pro X — Technical Specifications",
        "body": (
            "Widget Pro X supports USB-C and Bluetooth 5.2. "
            "Battery life: 24 hours. Weight: 145 g. Warranty: 2 years."
        ),
        "category": "product",
        "tags": "widget pro x specs hardware",
    },
    {
        "id": "KB-003",
        "title": "Shipping & Delivery Times",
        "body": (
            "Standard shipping: 5-7 business days. Express: 2 business days. "
            "Same-day delivery available in Vancouver, Toronto, Montreal."
        ),
        "category": "policy",
        "tags": "shipping delivery times logistics",
    },
    {
        "id": "KB-004",
        "title": "Account Suspension Policy",
        "body": (
            "Accounts with overdue balances exceeding 90 days are automatically suspended. "
            "Contact support@acme.example.com to reinstate."
        ),
        "category": "policy",
        "tags": "account suspended overdue balance",
    },
    {
        "id": "KB-005",
        "title": "SuperSaaS Plan — Features",
        "body": (
            "SuperSaaS includes unlimited users, 1TB storage, SSO, and 24/7 support. "
            "Annual billing saves 20% vs monthly."
        ),
        "category": "product",
        "tags": "saas plan software features pricing",
    },
    {
        "id": "KB-006",
        "title": "Security Incident Response",
        "body": (
            "Report security incidents to security@acme.example.com within 1 hour of discovery. "
            "Do not attempt to remediate production systems without SRE approval."
        ),
        "category": "security",
        "tags": "security incident response sre",
    },
    {
        "id": "KB-007",
        "title": "Password Reset Procedure",
        "body": (
            "Visit https://accounts.acme.example.com/reset and enter your email. "
            "A reset link valid for 15 minutes will be sent."
        ),
        "category": "support",
        "tags": "password reset account access",
    },
    {
        "id": "KB-008",
        "title": "Q3 2024 Product Roadmap",
        "body": (
            "Q4 launches: Widget Pro X v2 (Oct), Cloud Storage 10TB tier (Nov), "
            "Mobile SDK beta (Dec). Internal — do not share with customers."
        ),
        "category": "internal",
        "tags": "roadmap product q4 internal",
    },
]


def _load_csv_kb() -> None:
    """Supplement the inline KB with records from the CSV fixture if present."""
    csv_path = settings.sensitive_records_file
    if not csv_path.exists():
        return
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("id") and row.get("title"):
                    _KNOWLEDGE_BASE.append(row)
    except Exception:
        pass  # graceful degradation — inline KB is sufficient for the demo


_load_csv_kb()


def _score(record: dict[str, str], terms: list[str]) -> int:
    """Simple TF-style relevance score: count term hits across searchable fields."""
    searchable = " ".join(
        [record.get("title", ""), record.get("body", ""), record.get("tags", "")]
    ).lower()
    return sum(searchable.count(t) for t in terms)


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search the dummy knowledge base with a keyword query.

    Parameters
    ----------
    query:
        Free-text search query.
    max_results:
        Maximum number of results (1–10, default 5).

    Returns
    -------
    dict with keys: ``success``, ``query``, ``results``, ``result_count``, ``error``.

    Each result has: ``id``, ``title``, ``snippet`` (first 200 chars of body),
    ``category``, ``score``.
    """
    if not query or not query.strip():
        return {
            "success": False,
            "query": query,
            "results": [],
            "result_count": 0,
            "error": "Query must not be empty.",
        }

    max_results = max(1, min(int(max_results), 10))
    terms = [t.lower() for t in re.split(r"\W+", query.strip()) if t]

    scored = [
        (record, _score(record, terms))
        for record in _KNOWLEDGE_BASE
    ]
    scored = [(r, s) for r, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = [
        {
            "id": r.get("id", ""),
            "title": r.get("title", ""),
            "snippet": r.get("body", "")[:200],
            "category": r.get("category", ""),
            "score": s,
        }
        for r, s in scored[:max_results]
    ]

    return {
        "success": True,
        "query": query,
        "results": results,
        "result_count": len(results),
        "error": None,
    }


# ===========================================================================
# Email tool
# ===========================================================================

# Audit trail of "sent" emails — inspectable in tests and the demo CLI
_EMAIL_AUDIT_LOG: list[dict[str, Any]] = []


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """
    Simulate sending an email — writes to an in-process audit log only.
    No real email is ever sent.

    Parameters
    ----------
    to:      Recipient email address.
    subject: Email subject line.
    body:    Plain-text body.

    Returns
    -------
    dict with keys: ``success``, ``message_id``, ``to``, ``subject``, ``error``.
    """
    import uuid
    from datetime import datetime, timezone

    # Basic validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", to):
        return {
            "success": False,
            "message_id": None,
            "to": to,
            "subject": subject,
            "error": f"Invalid recipient address: '{to}'",
        }

    if len(body) > 10_000:
        return {
            "success": False,
            "message_id": None,
            "to": to,
            "subject": subject,
            "error": "Email body exceeds 10 000 character limit.",
        }

    message_id = f"MSG-{uuid.uuid4().hex[:8].upper()}"
    entry = {
        "message_id": message_id,
        "to": to,
        "subject": subject,
        "body_preview": body[:120] + ("..." if len(body) > 120 else ""),
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "simulated": True,
    }
    _EMAIL_AUDIT_LOG.append(entry)

    return {
        "success": True,
        "message_id": message_id,
        "to": to,
        "subject": subject,
        "error": None,
        "note": "SIMULATED — no real email was sent.",
    }


def get_email_audit_log() -> list[dict[str, Any]]:
    """Return the in-process email audit log (for demo / test inspection)."""
    return list(_EMAIL_AUDIT_LOG)