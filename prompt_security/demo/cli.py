"""
demo.cli
=========
Interactive CLI demo for llm-shield.

Usage
-----
    # Run all 10 scenarios (attack + safe equivalent)
    python -m demo.cli run-all

    # Run a single scenario by ID
    python -m demo.cli run S01

    # List all scenarios
    python -m demo.cli list

    # Show the audit log tail
    python -m demo.cli audit

    # Interactive REPL — type messages as different users
    python -m demo.cli chat --user user-alice
    python -m demo.cli chat --user admin-carol

    # Print the security report
    python -m demo.cli report
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Add repo root to sys.path so `python -m demo.cli` works from anywhere
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from llm_shield.audit import NullAuditLogger
from llm_shield.config import InputValidationSettings, PromptValidationSettings
from llm_shield.exceptions import LLMShieldError
from llm_shield.layers.input_validation import InputValidator, UserInput
from llm_shield.layers.prompt_validation import PromptValidator
from llm_shield.layers.rbac import RBACController, allowed_tools_for
from llm_shield.layers.tool_control import CallCounter, ToolController
from llm_shield.mcp.server import registry
from llm_shield.pipeline import Pipeline, PipelineRequest, RunMode
from demo.scenarios import SCENARIOS, SCENARIO_MAP, Scenario

# ---------------------------------------------------------------------------
# Rich console
# ---------------------------------------------------------------------------

console = Console(highlight=True)
app = typer.Typer(
    name="llm-shield",
    help="Production-grade prompt injection prevention demo.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_TITLE   = "bold cyan"
C_OK      = "bold green"
C_BLOCK   = "bold red"
C_WARN    = "bold yellow"
C_LAYER   = "bold blue"
C_MUTED   = "dim white"
C_PAYLOAD = "italic yellow"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_pipeline(dry_run: bool = True) -> Pipeline:
    """Build a pipeline configured for the demo (dry_run skips Claude calls)."""
    # Layer 1: strict
    iv = InputValidator()

    # Layer 2: intent classification disabled (no API key needed for demo)
    pv_cfg = PromptValidationSettings(
        enable_intent_classification=True,
        enable_canary_check=True,
    )
    
    pv = PromptValidator(pv_cfg=pv_cfg)

    # Layer 4: real RBAC, null audit logger (writes to console instead)
    null_audit = NullAuditLogger()
    rbac = RBACController(audit_logger=null_audit)

    # Layer 3: real tool control
    tc = ToolController()

    return Pipeline(
        input_validator=iv,
        prompt_validator=pv,
        rbac=rbac,
        tool_controller=tc,
        audit_logger=null_audit,
    )


def _header(title: str, subtitle: str = "") -> None:
    console.print()
    console.rule(f"[{C_TITLE}]{title}[/]")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/]")
    console.print()


def _scenario_banner(s: Scenario, index: int, total: int) -> None:
    console.print(
        Panel(
            f"[{C_TITLE}]{s.id}  {s.title}[/]\n"
            f"[{C_MUTED}]Category: {s.category.upper()}   "
            f"Expected block: {s.expected_layer}[/]\n\n"
            f"{s.explanation}",
            title=f"[{C_TITLE}]Scenario {index}/{total}[/]",
            border_style="cyan",
            expand=False,
        )
    )


def _print_attack(payload: str, user_id: str) -> None:
    console.print(f"  [bold]Attack payload[/]  [{C_PAYLOAD}](user: {user_id})[/]")
    console.print(
        Panel(payload, border_style="red", expand=False, padding=(0, 2))
    )


def _print_blocked(exc: LLMShieldError) -> None:
    console.print(
        f"  [{C_BLOCK}]✗ BLOCKED[/]  "
        f"code=[bold]{exc.code}[/]  "
        f"message={exc.message[:100]}"
    )


def _print_passed(msg: str) -> None:
    console.print(f"  [{C_OK}]✓ PASSED[/]  {msg}")


def _print_safe(payload: str, user_id: str) -> None:
    console.print(f"\n  [bold]Safe equivalent[/]  [{C_MUTED}](user: {user_id})[/]")
    console.print(
        Panel(payload, border_style="green", expand=False, padding=(0, 2))
    )


def _run_one_scenario(s: Scenario, pipeline: Pipeline) -> dict:
    """Run attack + safe for a single scenario. Returns result dict."""
    result = {"scenario_id": s.id, "attack_blocked": False, "safe_passed": False}

    # ── Attack ─────────────────────────────────────────────────────────────
    _print_attack(s.attack_payload, s.attack_user_id)
    try:
        req = PipelineRequest(
            user_input=UserInput(content=s.attack_payload, user_id=s.attack_user_id),
            mode=RunMode.full,
        )
        pipeline.run(req)
        console.print(f"  [{C_WARN}]⚠ NOT BLOCKED (unexpected)[/]")
    except LLMShieldError as exc:
        _print_blocked(exc)
        result["attack_blocked"] = True
    except Exception as exc:
        console.print(f"  [{C_WARN}]⚠ Unexpected error: {exc}[/]")

    # ── Safe ───────────────────────────────────────────────────────────────
    _print_safe(s.safe_equivalent, s.safe_user_id)
    try:
        req = PipelineRequest(
            user_input=UserInput(content=s.safe_equivalent, user_id=s.safe_user_id),
            mode=RunMode.dry_run,
        )
        pipeline.run(req)
        _print_passed("Request accepted — all layers passed.")
        result["safe_passed"] = True
    except LLMShieldError as exc:
        console.print(f"  [{C_WARN}]⚠ False positive — blocked: {exc.code}[/]")
    except Exception as exc:
        console.print(f"  [{C_WARN}]⚠ Unexpected error: {exc}[/]")

    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def list_scenarios() -> None:
    """List all available demo scenarios."""
    _header("llm-shield — Scenario Catalogue")

    table = Table(box=box.ROUNDED, show_lines=True, expand=False)
    table.add_column("ID",       style="bold cyan",  width=5)
    table.add_column("Category", style="bold blue",  width=10)
    table.add_column("Title",    style="white",       width=40)
    table.add_column("Blocks at",style="yellow",      width=30)

    for s in SCENARIOS:
        table.add_row(s.id, s.category.upper(), s.title, s.expected_layer)

    console.print(table)
    console.print(f"\n  [dim]Run a scenario:  python -m demo.cli run S01[/]")
    console.print(f"  [dim]Run all:         python -m demo.cli run-all[/]\n")


@app.command(name="run")
def run_scenario(
    scenario_id: str = typer.Argument(..., help="Scenario ID e.g. S01"),
) -> None:
    """Run a single attack scenario (attack + safe equivalent)."""
    s = SCENARIO_MAP.get(scenario_id.upper())
    if s is None:
        console.print(f"[{C_BLOCK}]Unknown scenario '{scenario_id}'. Use 'list' to see options.[/]")
        raise typer.Exit(1)

    _header(f"llm-shield — Running {s.id}: {s.title}")
    pipeline = _build_pipeline(dry_run=True)
    _scenario_banner(s, 1, 1)
    _run_one_scenario(s, pipeline)
    console.print()


@app.command(name="run-all")
def run_all(
    pause: float = typer.Option(0.4, help="Seconds to pause between scenarios."),
) -> None:
    """Run all 10 attack scenarios and print a summary table."""
    _header(
        "llm-shield — Full Demo Run",
        f"{len(SCENARIOS)} scenarios: attack payloads + safe equivalents",
    )

    pipeline = _build_pipeline(dry_run=True)
    results: list[dict] = []

    for i, s in enumerate(SCENARIOS, start=1):
        _scenario_banner(s, i, len(SCENARIOS))
        r = _run_one_scenario(s, pipeline)
        results.append(r)
        console.print()
        time.sleep(pause)

    # ── Summary ─────────────────────────────────────────────────────────────
    console.rule(f"[{C_TITLE}]Summary[/]")
    table = Table(box=box.ROUNDED, expand=False)
    table.add_column("ID",             style="bold cyan", width=5)
    table.add_column("Title",          width=40)
    table.add_column("Attack blocked", width=16, justify="center")
    table.add_column("Safe passed",    width=14, justify="center")

    blocked_count = 0
    passed_count = 0
    for s, r in zip(SCENARIOS, results):
        ab = r["attack_blocked"]
        sp = r["safe_passed"]
        blocked_count += int(ab)
        passed_count  += int(sp)
        table.add_row(
            s.id,
            s.title,
            f"[{C_OK}]✓[/]" if ab else f"[{C_BLOCK}]✗[/]",
            f"[{C_OK}]✓[/]" if sp else f"[{C_WARN}]✗[/]",
        )

    console.print(table)
    console.print(
        f"\n  Attacks blocked: [{C_OK}]{blocked_count}/{len(SCENARIOS)}[/]   "
        f"Safe requests passed: [{C_OK}]{passed_count}/{len(SCENARIOS)}[/]\n"
    )


@app.command()
def chat(
    user: str  = typer.Option("user-alice", help="User ID to send messages as."),
    no_api: bool = typer.Option(True, help="Skip real Claude API calls (dry-run)."),
) -> None:
    """
    Interactive REPL — type messages as a specific user and see which
    layer blocks or passes each request.
    """
    _header(
        "llm-shield — Interactive Chat",
        f"User: {user}   Mode: {'dry-run (no API)' if no_api else 'full (calls Claude)'}",
    )

    pipeline = _build_pipeline(dry_run=no_api)
    mode = RunMode.dry_run if no_api else RunMode.full

    # Show available tools for this user's role
    rbac = RBACController(audit_logger=NullAuditLogger())
    try:
        identity = rbac.authenticate(user)
        tools = allowed_tools_for(identity.role)
        console.print(
            f"  Role: [bold]{identity.role}[/]   "
            f"Permitted tools: [dim]{', '.join(tools) or 'none'}[/]\n"
        )
    except LLMShieldError as exc:
        console.print(f"[{C_BLOCK}]Authentication failed: {exc.message}[/]")
        raise typer.Exit(1)

    console.print("  Type your message and press Enter.  [dim]Ctrl-C or 'exit' to quit.[/]\n")

    while True:
        try:
            message = console.input("  [bold cyan]>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Goodbye.[/]\n")
            break

        if message.lower() in ("exit", "quit", "q"):
            console.print("  [dim]Goodbye.[/]\n")
            break
        if not message:
            continue

        t0 = time.perf_counter()
        try:
            req = PipelineRequest(
                user_input=UserInput(content=message, user_id=user),
                mode=mode,
            )
            resp = pipeline.run(req)
            ms = (time.perf_counter() - t0) * 1000
            console.print(
                f"\n  [{C_OK}]✓ All {len(resp.layers_passed)} layers passed[/]  "
                f"[dim]({ms:.0f} ms)[/]"
            )
            if resp.answer and resp.answer != "[DRY RUN — no Claude call made]":
                console.print(Panel(resp.answer, border_style="green", expand=False))
            else:
                console.print(f"  [dim]{resp.answer}[/]")
        except LLMShieldError as exc:
            ms = (time.perf_counter() - t0) * 1000
            console.print(
                f"\n  [{C_BLOCK}]✗ BLOCKED[/]  [bold]{exc.code}[/]  [dim]({ms:.0f} ms)[/]"
            )
            console.print(f"  {exc.message}\n")
        except Exception as exc:
            console.print(f"\n  [{C_WARN}]⚠ Error:[/] {exc}\n")


@app.command()
def tools() -> None:
    """Show the MCP tool registry and per-role permissions."""
    _header("llm-shield — MCP Tool Registry")

    table = Table(box=box.ROUNDED, show_lines=True, expand=False)
    table.add_column("Tool",        style="bold cyan", width=20)
    table.add_column("Category",    style="blue",      width=12)
    table.add_column("Description", width=42)
    table.add_column("guest", justify="center", width=7)
    table.add_column("user",  justify="center", width=7)
    table.add_column("admin", justify="center", width=7)

    for t in sorted(registry.all_tools(), key=lambda x: (x.category, x.name)):
        def _tick(role: str) -> str:
            return f"[{C_OK}]✓[/]" if role in t.allowed_roles else f"[{C_BLOCK}]✗[/]"

        table.add_row(
            t.name,
            t.category,
            t.description[:42],
            _tick("guest"),
            _tick("user"),
            _tick("admin"),
        )

    console.print(table)
    console.print()


@app.command()
def report() -> None:
    """Print a security architecture summary."""
    _header("llm-shield — Security Architecture Report")

    layers = [
        ("Layer 1", "Input Validation",  "< 1 ms", "Pattern detection, length guard, blocklist, encoding"),
        ("Layer 2", "Prompt Validation", "~300 ms", "Intent classifier (LLM), canary tokens, scope boundaries"),
        ("Layer 3", "Tool Control",      "< 1 ms", "Allowlist, parameter sandboxing, output scrubbing, quota"),
        ("Layer 4", "RBAC",             "< 1 ms", "Role resolution, per-tool authorisation, audit log"),
    ]

    table = Table(box=box.ROUNDED, expand=False)
    table.add_column("Layer",   style="bold cyan", width=8)
    table.add_column("Name",    style="bold",      width=20)
    table.add_column("Latency", style="dim",       width=10)
    table.add_column("Checks",                     width=54)

    for layer, name, latency, checks in layers:
        table.add_row(layer, name, latency, checks)

    console.print(table)

    console.print()
    console.print("  [bold]Roles & permissions[/]")
    for role in ["guest", "user", "admin"]:
        tools = allowed_tools_for(role)
        console.print(f"  [bold cyan]{role:8}[/]  {', '.join(tools) or '(none)'}")

    console.print()
    console.print("  [bold]Dummy MCP tools[/]")
    for t in sorted(registry.all_tools(), key=lambda x: x.name):
        console.print(f"  [cyan]{t.name:20}[/]  [dim]{t.category}[/]  {t.description}")

    console.print(f"\n  Run [bold]python -m demo.cli run-all[/] to see all 10 attack scenarios.\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()