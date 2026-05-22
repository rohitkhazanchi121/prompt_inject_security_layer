# llm-shield

**Production-grade demonstration of LLM prompt injection prevention.**

A fully runnable Python project showing how to build a four-layer defence
pipeline around an Anthropic Claude integration.  Every layer is real,
testable code — not slides.

```
┌─────────────────────────────────────────────────┐
│              User / API request                 │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────▼─────────────┐
        │  Layer 1 · Input Valid.  │  < 1 ms · pure Python
        │  Pattern · Length · Enc. │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │  Layer 4 · RBAC (AuthN)  │  < 1 ms · user store lookup
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │  Layer 2 · Prompt Valid. │  ~300 ms · LLM classifier call
        │  Intent · Canary · Scope │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │     Langchain (Gemini).  |
        |  (main call)             │  agentic tool-use loop
        └──────┬──────────┬────────┘
               │          │ tool requests
        ┌──────▼──┐  ┌────▼────────────────┐
        │Layer 4b │  │ Layer 3 · Tool Ctrl │
        │AuthZ    │  │ Allowlist·Params·   │
        └─────────┘  │ Scrub · Quota       │
                     └─────────────────────┘
```

---

## Features

| Layer | What it stops |
|---|---|
| **Input Validation** | 17 injection pattern regexes, blocked-substring list, encoding attacks, oversized inputs |
| **Prompt Validation** | LLM-based intent classifier, canary token exfiltration, scope-boundary tag injection |
| **Tool Control** | Tool allowlist per role, path traversal, parameter type/length guards, PII output scrubbing, per-request quota |
| **RBAC** | Role-based tool access (`guest` / `user` / `admin`), suspended account rejection, structured audit log |

---

## Quick start

### 1. Install

```bash
git clone https://github.com/your-org/llm-shield
cd llm-shield

# Poetry (recommended)
poetry install

# or pip
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC__API_KEY=sk-ant-...
```

### 3. Run the demo

```bash
# List all 10 attack scenarios
python -m demo.cli list

# Run a single scenario (no API key needed — dry-run by default)
python -m demo.cli run S01

# Run all 10 scenarios with summary table
python -m demo.cli run-all

# Interactive chat REPL (as a specific user)
python -m demo.cli chat --user user-alice
python -m demo.cli chat --user admin-carol

# Show tool registry + RBAC matrix
python -m demo.cli tools

# Architecture summary
python -m demo.cli report
```

### 4. Run the test suite

```bash
pytest tests/ -v

# Layer-specific
pytest tests/defenses/test_input_validation.py -v
pytest tests/defenses/test_prompt_validation.py -v
```

---

## Project structure

```
llm-shield/
├── data/
│   ├── users.json              # User roster with roles
│   ├── tools_manifest.json     # Tool definitions + RBAC matrix
│   └── sensitive_records.csv   # Fixture PII for output-scrubbing demo
│
├── llm_shield/
│   ├── config.py               # Pydantic-settings (all tuneable via env)
│   ├── exceptions.py           # Typed exception hierarchy (one per layer)
│   ├── audit.py                # JSON-Lines audit logger
│   ├── pipeline.py             # Four-layer orchestrator
│   │
│   ├── layers/
│   │   ├── input_validation.py # Layer 1
│   │   ├── prompt_validation.py# Layer 2
│   │   ├── tool_control.py     # Layer 3
│   │   └── rbac.py             # Layer 4
│   │
│   └── mcp/
│       ├── server.py           # In-process tool registry + dispatcher
│       └── tools/
│           ├── filesystem.py   # read_file · write_file · list_dir · delete_file
│           ├── database.py     # query_db · insert_record · get_user_profile
│           └── search_email.py # web_search · send_email
│
├── demo/
│   ├── cli.py                  # Rich-powered CLI (typer)
│   └── scenarios.py            # 10 named attack scenarios
│
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── defenses/
│   │   ├── test_input_validation.py
│   │   └── test_prompt_validation.py
│   └── attacks/
│       ├── test_jailbreaks.py
│       ├── test_indirect.py
│       └── test_privilege_esc.py
│
└── docs/
    ├── attack_taxonomy.md
    ├── defense_layers.md
    └── rbac_design.md
```

---

## The 10 attack scenarios

| ID | Title | Layer that blocks it |
|---|---|---|
| S01 | Classic instruction override | Layer 1 — Blocked Substring |
| S02 | XML role-override tag injection | Layer 1 — Injection Pattern |
| S03 | Prompt leak via markdown delimiter | Layer 1 — Injection Pattern |
| S04 | DAN jailbreak attempt | Layer 1 — Blocked Substring |
| S05 | Scope boundary tag injection | Layer 2 — Scope Boundary |
| S06 | Data exfiltration (intent classifier) | Layer 2 — Intent Classifier |
| S07 | Path traversal in filesystem tool | Layer 3 — Parameter Sandboxing |
| S08 | Tool quota exhaustion (DoS) | Layer 3 — Quota |
| S09 | Guest accessing user-only DB tool | Layer 4 — RBAC Authorisation |
| S10 | User attempting admin-only email tool | Layer 4 — RBAC Authorisation |

---

## Configuration reference

All values are overridable via environment variables or `.env`:

```bash
# Core
ENVIRONMENT=development          # development | staging | production
LOG_LEVEL=INFO
DEBUG=false

# LLM (langchain implementation, provide any one of the APIKey, model and provider (Gemini, OpenAI, Anthropic))
LLM_ANTHROPIC_API_KEY=sk-ant-...
LLM_OPENAI_API_KEY=sk-...
LLM_GEMINI_API_KEY=AI...
LLM_MODEL=claude-haiku-4-5-20251001   
LLM_PROVIDER=anthropic

# Layer 1
IV_MAX_INPUT_LENGTH=8000
IV_ENABLE_PATTERN_DETECTION=true
IV_PATTERN_DETECTION_THRESHOLD=1

# Layer 2
PV_ENABLE_INTENT_CLASSIFICATION=true
PV_INTENT_CLASSIFICATION_MODEL=claude-haiku-4-5-20251001
PV_CANARY_TOKEN=SHIELD-CANARY-7f3a
PV_ENABLE_CANARY_CHECK=true

# Layer 3
TC_MAX_TOOL_CALLS_PER_REQUEST=5
TC_ENABLE_OUTPUT_SCRUBBING=true

# Layer 4
RBAC_DEFAULT_ROLE=guest
RBAC_ENABLE_AUDIT_LOG=true
RBAC_AUDIT_LOG_PATH=logs/audit.jsonl
```

---

## Extending the project

### Add a new injection pattern (Layer 1)

In `llm_shield/layers/input_validation.py`, add to `_INJECTION_PATTERNS`:

```python
("my_new_pattern", r"(?i)your-regex-here"),
```

### Add a new tool

1. Implement the handler in `llm_shield/mcp/tools/`.
2. Register it in `llm_shield/mcp/server.py` → `_register_all()`.
3. Add it to `data/tools_manifest.json` with `allowed_roles`.

### Add a new role

1. Add the role to `ROLE_ORDER` in `llm_shield/layers/rbac.py`.
2. Add users with that role to `data/users.json`.
3. Update `data/tools_manifest.json` with the new role in each tool's `allowed_roles`.

### Swap to a real MCP server

See the comment block at the bottom of `llm_shield/mcp/server.py` for the
FastMCP swap-in pattern.

---

## Audit log

Every request writes a structured JSON-Lines entry to `logs/audit.jsonl`:

```json
{
  "ts": "2024-09-01T12:34:56.789000+00:00",
  "event_id": "...",
  "request_id": "uuid",
  "user_id": "user-alice",
  "role": "user",
  "session_id": null,
  "outcome": "blocked",
  "layer": "input",
  "action": "pipeline_blocked",
  "tool_name": null,
  "error_code": "INJECTION_PATTERN_DETECTED",
  "error_detail": "Input contains 1 injection pattern(s). Request blocked.",
  "latency_ms": 0.43,
  "metadata": {}
}
```

---

## Tech stack

| Concern | Library |
|---|---|
| LLM | `anthropic` SDK |
| Config | `pydantic-settings` v2 |
| Schemas | `pydantic` v2 |
| CLI | `typer` + `rich` |
| Testing | `pytest` + `pytest-asyncio` |
| Packaging | `pyproject.toml` (Poetry) |

---

## Security notes

This project is a **demonstration**.  For production deployment:

- Replace `NullAuditLogger` with a queue-backed async logger.
- Add rate limiting at the API gateway layer.
- Store `users.json` in a real identity provider (Auth0, Cognito, etc.).
- Use secrets management (AWS Secrets Manager, Vault) for the API key.
- Enable `PV_ENABLE_INTENT_CLASSIFICATION=true` — the classifier adds ~300 ms
  latency but catches sophisticated attacks that patterns miss.
- Set `ENVIRONMENT=production` to enable stricter defaults.

---

## References

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Anthropic Prompt Injection Guidance](https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/prompt-injection)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [PromptInject Benchmark](https://github.com/agencyenterprise/promptinject)

---

## License

MIT — see `LICENSE`.