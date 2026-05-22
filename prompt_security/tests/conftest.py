"""
conftest.py
===========
Shared pytest fixtures available to every test module in the suite.

Fixtures defined here are automatically discovered by pytest — no import needed.

Fixture catalogue
-----------------
``valid_user_input``         — a pre-built, already-validated UserInput.
``sanitised_input``          — a SanitisedInput ready for Layer 2.
``guest_identity``           — RequestIdentity with role='guest'.
``user_identity``            — RequestIdentity with role='user'.
``admin_identity``           — RequestIdentity with role='admin'.
``no_pattern_iv_settings``   — InputValidationSettings with patterns disabled.
``no_api_pv_settings``       — PromptValidationSettings with LLM calls disabled.
"""

from __future__ import annotations

import pytest

from llm_shield.config import InputValidationSettings, PromptValidationSettings
from llm_shield.layers.input_validation import (
    InputValidator,
    SanitisedInput,
    UserInput,
)


# ---------------------------------------------------------------------------
# Layer 1 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_user_input() -> UserInput:
    """A well-formed UserInput that passes all Layer 1 checks."""
    return UserInput(
        content="What is the current balance on account #ACC-001?",
        user_id="test-user-42",
        session_id="sess-abc123",
        metadata={"source": "test-suite"},
    )


@pytest.fixture()
def sanitised_input(valid_user_input: UserInput) -> SanitisedInput:
    """A SanitisedInput produced by running the default InputValidator."""
    cfg = InputValidationSettings(
        enable_pattern_detection=False,
        blocked_substrings=[],
    )
    return InputValidator(cfg=cfg).validate(valid_user_input)


# ---------------------------------------------------------------------------
# Settings fixtures (no-network variants for CI)
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_pattern_iv_settings() -> InputValidationSettings:
    """InputValidationSettings with regex patterns and blocklist disabled."""
    return InputValidationSettings(
        enable_pattern_detection=False,
        blocked_substrings=[],
    )


@pytest.fixture()
def no_api_pv_settings() -> PromptValidationSettings:
    """PromptValidationSettings that skips LLM-based intent classification."""
    return PromptValidationSettings(
        enable_intent_classification=False,
        enable_canary_check=True,
        canary_token="TEST-CANARY-XYZ",
    )