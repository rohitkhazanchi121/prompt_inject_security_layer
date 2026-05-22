"""
llm_shield
==========
Production-grade demonstration of LLM prompt injection prevention.

Defence layers
--------------
1. Input Validation   — encoding, length, blocklist, injection patterns
2. Prompt Validation  — intent classification, canary tokens, scope boundaries
3. Tool Control       — allowlist, parameter sandboxing, output scrubbing
4. RBAC               — role/permission matrix, audit logging

Quick start::

    from llm_shield.layers.input_validation import validate_raw
    from llm_shield.layers.prompt_validation import PromptValidator
    from llm_shield.pipeline import Pipeline

Public re-exports (most callers only need these)::

    from llm_shield import (
        settings,           # global Settings singleton
        Pipeline,           # full 4-layer pipeline
        UserInput,          # input model
        LLMShieldError,     # base exception — catch all shield errors here
    )
"""

from __future__ import annotations

from llm_shield.config import settings
from llm_shield.exceptions import LLMShieldError
from llm_shield.layers.input_validation import UserInput

__version__ = "0.1.0"
__all__ = [
    "settings",
    "LLMShieldError",
    "UserInput",
    "__version__",
]