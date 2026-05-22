"""
llm_shield.layers
=================
The four ordered defence layers of the llm-shield pipeline.

Each layer is a self-contained module with its own validator class and a
module-level convenience function.  Import directly from the submodule for
full control, or use the re-exports below for the happy path.

Layer execution order
---------------------
1. ``input_validation``  — pure Python, no network, < 1 ms
2. ``prompt_validation`` — one lightweight LLM call for intent classification
3. ``tool_control``      — enforced before every tool invocation
4. ``rbac``              — role/permission matrix checked at request entry

Typical usage::

    from llm_shield.layers import validate_input, PromptValidator
    from llm_shield.layers.input_validation import UserInput

    sanitised = validate_input(UserInput(content="...", user_id="u1"))
    clean_prompt = PromptValidator().validate(sanitised, system_prompt="...")
"""

from __future__ import annotations

# Layer 1
from llm_shield.layers.input_validation import (
    InputValidator,
    SanitisedInput,
    UserInput,
    validate as validate_input,
    validate_raw,
)
# Layer 2
from llm_shield.layers.prompt_validation import(
    PromptValidator,
    validate as validate_prompt,
    ValidatedPrompt,
    ScopeCheckResult,
    CanaryCheckResult,
    ClassificationResult,   
)


__all__ = [
    # Layer 1
    "InputValidator",
    "SanitisedInput",
    "UserInput",
    "validate_input",
    "validate_raw",
    #Layer2
    "PromptValidator",
    "validate_prompt",
    "ValidatedPrompt",
    "ScopeCheckResult",
    "CanaryCheckResult",
    "ClassificationResult",
]