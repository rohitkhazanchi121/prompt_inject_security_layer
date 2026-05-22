"""
llm_shield.config
=================
Centralised configuration via pydantic-settings.

All values can be overridden by environment variables (or a .env file).
The module exposes a single module-level ``settings`` singleton so every
other module can do::

    from llm_shield.config import settings

and receive the same, already-validated object.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, Literal
import os
from pydantic import Field, field_validator, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

# Supported providers — extend this tuple as you add new ones
Provider = Literal["openai", "anthropic", "google-genai"]


class Environment(str, Enum):
    """Runtime environment tag — controls log verbosity and strictness."""

    development = "development"
    staging = "staging"
    production = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Sub-settings groups
# ---------------------------------------------------------------------------


class AnthropicSettings(BaseSettings):
    """Anthropic API credentials and model selection."""

    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_")

    api_key: str = Field(
        default="",
        description="Anthropic API key.  Required in staging / production.",
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model identifier used for prompt-validation and the demo pipeline.",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum tokens for validation calls.",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="HTTP timeout for Anthropic API calls.",
    )

class LLMSettings(BaseSettings):
    """Anthropic API credentials and model selection.
    
    Typical .env entries
    --------------------
    LLM_ANTHROPIC_API_KEY=sk-ant-...
    LLM_OPENAI_API_KEY=sk-...
    LLM_GEMINI_API_KEY=AI...
    LLM_MODEL=claude-haiku-4-5-20251001   
    LLM_PROVIDER=anthropic              
    """
    
    # Allows pulling global variables or specific prefixes if preferred
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Define the API keys as fields so Pydantic reads them from the .env file
    # Using SecretStr ensures keys don't accidentally leak into your log files
    gemini_api_key: Optional[SecretStr] = Field(default=None, description="Anthropic API key (env: LLM_ANTHROPIC_API_KEY).",)
    openai_api_key: Optional[SecretStr] = Field(default=None, description="OpenAI API key (env: LLM_OPENAI_API_KEY).",)
    anthropic_api_key: Optional[SecretStr] = Field(default=None, description="Google Gemini API key (env: LLM_GEMINI_API_KEY).",)

    # Core identification keys
    model: Optional[str] = Field(
        default=None,
        description="Override to explicitly set model name (e.g., 'gemini-2.5-flash', 'gpt-4o-mini')."
    )
    provider: Optional[Literal[Provider]] = Field(
        default=None,
        description="Override to explicitly lock in a provider strategy."
    )

    # Global runtime configurations matching LangChain standard parameter names
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    timeout: float = Field(default=30.0, gt=0, description="HTTP timeout in seconds.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def _validate_provider_model_coherence(self) -> "LLMSettings":
        """
        Catch misconfigurations at startup rather than at first inference call.

        Rules:
        - If provider is set, model should also be set (warn otherwise — we'll
          use the provider default, but explicit is better).
        - If provider is set explicitly, the corresponding API key must exist.
        - At least one API key or explicit model must be resolvable.
        """
        if self.provider and not self.model:
            # Not an error — we have a default for every provider —
            # but surface it so it's visible in config dumps.
            pass  # could add a warning here via structlog

        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "provider='anthropic' requires LLM_ANTHROPIC_API_KEY to be set."
            )
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "provider='openai' requires LLM_OPENAI_API_KEY to be set."
            )
        if self.provider == "google-genai" and not self.gemini_api_key:
            raise ValueError(
                "provider='google-genai' requires LLM_GEMINI_API_KEY to be set."
            )
        
        # Ensure at least one path to a model exists
        try:
            self._resolve_provider()
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        return self

    def _resolve_provider(self) -> tuple[str, Optional[str]]:
        """
        Determines the model name and provider based on explicit overrides 
        or active API keys found in the environment.
        """
        # 1. Respect explicit manual overrides first
        if self.model and self.provider:
            return self.model, self.provider
        
        # 2. Explicit model only — infer provider from model name prefix
        if self.model and not self.provider:
            if "gemini" in self.model:
                return self.model, "google-genai"
            if "gpt" in self.model or "o1" in self.model or "o3" in self.model:
                return self.model, "openai"
            if "claude" in self.model:
                return self.model, "anthropic"
            raise ValueError(
                f"Cannot infer provider from model name '{self.model}'. "
                "Set LLM_PROVIDER explicitly."
            )
            
        raise ValueError("No LLM API keys or explicit models were found by Pydantic.")
    
    def _get_api_key_for(self, provider: Provider) -> Optional[str]:
        """Extract the secret value for the resolved provider."""
        key_map: dict[Provider, Optional[SecretStr]] = {
            "anthropic":   self.anthropic_api_key,
            "openai":      self.openai_api_key,
            "google-genai": self.gemini_api_key,
        }
        secret = key_map.get(provider)
        return secret.get_secret_value() if secret else None
    

    @property
    def resolved_model(self) -> str:
        """The model name that will be used — useful for logging at startup."""
        return self._resolve_provider()[0]

    @property
    def resolved_provider(self) -> Provider:
        """The provider that will be used — useful for logging at startup."""
        return self._resolve_provider()[1]
    
    def get_chat_model(self):
        """
        Builds and returns the dynamically selected LangChain LLM instance.
        """
        from langchain.chat_models import init_chat_model  # lazy import
        
        model_name, provider_name = self._resolve_provider()
        api_key = self._get_api_key_for(provider_name)

        return init_chat_model(
            model=model_name,
            model_provider=provider_name,
            api_key=api_key,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            temperature=self.temperature,
        )
    
class InputValidationSettings(BaseSettings):
    """Tuneable knobs for Layer 1 — Input Validation."""

    model_config = SettingsConfigDict(env_prefix="IV_")

    max_input_length: int = Field(
        default=8_000,
        ge=1,
        description="Hard upper limit on raw user input (characters).",
    )
    max_prompt_tokens: int = Field(
        default=2_000,
        ge=1,
        description="Rough token budget; 1 token ≈ 4 chars is used as the heuristic.",
    )
    # Injection pattern detection
    enable_pattern_detection: bool = Field(
        default=True,
        description="Whether to run regex-based injection pattern scanning.",
    )
    pattern_detection_threshold: int = Field(
        default=1,
        ge=1,
        description="Minimum number of pattern hits required to flag the input.",
    )
    # Encoding / normalisation
    normalise_unicode: bool = Field(
        default=True,
        description="Apply NFC normalisation before any other check.",
    )
    strip_null_bytes: bool = Field(
        default=True,
        description="Remove null bytes (\\x00) which can confuse parsers.",
    )
    # Allow / block lists (comma-separated strings from env)
    blocked_substrings: list[str] = Field(
        default_factory=lambda: [
            "ignore previous instructions",
            "ignore all previous",
            "disregard your instructions",
            "you are now",
            "act as if",
            "pretend you are",
            "new persona",
            "jailbreak",
            "dan mode",
            "dan",
            "developer mode",
        ],
        description="Literal substrings that are always blocked (case-insensitive).",
    )


class PromptValidationSettings(BaseSettings):
    """Tuneable knobs for Layer 2 — Prompt Validation."""

    model_config = SettingsConfigDict(env_prefix="PV_")

    enable_intent_classification: bool = Field(
        default=True,
        description="Use an LLM call to classify prompt intent.",
    )
    intent_classification_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Smaller / cheaper model used for fast intent classification.",
    )
    canary_token: str = Field(
        default="SHIELD-CANARY-7f3a",
        description=(
            "Token injected into the system prompt.  "
            "If the model echoes it in user-visible output, integrity has been breached."
        ),
    )
    enable_canary_check: bool = Field(
        default=True,
        description="Whether to scan model outputs for canary token leakage.",
    )
    scope_tags: list[str] = Field(
        default_factory=lambda: ["SYSTEM", "USER", "TOOL_RESULT"],
        description="Valid scope boundary tags that may appear in a prompt.",
    )


class ToolControlSettings(BaseSettings):
    """Tuneable knobs for Layer 3 — Tool Control."""

    model_config = SettingsConfigDict(env_prefix="TC_")

    max_tool_calls_per_request: int = Field(
        default=5,
        ge=1,
        description="Upper bound on tool invocations within a single pipeline run.",
    )
    enable_output_scrubbing: bool = Field(
        default=True,
        description="Whether to redact PII / secrets from tool outputs.",
    )
    # Patterns that look like secrets in tool outputs
    sensitive_output_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",  # email
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b",  # credit card
            r"(?i)(?:password|passwd|secret|api[_\-]?key)\s*[:=]\s*\S+",  # creds
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IPv4
        ],
        description="Regex patterns; matches are replaced with [REDACTED].",
    )
    
    tool_quota_behaviour: Literal["raise", "graceful"] = Field(
    default="raise",
    description=(
        "'raise' — pipeline errors on quota exceeded (strict, good for adversarial contexts). "
        "'graceful' — generation loop exits and model summarises with results so far."
        ),
    )


class RBACSettings(BaseSettings):
    """Tuneable knobs for Layer 4 — RBAC."""

    model_config = SettingsConfigDict(env_prefix="RBAC_")

    default_role: str = Field(
        default="guest",
        description="Role assigned when no identity is provided.",
    )
    enable_audit_log: bool = Field(
        default=True,
        description="Write a structured JSON audit entry for every request.",
    )
    audit_log_path: Path = Field(
        default=Path("logs/audit.jsonl"),
        description="Path to the append-only audit log (JSON Lines format).",
    )


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root settings object.

    Load order (last wins):
      1. Field defaults
      2. .env file  (if present)
      3. Environment variables
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",   # e.g. ANTHROPIC__MODEL=... maps to anthropic.model
        case_sensitive=False,
        extra="ignore",
    )

    # ── Meta ────────────────────────────────────────────────────────────────
    environment: Environment = Field(
        default=Environment.development,
        description="Runtime environment.",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level emitted by structlog.",
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose debug output (overrides log_level → DEBUG).",
    )

    # ── Data paths ───────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory containing JSON / CSV fixture files.",
    )

    # ── Sub-configs ──────────────────────────────────────────────────────────
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    llm: LLMSettings= Field(default_factory= LLMSettings)
    input_validation: InputValidationSettings = Field(
        default_factory=InputValidationSettings
    )
    prompt_validation: PromptValidationSettings = Field(
        default_factory=PromptValidationSettings
    )
    tool_control: ToolControlSettings = Field(default_factory=ToolControlSettings)
    rbac: RBACSettings = Field(default_factory=RBACSettings)

    # ── Derived helpers ──────────────────────────────────────────────────────
    @field_validator("log_level", mode="before")
    @classmethod
    def _coerce_log_level(cls, v: object) -> object:
        if isinstance(v, str):
            return v.upper()
        return v

    @property
    def effective_log_level(self) -> str:
        """Returns DEBUG when debug=True, otherwise the configured log_level."""
        return "DEBUG" if self.debug else self.log_level.value

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.production

    @property
    def users_file(self) -> Path:
        return self.data_dir / "users.json"

    @property
    def tools_manifest_file(self) -> Path:
        return self.data_dir / "tools_manifest.json"

    @property
    def sensitive_records_file(self) -> Path:
        return self.data_dir / "sensitive_records.csv"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global settings instance — import this everywhere.
settings: Settings = Settings()