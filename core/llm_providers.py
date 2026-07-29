"""Provider-agnostic structured LLM calls (Anthropic + Groq/OpenAI-compatible).

Locked experiments continue to use Anthropic/Haiku via resolve_llm_config().
One model family per experiment_id — never mix providers within a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from core.config import settings

T = TypeVar("T", bound=BaseModel)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Quality-validated model (tool calling); free-tier TPD=100k — too tight for CPR grids.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
# Higher free-tier TPD (500k) for factorial / multi-round experiment volume.
GROQ_VOLUME_MODEL = "llama-3.1-8b-instant"

_PLACEHOLDER_KEYS = frozenset({"", "your_key_here"})

# experiment_id → (provider, model). Unlisted IDs use Settings defaults (Anthropic).
EXPERIMENT_LLM_REGISTRY: dict[str, tuple[str, str]] = {
    "groq_smoke_test": ("groq", DEFAULT_GROQ_MODEL),
    # Partial llama-3.3-70b attempt (aborted: free-tier TPD exhausted) — do not reuse.
    "full_experiment_groq_v1_70b_partial": ("groq", DEFAULT_GROQ_MODEL),
    "full_experiment_groq_v1": ("groq", GROQ_VOLUME_MODEL),
    "bargaining_groq_v1": ("groq", GROQ_VOLUME_MODEL),
    "bargaining_risk_groq_v1": ("groq", GROQ_VOLUME_MODEL),
    # Haiku + English prompts (language isolation vs locked bargaining_v1 Turkish)
    "bargaining_haiku_en_v1": ("anthropic", settings.ANTHROPIC_MODEL),
    "bargaining_groq_v1_parallel_check": ("groq", GROQ_VOLUME_MODEL),
    "full_experiment_groq_v1_parallel_check": ("groq", GROQ_VOLUME_MODEL),
    "iterated_pd_groq_v1": ("groq", GROQ_VOLUME_MODEL),
    "stag_hunt_groq_v1": ("groq", GROQ_VOLUME_MODEL),
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str  # "anthropic" | "groq"
    model: str
    temperature: float


@dataclass
class ParsedResponse:
    """Structured parse result plus optional raw message for token accounting."""

    parsed: BaseModel
    raw: Any = None


def resolve_llm_config(
    experiment_id: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> LLMConfig:
    """Resolve provider/model for an experiment_id (registry overrides Settings)."""
    reg_provider: Optional[str] = None
    reg_model: Optional[str] = None
    if experiment_id and experiment_id in EXPERIMENT_LLM_REGISTRY:
        reg_provider, reg_model = EXPERIMENT_LLM_REGISTRY[experiment_id]

    resolved_provider = (
        provider
        or reg_provider
        or settings.LLM_PROVIDER
        or "anthropic"
    ).lower()
    if resolved_provider == "groq":
        resolved_model = (
            model or reg_model or settings.GROQ_MODEL or DEFAULT_GROQ_MODEL
        )
    else:
        resolved_model = (
            model or reg_model or settings.ANTHROPIC_MODEL
        )
    return LLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        temperature=float(
            temperature if temperature is not None else settings.TEMPERATURE
        ),
    )


def _require_anthropic_key() -> str:
    key = settings.ANTHROPIC_API_KEY
    if not key or key in _PLACEHOLDER_KEYS:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def _require_groq_key() -> str:
    key = settings.GROQ_API_KEY
    if not key or key in _PLACEHOLDER_KEYS:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
        )
    return key


def _build_structured_llm(schema: Type[T], cfg: LLMConfig):
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            api_key=_require_anthropic_key(),
        )
        return llm.with_structured_output(schema, include_raw=True)

    if cfg.provider == "groq":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            api_key=_require_groq_key(),
            base_url=GROQ_BASE_URL,
        )
        # llama-3.3-70b-versatile supports tool calling but not json_schema
        return llm.with_structured_output(
            schema, include_raw=True, method="function_calling"
        )

    raise ValueError(f"Unsupported LLM provider: {cfg.provider!r}")


def _extract_parsed(result: dict, schema: Type[T]) -> T:
    parsed = result.get("parsed")
    if parsed is not None:
        if isinstance(parsed, schema):
            return parsed
        return schema.model_validate(parsed)

    raw = result.get("raw")
    tool_calls = getattr(raw, "tool_calls", None) or []
    if tool_calls:
        args = dict(tool_calls[0].get("args") or {})
        return schema.model_validate(args)

    raise ValueError(
        f"{schema.__name__} parse failed: parsing_error={result.get('parsing_error')!r}"
    )


def _parse_failed_generation(err_body: object, schema: Type[T]) -> T:
    """Recover schema fields when Groq rejects tool calls that include extra keys."""
    import json
    import re

    text = str(err_body)
    # Prefer JSON object inside failed_generation
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in failed_generation: {text[:200]}")
    raw_obj = json.loads(m.group(0))
    if not isinstance(raw_obj, dict):
        raise ValueError("failed_generation JSON is not an object")
    allowed = set(schema.model_fields.keys())
    filtered = {k: v for k, v in raw_obj.items() if k in allowed}
    return schema.model_validate(filtered)


async def call_agent(
    messages: list[tuple[str, str]] | list[dict[str, str]],
    schema: Type[T],
    *,
    experiment_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ParsedResponse:
    """Invoke structured output and return a validated Pydantic model."""
    cfg = resolve_llm_config(
        experiment_id,
        provider=provider,
        model=model,
        temperature=temperature,
    )
    structured = _build_structured_llm(schema, cfg)
    try:
        result = await structured.ainvoke(messages)
    except Exception as exc:  # noqa: BLE001 — Groq tool_use_failed recovery
        msg = str(exc)
        if cfg.provider == "groq" and "failed_generation" in msg:
            parsed = _parse_failed_generation(msg, schema)
            return ParsedResponse(parsed=parsed, raw=None)
        raise
    if not isinstance(result, dict):
        # Some backends return the model directly
        parsed = result if isinstance(result, schema) else schema.model_validate(result)
        return ParsedResponse(parsed=parsed, raw=None)
    parsed = _extract_parsed(result, schema)
    return ParsedResponse(parsed=parsed, raw=result.get("raw"))
