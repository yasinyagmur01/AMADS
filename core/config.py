from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(_ENV_FILE, override=True)


class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None

    # Anthropic model (default for locked Haiku experiments)
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"

    # LangSmith Tracing
    LANGSMITH_TRACING: bool = True
    LANGSMITH_PROJECT: str = "amads-mock-dev"

    # Simulation Constraints (Hard Caps)
    AGENT_COUNT: int = 5
    MAX_ROUNDS: int = 15
    # Per-round extraction cap: pool_after * EXTRACTION_LIMIT_RATIO.
    # Calibrated with real LLM tests (3 runs, fixed trait composition,
    # all collapsed at round 12). Low within-condition variance at
    # temperature=0.2 with identical traits is expected; experimental
    # variance comes from changing traits across conditions (Section 12).
    EXTRACTION_LIMIT_RATIO: float = 0.12
    # Pool collapse threshold: pool_capacity * COLLAPSE_EPSILON_RATIO
    COLLAPSE_EPSILON_RATIO: float = 0.01

    # Defaults
    TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
