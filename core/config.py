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
    GROQ_API_KEY: Optional[str] = None

    # LangSmith Tracing
    LANGSMITH_TRACING: bool = True
    LANGSMITH_PROJECT: str = "amads-mock-dev"

    # Simulation Constraints (Hard Caps)
    AGENT_COUNT: int = 5
    MAX_ROUNDS: int = 15

    # Defaults
    TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
