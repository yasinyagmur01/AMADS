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

    # Anthropic model (Bölüm 10)
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"

    # LangSmith Tracing
    LANGSMITH_TRACING: bool = True
    LANGSMITH_PROJECT: str = "amads-mock-dev"

    # Simulation Constraints (Hard Caps)
    AGENT_COUNT: int = 5
    MAX_ROUNDS: int = 15
    # Round başına çekilebilir üst sınır: pool_after * EXTRACTION_LIMIT_RATIO
    # Gerçek LLM testleriyle kalibre edildi (3 run, sabit trait kompozisyonu,
    # hepsi round 12'de collapse). Not: bu sabit trait kompozisyonuyla varyans
    # GÖRÜLMEDİ — bu beklenen bir durum, çünkü temperature=0.2 + aynı trait'ler
    # + aynı senaryo doğası gereği tutarlı sonuç üretir. Gerçek varyans, asıl
    # deneyde run'lar arası trait kompozisyonu DEĞİŞTİRİLEREK elde edilecek
    # (Bölüm 12, istatistiksel deney tasarımı).
    EXTRACTION_LIMIT_RATIO: float = 0.12
    # Havuz çöküş eşiği: pool_capacity * COLLAPSE_EPSILON_RATIO
    COLLAPSE_EPSILON_RATIO: float = 0.01

    # Defaults
    TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
