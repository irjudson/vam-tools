"""Database configuration."""

import os
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # PostgreSQL connection
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "pg"
    postgres_password: str = "buffalo-jump"
    postgres_db: str = "lumina"

    # Redis connection
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 2
    redis_password: str = "buffalo-jump"

    # SQLAlchemy settings
    sql_echo: bool = False  # Set to True to log all SQL queries

    # Data directory for caches and indices
    data_dir: str = "/var/lib/lumina"

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Construct Redis URL for Celery."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def faiss_index_dir(self) -> Path:
        """Get directory for FAISS index files."""
        return Path(self.data_dir) / "faiss_indices"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env
    )


# Global settings instance
settings = Settings()
