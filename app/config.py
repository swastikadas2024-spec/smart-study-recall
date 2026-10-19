from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Smart Study Recall API"
    app_version: str = "1.1.0"

    database_url: str = ""
    ml_service_url: str = "http://127.0.0.1:8000"

    supabase_url: str = ""
    supabase_anon_key: str = ""

    # Comma-separated list of origins allowed to call this API from a browser.
    frontend_origins: str = "http://127.0.0.1:5500,http://localhost:5500"

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.frontend_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
