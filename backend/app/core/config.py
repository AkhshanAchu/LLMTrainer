from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    app_name: str = "FineTune Studio"
    api_prefix: str = "/api"

    data_dir: Path = DATA_DIR
    db_path: Path = DATA_DIR / "db" / "finetune_studio.sqlite3"
    uploads_dir: Path = DATA_DIR / "uploads"
    checkpoints_dir: Path = DATA_DIR / "checkpoints"
    models_cache_dir: Path = DATA_DIR / "models_cache"
    exports_dir: Path = DATA_DIR / "exports"

    cors_origins: list[str] = ["http://localhost:3000"]

    hf_token: str | None = None

    class Config:
        env_prefix = "FTS_"
        env_file = ".env"

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for d in (
        settings.data_dir,
        settings.db_path.parent,
        settings.uploads_dir,
        settings.checkpoints_dir,
        settings.models_cache_dir,
        settings.exports_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    return settings
