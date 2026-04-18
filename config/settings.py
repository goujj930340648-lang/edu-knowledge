"""
服务层配置：MongoDB、MinIO、API。

环境变量优先读 ``MONGODB_URI``；兼容 ``.env.example`` 中的 ``MONGO_URL``。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mongodb_uri: str = Field(default="", validation_alias="MONGODB_URI")
    mongo_url: str = Field(default="", validation_alias="MONGO_URL")
    mongodb_database: str = Field(default="edu_knowledge", validation_alias="MONGODB_DATABASE")

    minio_endpoint: str = Field(default="", validation_alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="", validation_alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="", validation_alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="education-knowledge", validation_alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, validation_alias="MINIO_SECURE")
    minio_public_base_url: str = Field(
        default="",
        validation_alias="MINIO_PUBLIC_BASE_URL",
        description="可选，如 http://127.0.0.1:9000/education-knowledge 用于拼接图片外链",
    )

    api_keys: str = Field(
        default="",
        validation_alias="API_KEYS",
        description="逗号分隔；非空时要求请求头 X-API-Key 命中其一",
    )
    intent_use_llm: bool = Field(default=False, validation_alias="INTENT_USE_LLM")

    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")

    def resolved_mongo_uri(self) -> str:
        u = (self.mongodb_uri or "").strip()
        if u:
            return u
        return (self.mongo_url or "").strip()

    def mongo_configured(self) -> bool:
        return bool(self.resolved_mongo_uri())

    def minio_configured(self) -> bool:
        return bool((self.minio_endpoint or "").strip() and (self.minio_access_key or "").strip())

    def api_key_list(self) -> list[str]:
        return [x.strip() for x in (self.api_keys or "").split(",") if x.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
