"""Cloudflare R2(S3-compatible) 업로드 유틸리티.

R2_ACCESS_KEY_ID와 R2_SECRET_ACCESS_KEY는 절대 저장소에 커밋하지 않는다.
터미널 환경변수 또는 비밀 관리 서비스에서 주입한다.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any


def _load_local_secrets() -> dict[str, str]:
    """커밋되지 않는 .streamlit/secrets.toml을 로컬 업로드에도 사용할 수 있게 한다."""

    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        return {}
    try:
        import tomllib

        with secrets_path.open("rb") as file:
            values = tomllib.load(file)
        return {str(key): str(value) for key, value in values.items()}
    except (OSError, ImportError, ValueError):
        return {}


def _required_value(name: str, secrets: dict[str, str]) -> str:
    value = os.getenv(name, "").strip() or secrets.get(name, "").strip()
    if not value:
        raise RuntimeError(f"환경변수 또는 secrets.toml의 {name}이(가) 설정되지 않았습니다.")
    return value


def get_r2_settings() -> dict[str, str]:
    """환경변수를 우선하고 로컬 secrets.toml을 보조 설정으로 읽는다."""

    secrets = _load_local_secrets()
    return {
        "account_id": _required_value("R2_ACCOUNT_ID", secrets),
        "access_key_id": _required_value("R2_ACCESS_KEY_ID", secrets),
        "secret_access_key": _required_value("R2_SECRET_ACCESS_KEY", secrets),
        "bucket_name": _required_value("R2_BUCKET_NAME", secrets),
        "public_base_url": _required_value("R2_PUBLIC_BASE_URL", secrets).rstrip("/"),
    }


def create_r2_client(settings: dict[str, str] | None = None) -> Any:
    """Cloudflare R2 S3-compatible client를 생성한다."""

    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("boto3가 없습니다. requirements.txt를 설치해 주세요.") from error

    settings = settings or get_r2_settings()
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=settings["access_key_id"],
        aws_secret_access_key=settings["secret_access_key"],
        region_name="auto",
    )


def upload_file_to_r2(
    file_path: str | Path,
    object_key: str,
    client: Any,
    settings: dict[str, str],
) -> str:
    """파일을 R2에 업로드하고 공개 URL을 반환한다."""

    source_path = Path(file_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    content_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    client.upload_file(
        str(source_path),
        settings["bucket_name"],
        object_key,
        ExtraArgs={
            "ContentType": content_type,
            "CacheControl": "public, max-age=31536000, immutable",
        },
    )
    return f"{settings['public_base_url']}/{object_key.lstrip('/')}"
