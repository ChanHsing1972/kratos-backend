from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import UploadFile

import oss2

from app.core.config import BASE_DIR, settings


UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_AVATAR_BYTES = 5 * 1024 * 1024

ALLOWED_ATTACHMENT_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
    "text/csv",
    "text/plain",
    "application/json",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

ALLOWED_AVATAR_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


def is_allowed_upload(file: UploadFile, allowed_types: set[str]) -> bool:
    return (file.content_type or "").lower() in allowed_types


async def save_upload_file(file: UploadFile, *, user_id: int, purpose: str) -> dict[str, str | int]:
    content = await file.read()
    if not content:
        raise ValueError("文件不能为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("文件不能超过 12MB")

    extension = _safe_extension(file.filename or "")
    filename = f"{uuid4().hex}{extension}"
    relative_dir = Path(str(user_id)) / purpose
    target_dir = UPLOAD_DIR / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    target.write_bytes(content)

    original_name = _safe_display_name(file.filename or "upload")
    url = f"/uploads/{user_id}/{purpose}/{filename}"
    return {
        "content_type": file.content_type or "application/octet-stream",
        "filename": original_name,
        "size": len(content),
        "url": url,
    }


async def upload_avatar_to_oss(file: UploadFile, *, user_id: int) -> dict[str, str | int]:
    if not settings.ALIYUN_OSS_ACCESS_KEY_ID or not settings.ALIYUN_OSS_ACCESS_KEY_SECRET:
        raise ValueError("OSS 访问密钥未配置")

    content = await file.read()
    if not content:
        raise ValueError("头像不能为空")
    if len(content) > MAX_AVATAR_BYTES:
        raise ValueError("头像不能超过 5MB")

    extension = _safe_extension(file.filename or "")
    object_name = f"avatars/{user_id}/{uuid4().hex}{extension}"
    auth = oss2.Auth(
        settings.ALIYUN_OSS_ACCESS_KEY_ID,
        settings.ALIYUN_OSS_ACCESS_KEY_SECRET,
    )
    bucket = oss2.Bucket(auth, settings.ALIYUN_OSS_ENDPOINT, settings.ALIYUN_OSS_BUCKET)
    bucket.put_object(
        object_name,
        content,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "Content-Type": file.content_type or "application/octet-stream",
        },
    )

    return {
        "content_type": file.content_type or "application/octet-stream",
        "filename": _safe_display_name(file.filename or "avatar"),
        "size": len(content),
        "url": _public_oss_url(object_name),
    }


def _public_oss_url(object_name: str) -> str:
    if settings.ALIYUN_OSS_PUBLIC_BASE_URL:
        base_url = settings.ALIYUN_OSS_PUBLIC_BASE_URL.rstrip("/")
    else:
        endpoint = settings.ALIYUN_OSS_ENDPOINT
        parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
        base_url = f"{parsed.scheme}://{settings.ALIYUN_OSS_BUCKET}.{parsed.netloc}"
    return f"{base_url}/{object_name}"


def _safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) else ""


def _safe_display_name(filename: str) -> str:
    compact = re.sub(r"[\r\n\t/\\]+", " ", filename).strip()
    return compact[:120] or "upload"
