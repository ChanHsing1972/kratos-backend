from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import BASE_DIR


UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

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


def _safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) else ""


def _safe_display_name(filename: str) -> str:
    compact = re.sub(r"[\r\n\t/\\]+", " ", filename).strip()
    return compact[:120] or "upload"
