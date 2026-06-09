from __future__ import annotations

import base64
from io import BytesIO
import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4
from xml.etree import ElementTree

from fastapi import UploadFile

try:
    import oss2
except Exception:  # pragma: no cover - optional runtime dependency fallback
    oss2 = None

try:
    from PIL import Image
    from PIL import ImageOps
except Exception:  # pragma: no cover - optional runtime dependency fallback
    Image = None
    ImageOps = None

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

TEXT_ATTACHMENT_TYPES = {
    "application/json",
    "text/csv",
    "text/plain",
}

DOCX_ATTACHMENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_ATTACHMENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PDF_ATTACHMENT_TYPE = "application/pdf"
ATTACHMENT_TEXT_PREVIEW_CHARS = 4_000
AGENT_IMAGE_MAX_EDGE_PX = 768
AGENT_IMAGE_JPEG_QUALITY = 64
AGENT_IMAGE_TARGET_BYTES = 220 * 1024


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
    if oss2 is None:
        raise ValueError("OSS SDK 未安装，请安装 oss2 后重试")

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


def build_agent_attachment_parts(
    *,
    user_id: int,
    message: str,
    attachments: list[dict],
) -> list[dict[str, object]] | str:
    """Build OpenAI-compatible multimodal content parts for the chat model.

    Images are sent as data URLs so the model sees pixels, while documents are
    sent both as file parts and as extracted text previews when we can parse
    them locally. The text preview keeps the experience useful even when a
    provider only partially supports generic file parts.
    """
    text = message.strip()
    if not attachments:
        return text

    parts: list[dict[str, object]] = []
    intro_lines = [text] if text else []
    intro_lines.append("用户上传了以下附件，请结合附件内容回答：")
    for index, attachment in enumerate(attachments, start=1):
        intro_lines.append(
            f"{index}. {attachment.get('filename') or '附件'} "
            f"({attachment.get('content_type') or 'application/octet-stream'})"
        )
    parts.append({"type": "text", "text": "\n".join(intro_lines)})

    for attachment in attachments[:8]:
        filename = str(attachment.get("filename") or "attachment")
        content_type = str(attachment.get("content_type") or "application/octet-stream")
        inline_data_url = str(attachment.get("data_url") or "")

        url = str(attachment.get("url") or "")
        file_path = resolve_upload_url_for_user(user_id=user_id, url=url)
        if content_type.startswith("image/"):
            image_data_url = None
            if file_path is not None and file_path.is_file():
                image_data_url = image_data_url_for_agent(
                    file_path.read_bytes(),
                    content_type,
                )
            if image_data_url is None and inline_data_url.startswith("data:image/"):
                image_data_url = image_data_url_from_inline(inline_data_url)
            if image_data_url is not None:
                parts.append({"type": "image_url", "image_url": {"url": image_data_url}})
            else:
                parts.append(
                    {
                        "type": "text",
                        "text": f"图片附件 {filename} 暂时无法压缩读取，请根据文件名和用户描述回答。",
                    }
                )
            continue

        if file_path is None or not file_path.is_file():
            parts.append(
                {
                    "type": "text",
                    "text": f"附件 {filename} 无法在服务器读取，可能已被移动或不属于当前用户。",
                }
            )
            continue

        content = file_path.read_bytes()
        data_url = _data_url(content_type, content)
        parts.append(
            {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": data_url,
                },
            }
        )
        preview = extract_attachment_text_preview(
            file_path=file_path,
            content_type=content_type,
            content=content,
        )
        if preview:
            parts.append(
                {
                    "type": "text",
                    "text": f"附件 {filename} 的文本预览：\n{preview}",
                }
            )

    return parts


def resolve_upload_url_for_user(*, user_id: int, url: str) -> Path | None:
    parsed = urlparse(url)
    path = parsed.path if parsed.scheme else url
    parts = [part for part in Path(path).parts if part not in {"/", ""}]
    expected_prefix = ["uploads", str(user_id)]
    if parts[:2] != expected_prefix:
        return None

    target = (UPLOAD_DIR / Path(*parts[1:])).resolve()
    upload_root = UPLOAD_DIR.resolve()
    try:
        target.relative_to(upload_root)
    except ValueError:
        return None
    return target


def extract_attachment_text_preview(
    *,
    file_path: Path,
    content_type: str,
    content: bytes,
) -> str | None:
    if content_type in TEXT_ATTACHMENT_TYPES:
        return _clip_preview(_decode_text(content))
    if content_type == DOCX_ATTACHMENT_TYPE:
        return _clip_preview(_extract_docx_text(file_path))
    if content_type == XLSX_ATTACHMENT_TYPE:
        return _clip_preview(_extract_xlsx_text(file_path))
    if content_type == PDF_ATTACHMENT_TYPE:
        return _clip_preview(_extract_pdf_text(file_path))
    return None


def _data_url(content_type: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def image_data_url_for_agent(content: bytes, content_type: str = "image/jpeg") -> str | None:
    compressed = _compress_image_for_agent(content)
    if compressed is None:
        return (
            _data_url(content_type, content)
            if len(content) <= AGENT_IMAGE_TARGET_BYTES
            else None
        )
    return _data_url("image/jpeg", compressed)


def image_data_url_from_inline(data_url: str) -> str | None:
    match = re.fullmatch(r"data:(image/[^;]+);base64,(.+)", data_url, flags=re.DOTALL)
    if not match:
        return None
    try:
        content = base64.b64decode(match.group(2), validate=True)
    except ValueError:
        return None
    return image_data_url_for_agent(content, match.group(1))


def _compress_image_for_agent(content: bytes) -> bytes | None:
    if Image is None or ImageOps is None:
        return None

    try:
        with Image.open(BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((AGENT_IMAGE_MAX_EDGE_PX, AGENT_IMAGE_MAX_EDGE_PX))
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, "white")
                alpha = image.getchannel("A")
                background.paste(image, mask=alpha)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            for quality in (AGENT_IMAGE_JPEG_QUALITY, 64, 56, 48):
                output = BytesIO()
                image.save(output, format="JPEG", quality=quality, optimize=True)
                value = output.getvalue()
                if len(value) <= AGENT_IMAGE_TARGET_BYTES or quality == 48:
                    return value
    except Exception:
        return None
    return None


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _extract_docx_text(file_path: Path) -> str | None:
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_text = docx.read("word/document.xml")
    except (KeyError, OSError, zipfile.BadZipFile):
        return None

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return None
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [
            node.text or ""
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs) or None


def _extract_xlsx_text(file_path: Path) -> str | None:
    try:
        with zipfile.ZipFile(file_path) as workbook:
            shared_strings = _read_xlsx_shared_strings(workbook)
            sheet_names = [
                name
                for name in workbook.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            ]
            rows: list[str] = []
            for sheet_name in sheet_names[:5]:
                root = ElementTree.fromstring(workbook.read(sheet_name))
                for row in root.findall(".//{*}row")[:80]:
                    values = [
                        _xlsx_cell_text(cell, shared_strings)
                        for cell in row.findall("{*}c")
                    ]
                    cleaned = [value for value in values if value]
                    if cleaned:
                        rows.append(", ".join(cleaned))
            return "\n".join(rows) or None
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError):
        return None


def _read_xlsx_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    except (KeyError, ElementTree.ParseError):
        return []
    values: list[str] = []
    for item in root.findall(".//{*}si"):
        text = "".join(node.text or "" for node in item.findall(".//{*}t"))
        values.append(text)
    return values


def _xlsx_cell_text(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    value_node = cell.find("{*}v")
    if value_node is None or value_node.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings[int(value_node.text)]
        except (IndexError, ValueError):
            return ""
    return value_node.text


def _extract_pdf_text(file_path: Path) -> str | None:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages[:12]]
    except Exception:
        return None
    return "\n".join(page.strip() for page in pages if page.strip()) or None


def _clip_preview(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) <= ATTACHMENT_TEXT_PREVIEW_CHARS:
        return text
    return f"{text[:ATTACHMENT_TEXT_PREVIEW_CHARS]}\n...[内容已截断]"
