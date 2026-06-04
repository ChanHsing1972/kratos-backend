import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.knowledge_base import KnowledgeBaseEntry


def retrieve_knowledge_contexts(
    db: Session,
    query: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    query_text = str(query or "").strip()
    if not query_text:
        return []

    entries = db.scalars(
        select(KnowledgeBaseEntry)
        .where(KnowledgeBaseEntry.is_active.is_(True))
        .order_by(KnowledgeBaseEntry.updated_at.desc(), KnowledgeBaseEntry.id.desc())
    ).all()

    scored: list[tuple[float, KnowledgeBaseEntry]] = []
    for entry in entries:
        score = _match_score(query_text, entry)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda item: (-item[0], item[1].id or 0))

    contexts: list[dict[str, Any]] = []
    for index, (_, entry) in enumerate(scored[: max(0, limit)], start=1):
        contexts.append(
            {
                "id": entry.id,
                "title": entry.title,
                "content": entry.content,
                "source": entry.source,
                "source_title": entry.source_title,
                "source_url": entry.source_url,
                "tags": _normalize_tags(entry.tags),
                "citation": f"[知识库:{entry.title}#{index}]",
            }
        )
    return contexts


def format_knowledge_contexts(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "未检索到外部知识库上下文。"
    lines = []
    for item in contexts:
        source_parts = []
        if item.get("source_title"):
            source_parts.append(f"文章/网页: {item['source_title']}")
        if item.get("source_url"):
            source_parts.append(f"URL: {item['source_url']}")
        if item.get("source"):
            source_parts.append(f"来源说明: {item['source']}")
        source = f" {'；'.join(source_parts)}" if source_parts else ""
        tags = "、".join(item.get("tags") or [])
        tag_text = f" 标签: {tags}" if tags else ""
        lines.append(f"- {item['citation']} {item['title']}:{source}{tag_text}\n  {item['content']}")
    return "\n".join(lines)


def _match_score(query: str, entry: KnowledgeBaseEntry) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0

    title_tokens = set(_tokenize(entry.title))
    content_tokens = set(_tokenize(entry.content))
    tag_tokens = {
        token
        for tag in _normalize_tags(entry.tags)
        for token in _tokenize(tag)
    }

    score = 0.0
    score += len(query_tokens & title_tokens) * 3.0
    score += len(query_tokens & tag_tokens) * 2.5
    score += len(query_tokens & content_tokens) * 1.0

    compact_query = re.sub(r"\s+", "", query.lower())
    for tag in _normalize_tags(entry.tags):
        compact_tag = re.sub(r"\s+", "", tag.lower())
        if compact_tag and compact_tag in compact_query:
            score += 2.0

    return score


def _tokenize(text: str) -> list[str]:
    normalized = str(text or "").lower()
    words = re.findall(r"[a-z0-9]+", normalized)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    chinese_bigrams = [
        "".join(pair)
        for pair in zip(chinese_chars, chinese_chars[1:])
    ]
    return words + chinese_chars + chinese_bigrams


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
