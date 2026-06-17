from typing import Any

from sqlalchemy.orm import Session

from app.services.rag import format_rag_contexts, retrieve_rag_contexts


def retrieve_knowledge_contexts(
    db: Session,
    query: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    query_text = str(query or "").strip()
    if not query_text:
        return []

    return retrieve_rag_contexts(db, query_text, limit=limit)


def format_knowledge_contexts(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "未检索到外部知识库上下文。"
    return format_rag_contexts(contexts)
