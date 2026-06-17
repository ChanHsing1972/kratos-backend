from __future__ import annotations

import math
import re
from typing import Any

import httpx
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session, selectinload

from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.services.embedding import embed_text, embedding_model_label


CHUNK_MIN_CHARS = 700
CHUNK_MAX_CHARS = 1800


def create_document_from_text(
    db: Session,
    *,
    title: str,
    content: str,
    created_by: int | None,
    source_type: str = "manual",
    source_url: str | None = None,
    file_url: str | None = None,
) -> KnowledgeDocument:
    document = KnowledgeDocument(
        title=title.strip()[:240] or "未命名知识文档",
        source_type=source_type,
        source_url=source_url,
        file_url=file_url,
        status="processing",
        created_by=created_by,
    )
    db.add(document)
    db.flush()
    chunks = chunk_text(content)
    _replace_document_chunks(db, document, chunks)
    document.status = "ready"
    document.chunk_count = len(chunks)
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


async def create_document_from_upload(
    db: Session,
    *,
    file,
    created_by: int | None,
) -> KnowledgeDocument:
    raw = await file.read()
    content_type = (file.content_type or "").lower()
    filename = file.filename or "uploaded-document"
    if filename.lower().endswith(".pdf") or "pdf" in content_type:
        content = extract_pdf_text(raw)
        source_type = "pdf"
    else:
        content = raw.decode("utf-8", errors="ignore")
        source_type = "markdown" if filename.lower().endswith((".md", ".markdown")) else "txt"
    return create_document_from_text(
        db,
        title=filename,
        content=content,
        created_by=created_by,
        source_type=source_type,
    )


def create_document_from_url(
    db: Session,
    *,
    url: str,
    title: str | None,
    created_by: int | None,
) -> KnowledgeDocument:
    response = httpx.get(url, timeout=20, follow_redirects=True)
    response.raise_for_status()
    content = html_to_text(response.text) if "html" in response.headers.get("content-type", "") else response.text
    return create_document_from_text(
        db,
        title=title or url,
        content=content,
        created_by=created_by,
        source_type="url",
        source_url=url,
    )


def list_documents(db: Session) -> list[KnowledgeDocument]:
    return list(
        db.scalars(
            select(KnowledgeDocument)
            .options(selectinload(KnowledgeDocument.chunks))
            .order_by(KnowledgeDocument.created_at.desc(), KnowledgeDocument.id.desc())
        ).all()
    )


def set_document_active(db: Session, document_id: int, is_active: bool) -> KnowledgeDocument | None:
    document = db.get(KnowledgeDocument, document_id)
    if document is None:
        return None
    document.is_active = is_active
    for chunk in document.chunks:
        chunk.is_active = is_active
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def retrieve_rag_contexts(
    db: Session,
    query: str,
    *,
    limit: int = 6,
    candidate_limit: int = 20,
    min_score: float = 0.05,
) -> list[dict[str, Any]]:
    query_text = str(query or "").strip()
    if not query_text:
        return []

    query_embedding = embed_text(query_text)
    contexts = _retrieve_pgvector(db, query_embedding, candidate_limit)
    if not contexts:
        contexts = _retrieve_python_similarity(db, query_embedding, candidate_limit)

    contexts = _apply_retrieval_rules(contexts, limit=limit, min_score=min_score)
    for index, item in enumerate(contexts, start=1):
        item["citation"] = _chunk_citation(item, index)
    return contexts


def format_rag_contexts(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "未检索到外部知识库上下文。"
    lines = []
    for item in contexts:
        page = f" p.{item['page_number']}" if item.get("page_number") else ""
        source = []
        if item.get("source_url"):
            source.append(f"URL: {item['source_url']}")
        if item.get("score") is not None:
            source.append(f"score: {float(item['score']):.3f}")
        source_text = f" ({'；'.join(source)})" if source else ""
        lines.append(
            f"- {item['citation']} {item['document_title']}{page} #chunk-{item['chunk_id']}{source_text}\n"
            f"  {item['content']}"
        )
    return "\n".join(lines)


def chunk_text(content: str) -> list[dict[str, Any]]:
    text_value = normalize_text(content)
    if not text_value:
        return []

    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text_value) if item.strip()]
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > CHUNK_MAX_CHARS:
            chunks.append(_build_chunk_payload("\n\n".join(current), len(chunks)))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)
        if current_len >= CHUNK_MIN_CHARS:
            chunks.append(_build_chunk_payload("\n\n".join(current), len(chunks)))
            current = []
            current_len = 0
    if current:
        chunks.append(_build_chunk_payload("\n\n".join(current), len(chunks)))
    return chunks


def normalize_text(content: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", str(content or ""))).strip()


def extract_pdf_text(raw: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text_value = page.extract_text() or ""
        if text_value.strip():
            pages.append(f"[page {index}]\n{text_value}")
    return "\n\n".join(pages)


def html_to_text(html: str) -> str:
    without_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
    text_value = re.sub(r"<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", text_value).strip()


def _replace_document_chunks(db: Session, document: KnowledgeDocument, chunks: list[dict[str, Any]]) -> None:
    db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))
    for chunk in chunks:
        db.add(
            KnowledgeChunk(
                document_id=document.id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                embedding=embed_text(chunk["content"]),
                metadata_json={"embedding_model": embedding_model_label()},
                token_count=chunk["token_count"],
                source_title=document.title,
                source_url=document.source_url,
                page_number=chunk.get("page_number"),
                is_active=document.is_active,
            )
        )


def _build_chunk_payload(content: str, index: int) -> dict[str, Any]:
    page_match = re.search(r"\[page\s+(\d+)\]", content, flags=re.I)
    return {
        "chunk_index": index,
        "content": content.strip(),
        "page_number": int(page_match.group(1)) if page_match else None,
        "token_count": max(1, len(content) // 2),
    }


def _retrieve_pgvector(db: Session, query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
    if db.bind is None or db.bind.dialect.name != "postgresql":
        return []
    vector_literal = "[" + ",".join(f"{float(item):.8f}" for item in query_embedding) + "]"
    try:
        rows = db.execute(
            text(
                """
                SELECT
                    c.id AS chunk_id,
                    c.chunk_index,
                    c.content,
                    c.source_title,
                    c.source_url,
                    c.page_number,
                    d.id AS document_id,
                    d.title AS document_title,
                    1 - (c.embedding <=> CAST(:query_embedding AS vector)) AS score
                FROM knowledge_chunks c
                JOIN knowledge_documents d ON d.id = c.document_id
                WHERE c.is_active = TRUE
                  AND d.is_active = TRUE
                  AND d.status = 'ready'
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :limit
                """
            ),
            {"query_embedding": vector_literal, "limit": limit},
        ).mappings()
        return [dict(row) for row in rows]
    except Exception:
        db.rollback()
        return []


def _retrieve_python_similarity(db: Session, query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        select(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
        .where(KnowledgeChunk.is_active.is_(True))
        .where(KnowledgeDocument.is_active.is_(True))
        .where(KnowledgeDocument.status == "ready")
        .order_by(KnowledgeDocument.updated_at.desc(), KnowledgeChunk.id.desc())
    ).all()
    scored = []
    for chunk, document in rows:
        if not chunk.embedding:
            continue
        score = cosine_similarity(query_embedding, list(chunk.embedding))
        scored.append(
            {
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "source_title": chunk.source_title,
                "source_url": chunk.source_url,
                "page_number": chunk.page_number,
                "document_id": document.id,
                "document_title": document.title,
                "score": score,
            }
        )
    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    return scored[:limit]


def _apply_retrieval_rules(contexts: list[dict[str, Any]], *, limit: int, min_score: float) -> list[dict[str, Any]]:
    selected = []
    per_document: dict[int, int] = {}
    for item in contexts:
        score = float(item.get("score") or 0)
        document_id = int(item["document_id"])
        if score < min_score:
            continue
        if per_document.get(document_id, 0) >= 3:
            continue
        selected.append(item)
        per_document[document_id] = per_document.get(document_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _chunk_citation(item: dict[str, Any], index: int) -> str:
    page = f" p.{item['page_number']}" if item.get("page_number") else ""
    return f"[{item['document_title']}{page} #chunk-{item['chunk_id']}]"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)
