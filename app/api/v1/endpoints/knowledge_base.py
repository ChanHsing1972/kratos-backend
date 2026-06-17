from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.knowledge_base import (
    KnowledgeDocumentResponse,
    KnowledgeDocumentUpdateRequest,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeTextCreateRequest,
    KnowledgeUrlCreateRequest,
)
from app.services.auth import get_current_user
from app.services.embedding import embedding_model_label
from app.services.rag import (
    create_document_from_text,
    create_document_from_upload,
    create_document_from_url,
    list_documents,
    retrieve_rag_contexts,
    set_document_active,
)

router = APIRouter()


@router.get("/documents", response_model=list[KnowledgeDocumentResponse])
def list_knowledge_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_documents(db)


@router.post("/documents/text", response_model=KnowledgeDocumentResponse, status_code=status.HTTP_201_CREATED)
def create_knowledge_document_from_text(
    payload: KnowledgeTextCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return create_document_from_text(
        db,
        title=payload.title,
        content=payload.content,
        source_url=payload.source_url,
        source_type="manual",
        created_by=current_user.id,
    )


@router.post("/documents/url", response_model=KnowledgeDocumentResponse, status_code=status.HTTP_201_CREATED)
def create_knowledge_document_from_url(
    payload: KnowledgeUrlCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return create_document_from_url(
            db,
            url=payload.url,
            title=payload.title,
            created_by=current_user.id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"网页导入失败：{exc}") from exc


@router.post("/documents/upload", response_model=KnowledgeDocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_knowledge_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    filename = (file.filename or "").lower()
    if not filename.endswith((".pdf", ".txt", ".md", ".markdown")):
        raise HTTPException(status_code=400, detail="知识库文档仅支持 PDF、Markdown、TXT")
    try:
        return await create_document_from_upload(db, file=file, created_by=current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"文档解析失败：{exc}") from exc


@router.patch("/documents/{document_id}", response_model=KnowledgeDocumentResponse)
def update_knowledge_document(
    document_id: int,
    payload: KnowledgeDocumentUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = set_document_active(db, document_id, payload.is_active)
    if document is None:
        raise HTTPException(status_code=404, detail="知识库文档不存在")
    return document


@router.post("/search", response_model=KnowledgeSearchResponse)
def search_knowledge_base(
    payload: KnowledgeSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    hits = retrieve_rag_contexts(db, payload.query, limit=payload.limit)
    return KnowledgeSearchResponse(
        query=payload.query,
        count=len(hits),
        hits=hits,
        metadata={"embedding_model": embedding_model_label()},
    )
