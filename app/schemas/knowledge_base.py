from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeDocumentResponse(BaseModel):
    id: int
    title: str
    source_type: str
    source_url: str | None
    file_url: str | None
    status: str
    is_active: bool
    chunk_count: int
    created_by: int | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class KnowledgeTextCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1)
    source_url: str | None = None


class KnowledgeUrlCreateRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    title: str | None = Field(default=None, max_length=240)


class KnowledgeDocumentUpdateRequest(BaseModel):
    is_active: bool


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=6, ge=1, le=12)


class KnowledgeSearchHit(BaseModel):
    chunk_id: int
    chunk_index: int
    document_id: int
    document_title: str
    content: str
    citation: str
    score: float | None = None
    source_title: str | None = None
    source_url: str | None = None
    page_number: int | None = None


class KnowledgeSearchResponse(BaseModel):
    query: str
    count: int
    hits: list[KnowledgeSearchHit]
    metadata: dict[str, Any] = Field(default_factory=dict)
