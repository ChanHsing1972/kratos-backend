from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.services.auth import get_current_user
from app.services.upload import (
    ALLOWED_ATTACHMENT_TYPES,
    ALLOWED_AVATAR_TYPES,
    is_allowed_upload,
    save_upload_file,
)


router = APIRouter()


class UploadResponse(BaseModel):
    content_type: str
    filename: str
    size: int
    url: str


@router.post("/attachments", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_attachment(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if not is_allowed_upload(file, ALLOWED_ATTACHMENT_TYPES):
        raise HTTPException(status_code=400, detail="暂不支持此文件类型")
    try:
        return await save_upload_file(file, user_id=current_user.id, purpose="attachments")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/avatar", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_avatar(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not is_allowed_upload(file, ALLOWED_AVATAR_TYPES):
        raise HTTPException(status_code=400, detail="头像仅支持 JPG、PNG 或 WebP")
    try:
        uploaded = await save_upload_file(file, user_id=current_user.id, purpose="avatar")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    current_user.avatar_url = str(uploaded["url"])
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return uploaded
