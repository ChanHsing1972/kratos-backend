from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.diet import (
    DietRecordBulkCreate,
    DietRecordResponse,
    FoodImageEstimateResponse,
)
from app.services.auth import get_current_user
from app.services.diet import create_diet_records, get_recent_diet_records
from app.services.diet_image_estimator import DietImageEstimatorError, estimate_food_from_image


router = APIRouter()

ALLOWED_FOOD_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FOOD_IMAGE_BYTES = 8 * 1024 * 1024


@router.get("/records", response_model=list[DietRecordResponse])
def list_records(
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_recent_diet_records(db, current_user.id, limit=limit)


@router.post("/estimate-from-image", response_model=FoodImageEstimateResponse)
async def estimate_from_image(
    image: UploadFile | None = File(default=None),
    current_user: User = Depends(get_current_user),
):
    if image is None:
        raise HTTPException(status_code=400, detail="请上传图片")

    content_type = (image.content_type or "").lower()
    if content_type not in ALLOWED_FOOD_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG 或 WebP 图片")

    content = await image.read(MAX_FOOD_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="图片不能为空")
    if len(content) > MAX_FOOD_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片不能超过 8MB")

    try:
        result = estimate_food_from_image(image_bytes=content, mime_type=content_type)
    except DietImageEstimatorError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="饮食图片识别服务暂时不可用，请稍后重试",
        ) from exc

    return FoodImageEstimateResponse(data=result)


@router.post(
    "/records",
    response_model=list[DietRecordResponse],
    status_code=status.HTTP_201_CREATED,
)
def create_records(
    payload: DietRecordBulkCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return create_diet_records(db, current_user, payload)
