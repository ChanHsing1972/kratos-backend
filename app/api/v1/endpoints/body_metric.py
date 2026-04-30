from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.body_metric import (
    BodyMetricCreate,
    BodyMetricResponse,
    BodyMetricUpdate,
)
from app.services.auth import get_current_user
from app.services.body_metric import (
    create_body_metric,
    delete_body_metric,
    get_body_metric_by_id,
    get_body_metrics_by_user_id,
    update_body_metric,
)

router = APIRouter()


@router.get("", response_model=list[BodyMetricResponse])
def list_body_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_body_metrics_by_user_id(db, current_user.id)


@router.get("/{metric_id}", response_model=BodyMetricResponse)
def get_body_metric(
    metric_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metric = get_body_metric_by_id(db, metric_id, current_user.id)
    if not metric:
        raise HTTPException(status_code=404, detail="身体指标记录不存在")
    return metric


@router.post("", response_model=BodyMetricResponse, status_code=status.HTTP_201_CREATED)
def create_metric(
    metric_in: BodyMetricCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return create_body_metric(db, current_user, metric_in)


@router.put("/{metric_id}", response_model=BodyMetricResponse)
def update_metric(
    metric_id: int,
    metric_in: BodyMetricUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metric = get_body_metric_by_id(db, metric_id, current_user.id)
    if not metric:
        raise HTTPException(status_code=404, detail="身体指标记录不存在")
    return update_body_metric(db, metric, metric_in)


@router.patch("/{metric_id}", response_model=BodyMetricResponse)
def patch_metric(
    metric_id: int,
    metric_in: BodyMetricUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metric = get_body_metric_by_id(db, metric_id, current_user.id)
    if not metric:
        raise HTTPException(status_code=404, detail="身体指标记录不存在")
    return update_body_metric(db, metric, metric_in)


@router.delete("/{metric_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_metric(
    metric_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metric = get_body_metric_by_id(db, metric_id, current_user.id)
    if not metric:
        raise HTTPException(status_code=404, detail="身体指标记录不存在")
    delete_body_metric(db, metric)
