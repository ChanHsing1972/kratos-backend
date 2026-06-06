from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.user import User
from app.schemas.health_metric import (
    HealthMetricCreate,
    HealthMetricResponse,
    HealthMetricUpdate,
)
from app.services.auth import get_current_user
from app.services.health_metric import (
    create_health_metric,
    delete_health_metric,
    get_health_metric_by_id,
    get_health_metrics_by_user_id,
    update_health_metric,
)

router = APIRouter()


@router.get("", response_model=list[HealthMetricResponse])
def list_health_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_health_metrics_by_user_id(db, current_user.id)


@router.get("/{metric_id}", response_model=HealthMetricResponse)
def get_health_metric(
    metric_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metric = get_health_metric_by_id(db, metric_id, current_user.id)
    if not metric:
        raise HTTPException(status_code=404, detail="健康数据记录不存在")
    return metric


@router.post("", response_model=HealthMetricResponse, status_code=status.HTTP_201_CREATED)
def create_metric(
    metric_in: HealthMetricCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return create_health_metric(db, current_user, metric_in)


@router.put("/{metric_id}", response_model=HealthMetricResponse)
def update_metric(
    metric_id: int,
    metric_in: HealthMetricUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metric = get_health_metric_by_id(db, metric_id, current_user.id)
    if not metric:
        raise HTTPException(status_code=404, detail="健康数据记录不存在")
    return update_health_metric(db, metric, metric_in)


@router.patch("/{metric_id}", response_model=HealthMetricResponse)
def patch_metric(
    metric_id: int,
    metric_in: HealthMetricUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metric = get_health_metric_by_id(db, metric_id, current_user.id)
    if not metric:
        raise HTTPException(status_code=404, detail="健康数据记录不存在")
    return update_health_metric(db, metric, metric_in)


@router.delete("/{metric_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_metric(
    metric_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metric = get_health_metric_by_id(db, metric_id, current_user.id)
    if not metric:
        raise HTTPException(status_code=404, detail="健康数据记录不存在")
    delete_health_metric(db, metric)
