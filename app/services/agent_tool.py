from sqlalchemy.orm import Session

from app.agent.tool_registry import TOOL_REGISTRY, ToolMetadata
from app.core.config import settings
from app.models.agent_tool import AgentToolConfig
from app.schemas.agent_tool import (
    AgentToolBulkUpdate,
    AgentToolConfigUpdate,
    AgentToolHealthUpdate,
)


def ensure_user_tool_configs(db: Session, user_id: int) -> None:
    existing = {
        item.name: item
        for item in db.query(AgentToolConfig).filter(AgentToolConfig.user_id == user_id).all()
    }
    changed = False
    for name, metadata in TOOL_REGISTRY.items():
        config = existing.get(name)
        if config is None:
            db.add(_new_config(user_id, metadata))
            changed = True
            continue
        if _sync_metadata(config, metadata):
            changed = True
    if changed:
        db.commit()


def list_tool_configs(db: Session, user_id: int) -> list[dict]:
    ensure_user_tool_configs(db, user_id)
    configs = (
        db.query(AgentToolConfig)
        .filter(AgentToolConfig.user_id == user_id)
        .order_by(AgentToolConfig.category.asc(), AgentToolConfig.name.asc())
        .all()
    )
    return [_to_response(config) for config in configs]


def get_tool_config(db: Session, user_id: int, name: str) -> AgentToolConfig | None:
    ensure_user_tool_configs(db, user_id)
    return (
        db.query(AgentToolConfig)
        .filter(AgentToolConfig.user_id == user_id, AgentToolConfig.name == name)
        .first()
    )


def update_tool_config(
    db: Session,
    user_id: int,
    name: str,
    payload: AgentToolConfigUpdate,
) -> dict | None:
    config = get_tool_config(db, user_id, name)
    if config is None:
        return None

    for field, value in payload.to_update_dict().items():
        setattr(config, field, value)
    if config.enabled is False:
        config.health_status = "disabled"
    elif config.health_status == "disabled":
        config.health_status = _default_health_status(config)

    db.add(config)
    db.commit()
    db.refresh(config)
    return _to_response(config)


def bulk_update_tool_configs(
    db: Session,
    user_id: int,
    payload: AgentToolBulkUpdate,
) -> list[dict]:
    ensure_user_tool_configs(db, user_id)
    enabled_by_name = {item.name: item.enabled for item in payload.tools}
    configs = (
        db.query(AgentToolConfig)
        .filter(AgentToolConfig.user_id == user_id, AgentToolConfig.name.in_(enabled_by_name.keys()))
        .all()
    )
    for config in configs:
        config.enabled = enabled_by_name[config.name]
        config.health_status = _default_health_status(config) if config.enabled else "disabled"
        db.add(config)
    db.commit()
    return list_tool_configs(db, user_id)


def update_tool_health(
    db: Session,
    user_id: int,
    name: str,
    payload: AgentToolHealthUpdate,
) -> dict | None:
    config = get_tool_config(db, user_id, name)
    if config is None:
        return None
    config.health_status = payload.health_status
    config.failure_count += payload.failure_delta
    db.add(config)
    db.commit()
    db.refresh(config)
    return _to_response(config)


def reset_tool_failures(db: Session, user_id: int, name: str) -> dict | None:
    config = get_tool_config(db, user_id, name)
    if config is None:
        return None
    config.failure_count = 0
    config.health_status = _default_health_status(config) if config.enabled else "disabled"
    db.add(config)
    db.commit()
    db.refresh(config)
    return _to_response(config)


def enabled_tool_names_for_user(db: Session, user_id: int) -> set[str]:
    ensure_user_tool_configs(db, user_id)
    return {
        item.name
        for item in db.query(AgentToolConfig)
        .filter(AgentToolConfig.user_id == user_id, AgentToolConfig.enabled.is_(True))
        .all()
        if item.name in TOOL_REGISTRY and _api_key_configured(TOOL_REGISTRY[item.name])
    }


def record_tool_failures_from_state(db: Session, user_id: int, state) -> None:
    failed_names: list[str] = []
    for task in getattr(getattr(state, "reasoning", None), "tasks", []) or []:
        for tool_call in getattr(task, "tool_calls", []) or []:
            status = getattr(tool_call, "status", "")
            if getattr(status, "value", status) == "failed":
                failed_names.append(str(getattr(tool_call, "name", "")))
    failed_names = [name for name in failed_names if name]
    if not failed_names:
        return

    ensure_user_tool_configs(db, user_id)
    configs = (
        db.query(AgentToolConfig)
        .filter(AgentToolConfig.user_id == user_id, AgentToolConfig.name.in_(failed_names))
        .all()
    )
    for config in configs:
        config.failure_count += failed_names.count(config.name)
        config.health_status = "degraded"
        db.add(config)
    db.commit()


def _new_config(user_id: int, metadata: ToolMetadata) -> AgentToolConfig:
    return AgentToolConfig(
        user_id=user_id,
        name=metadata.name,
        description=metadata.description,
        category=metadata.category,
        enabled=metadata.default_enabled,
        requires_api_key=metadata.requires_api_key,
        health_status="healthy" if _api_key_configured(metadata) else "unavailable",
        failure_count=0,
    )


def _sync_metadata(config: AgentToolConfig, metadata: ToolMetadata) -> bool:
    changed = False
    for field, value in {
        "description": metadata.description,
        "category": metadata.category,
        "requires_api_key": metadata.requires_api_key,
    }.items():
        if getattr(config, field) != value:
            setattr(config, field, value)
            changed = True
    if config.health_status == "unknown":
        config.health_status = _default_health_status(config)
        changed = True
    return changed


def _default_health_status(config: AgentToolConfig) -> str:
    metadata = TOOL_REGISTRY.get(config.name)
    if config.enabled is False:
        return "disabled"
    if metadata and not _api_key_configured(metadata):
        return "unavailable"
    return "healthy"


def _api_key_configured(metadata: ToolMetadata) -> bool:
    if not metadata.requires_api_key:
        return True
    return all(bool(getattr(settings, key, None)) for key in metadata.api_key_settings)


def _to_response(config: AgentToolConfig) -> dict:
    metadata = TOOL_REGISTRY.get(config.name)
    return {
        "id": config.id,
        "name": config.name,
        "description": config.description,
        "category": config.category,
        "enabled": config.enabled,
        "requires_api_key": config.requires_api_key,
        "health_status": config.health_status,
        "failure_count": config.failure_count,
        "api_key_configured": _api_key_configured(metadata) if metadata else False,
        "created_at": config.created_at,
        "updated_at": config.updated_at,
    }
