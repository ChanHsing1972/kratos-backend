from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


BODY_METRIC_COLUMN_DDL = {
    "height_cm": "NUMERIC(5, 2)",
    "weight_kg": "NUMERIC(5, 2)",
    "target_weight_kg": "NUMERIC(5, 2)",
    "body_fat_percentage": "NUMERIC(5, 2)",
    "skeletal_muscle_mass_kg": "NUMERIC(5, 2)",
    "bmi": "NUMERIC(5, 2)",
    "chest_cm": "NUMERIC(5, 2)",
    "waist_cm": "NUMERIC(5, 2)",
    "hip_cm": "NUMERIC(5, 2)",
    "sleep_hours": "NUMERIC(4, 2)",
    "notes": "TEXT",
}

WORKOUT_LOG_COLUMN_DDL = {
    "duration_seconds": "INTEGER",
}

AGENT_RUN_COLUMN_DDL = {
    "memory_payload": "JSON",
    "result_payload": "JSON",
}

CONVERSATION_SESSION_COLUMN_DDL = {
    "title": "VARCHAR(200) DEFAULT '新会话' NOT NULL",
    "summary": "TEXT DEFAULT '' NOT NULL",
    "is_pinned": "BOOLEAN DEFAULT FALSE NOT NULL",
    "is_archived": "BOOLEAN DEFAULT FALSE NOT NULL",
    "is_deleted": "BOOLEAN DEFAULT FALSE NOT NULL",
    "is_shared": "BOOLEAN DEFAULT FALSE NOT NULL",
    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL",
    "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL",
}


def ensure_runtime_schema(engine: Engine) -> None:
    """Add nullable columns introduced after the course prototype shipped.

    The project currently relies on SQLAlchemy create_all instead of Alembic.
    create_all does not alter existing tables, so this keeps local/Postgres
    databases compatible without dropping legacy columns or user data.
    """
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    missing_columns: list[tuple[str, str, str]] = []

    if "body_metrics" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("body_metrics")
        }
        missing_columns.extend(
            ("body_metrics", name, ddl)
            for name, ddl in BODY_METRIC_COLUMN_DDL.items()
            if name not in existing_columns
        )

    if "workout_logs" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("workout_logs")
        }
        missing_columns.extend(
            ("workout_logs", name, ddl)
            for name, ddl in WORKOUT_LOG_COLUMN_DDL.items()
            if name not in existing_columns
        )

    if "agent_runs" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("agent_runs")
        }
        missing_columns.extend(
            ("agent_runs", name, ddl)
            for name, ddl in AGENT_RUN_COLUMN_DDL.items()
            if name not in existing_columns
        )

    if "conversation_sessions" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("conversation_sessions")
        }
        missing_columns.extend(
            ("conversation_sessions", name, ddl)
            for name, ddl in CONVERSATION_SESSION_COLUMN_DDL.items()
            if name not in existing_columns
        )

    if missing_columns:
        with engine.begin() as connection:
            for table_name, name, ddl in missing_columns:
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {name} {ddl}"))
