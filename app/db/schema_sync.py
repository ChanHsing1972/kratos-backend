from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


BODY_METRIC_COLUMN_DDL = {
    "height_cm": "NUMERIC(5, 2)",
    "target_weight_kg": "NUMERIC(5, 2)",
    "sleep_hours": "NUMERIC(4, 2)",
}


def ensure_runtime_schema(engine: Engine) -> None:
    """Add nullable columns introduced after the course prototype shipped.

    The project currently relies on SQLAlchemy create_all instead of Alembic.
    create_all does not alter existing tables, so this keeps local/Postgres
    databases compatible without dropping legacy columns or user data.
    """
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "body_metrics" not in table_names:
        return

    existing_columns = {
        column["name"]
        for column in inspector.get_columns("body_metrics")
    }
    missing_columns = [
        (name, ddl)
        for name, ddl in BODY_METRIC_COLUMN_DDL.items()
        if name not in existing_columns
    ]
    if not missing_columns:
        return

    with engine.begin() as connection:
        for name, ddl in missing_columns:
            connection.execute(text(f"ALTER TABLE body_metrics ADD COLUMN {name} {ddl}"))
