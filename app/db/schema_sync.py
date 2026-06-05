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
    "measured_at": "TIMESTAMP",
    "source": "VARCHAR(30) DEFAULT 'manual' NOT NULL",
    "external_id": "VARCHAR(120)",
}

WORKOUT_LOG_COLUMN_DDL = {
    "duration_seconds": "INTEGER",
}

USER_PROFILE_COLUMN_DDL = {
    "hyperate_id": "VARCHAR(64)",
}

AGENT_RUN_COLUMN_DDL = {
    "memory_payload": "JSON",
    "result_payload": "JSON",
    "client_turn_id": "VARCHAR(64)",
}

TRAINING_PLAN_COLUMN_DDL = {
    "plan_kind": "VARCHAR(30) DEFAULT 'program' NOT NULL",
    "duration_weeks": "INTEGER",
    "schedule_json": "JSON",
}

AGENT_CHECKIN_COLUMN_DDL = {
    "checkin_date": "DATE",
    "sleep_hours": "NUMERIC(4, 2)",
    "pain_notes": "TEXT",
    "source": "VARCHAR(30) DEFAULT 'manual' NOT NULL",
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

EXERCISE_VIDEO_LINK_COLUMN_DDL = {
    "search_query": "VARCHAR(240)",
}

USER_COLUMN_DDL = {
    "avatar_url": "TEXT",
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

    if "user_profiles" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("user_profiles")
        }
        missing_columns.extend(
            ("user_profiles", name, ddl)
            for name, ddl in USER_PROFILE_COLUMN_DDL.items()
            if name not in existing_columns
        )

    if "training_plans" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("training_plans")
        }
        missing_columns.extend(
            ("training_plans", name, ddl)
            for name, ddl in TRAINING_PLAN_COLUMN_DDL.items()
            if name not in existing_columns
        )

    if "agent_checkins" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("agent_checkins")
        }
        missing_columns.extend(
            ("agent_checkins", name, ddl)
            for name, ddl in AGENT_CHECKIN_COLUMN_DDL.items()
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

    if "exercise_video_links" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("exercise_video_links")
        }
        missing_columns.extend(
            ("exercise_video_links", name, ddl)
            for name, ddl in EXERCISE_VIDEO_LINK_COLUMN_DDL.items()
            if name not in existing_columns
        )

    if "users" in table_names:
        existing_columns = {
            column["name"]
            for column in inspector.get_columns("users")
        }
        missing_columns.extend(
            ("users", name, ddl)
            for name, ddl in USER_COLUMN_DDL.items()
            if name not in existing_columns
        )

    if missing_columns:
        with engine.begin() as connection:
            for table_name, name, ddl in missing_columns:
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {name} {ddl}"))

    if "heart_rate_samples" not in table_names:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS heart_rate_samples (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        workout_session_id INTEGER NOT NULL REFERENCES workout_logs(id) ON DELETE CASCADE,
                        bpm INTEGER NOT NULL,
                        source VARCHAR(30) DEFAULT 'hyperate' NOT NULL,
                        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                        CONSTRAINT ck_heart_rate_samples_bpm_range CHECK (bpm >= 30 AND bpm <= 230)
                    )
                    """
                )
            )

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_heart_rate_samples_user_session_time "
                "ON heart_rate_samples (user_id, workout_session_id, recorded_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_heart_rate_samples_workout_session_id "
                "ON heart_rate_samples (workout_session_id)"
            )
        )

    if "agent_runs" in table_names:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_runs_user_client_turn "
                    "ON agent_runs (user_id, client_turn_id) WHERE client_turn_id IS NOT NULL"
                )
            )
