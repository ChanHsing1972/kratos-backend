ALTER TABLE body_metrics
    ADD COLUMN IF NOT EXISTS thigh_cm NUMERIC(5, 2),
    ADD COLUMN IF NOT EXISTS calf_cm NUMERIC(5, 2),
    ADD COLUMN IF NOT EXISTS arm_cm NUMERIC(5, 2);

CREATE TABLE IF NOT EXISTS health_metrics (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    metric_date DATE,
    sleep_hours NUMERIC(4, 2),
    active_kcal NUMERIC(8, 1),
    dietary_kcal NUMERIC(8, 1),
    hrv_ms NUMERIC(6, 2),
    stress_level INTEGER,
    resting_heart_rate INTEGER,
    vo2_max NUMERIC(5, 2),
    blood_oxygen_percentage NUMERIC(5, 2),
    notes TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    measured_at TIMESTAMP,
    source VARCHAR(30) DEFAULT 'manual' NOT NULL,
    external_id VARCHAR(120)
);

CREATE INDEX IF NOT EXISTS ix_health_metrics_user_metric_date
    ON health_metrics (user_id, metric_date);

COMMENT ON COLUMN body_metrics.skeletal_muscle_mass_kg IS
    'deprecated: retained for historical rows; no longer surfaced in the data center';
COMMENT ON COLUMN body_metrics.sleep_hours IS
    'deprecated: use health_metrics.sleep_hours';
COMMENT ON COLUMN agent_checkins.energy_level IS
    'deprecated: daily readiness is no longer a top-level data center metric';
COMMENT ON COLUMN agent_checkins.sleep_quality IS
    'deprecated: use health_metrics.sleep_hours and notes';
COMMENT ON COLUMN agent_checkins.soreness_level IS
    'deprecated: use workout log notes or health/body metrics';
