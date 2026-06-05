ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS hyperate_id VARCHAR(64);

CREATE TABLE IF NOT EXISTS heart_rate_samples (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workout_session_id INTEGER NOT NULL REFERENCES workout_logs(id) ON DELETE CASCADE,
    bpm INTEGER NOT NULL,
    source VARCHAR(30) DEFAULT 'hyperate' NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT ck_heart_rate_samples_bpm_range CHECK (bpm >= 30 AND bpm <= 230)
);

CREATE INDEX IF NOT EXISTS ix_heart_rate_samples_user_session_time
    ON heart_rate_samples (user_id, workout_session_id, recorded_at);

CREATE INDEX IF NOT EXISTS ix_heart_rate_samples_workout_session_id
    ON heart_rate_samples (workout_session_id);
