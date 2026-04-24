# CI/CD setup

This project deploys automatically when `dev` or `master` is pushed.

Required GitLab CI/CD variables:

- `SSH_PASSWORD`: server SSH password, or use `SSH_PRIVATE_KEY` instead.
- `DEPLOY_ENV_FILE`: production `.env` content for this backend.

Default deployment target:

- Host: `192.0.2.1`
- Directory: `/srv/se3/agent-backend`
- Service: `agent-backend`
- Port: `8000`

The pipeline uses the Tsinghua PyPI mirror and caches pip downloads to keep builds faster in mainland China.
