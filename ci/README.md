# CI/CD setup

This project verifies merge requests and deploys only from protected branches.

Required GitLab CI/CD variables:

- `SSH_PASSWORD`: server SSH password, or use `SSH_PRIVATE_KEY` instead.
- `DEPLOY_ENV_FILE`: production `.env` content for this backend.
- `DEPLOY_HOST`: deployment host.

Default deployment layout:

- Directory: `/srv/se3/agent-backend`
- Service: `agent-backend`
- Port: `8000`

The pipeline runs compile checks and pytest for merge requests, uses the Tsinghua PyPI mirror, and caches pip downloads to keep builds faster in mainland China.
