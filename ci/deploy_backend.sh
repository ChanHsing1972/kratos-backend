#!/usr/bin/env bash
set -euo pipefail

: "${SERVICE_NAME:?SERVICE_NAME is required}"
: "${APP_PORT:?APP_PORT is required}"

DEPLOY_HOST="${DEPLOY_HOST:-192.0.2.1}"
DEPLOY_USER="${DEPLOY_USER:-root}"
DEPLOY_PORT="${DEPLOY_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/srv/se3/$SERVICE_NAME}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
COMMIT_SHA="${CI_COMMIT_SHA:-manual}"
ARCHIVE="/tmp/${SERVICE_NAME}-${COMMIT_SHA}.tar.gz"
REMOTE_ARCHIVE="/tmp/${SERVICE_NAME}-${COMMIT_SHA}.tar.gz"
REMOTE_ENV="/tmp/${SERVICE_NAME}-${COMMIT_SHA}.env"
ENV_UPLOADED="0"
SSH_TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$SSH_TMP_DIR"
  rm -f "$ARCHIVE"
}
trap cleanup EXIT

tar \
  --format=ustar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.env' \
  --exclude='.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -czf "$ARCHIVE" .

ssh_common=(-p "$DEPLOY_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
scp_common=(-P "$DEPLOY_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

if [[ -n "${SSH_PRIVATE_KEY:-}" ]]; then
  SSH_KEY_FILE="$SSH_TMP_DIR/id_deploy"
  if [[ -f "$SSH_PRIVATE_KEY" ]]; then
    cp "$SSH_PRIVATE_KEY" "$SSH_KEY_FILE"
  else
    printf '%s\n' "$SSH_PRIVATE_KEY" > "$SSH_KEY_FILE"
  fi
  chmod 600 "$SSH_KEY_FILE"
  ssh_cmd=(ssh -i "$SSH_KEY_FILE" "${ssh_common[@]}")
  scp_cmd=(scp -i "$SSH_KEY_FILE" "${scp_common[@]}")
elif [[ -n "${SSH_PASSWORD:-}" ]]; then
  ssh_cmd=(sshpass -p "$SSH_PASSWORD" ssh "${ssh_common[@]}")
  scp_cmd=(sshpass -p "$SSH_PASSWORD" scp "${scp_common[@]}")
else
  echo "Set SSH_PRIVATE_KEY or SSH_PASSWORD in GitLab CI/CD Variables." >&2
  exit 1
fi

"${scp_cmd[@]}" "$ARCHIVE" "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_ARCHIVE"

if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  LOCAL_ENV="$SSH_TMP_DIR/.env"
  printf '%s\n' "$DEPLOY_ENV_FILE" > "$LOCAL_ENV"
  "${scp_cmd[@]}" "$LOCAL_ENV" "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_ENV"
  ENV_UPLOADED="1"
fi

"${ssh_cmd[@]}" "$DEPLOY_USER@$DEPLOY_HOST" \
  "SERVICE_NAME='$SERVICE_NAME' APP_PORT='$APP_PORT' REMOTE_DIR='$REMOTE_DIR' PIP_INDEX_URL='$PIP_INDEX_URL' COMMIT_SHA='$COMMIT_SHA' REMOTE_ARCHIVE='$REMOTE_ARCHIVE' REMOTE_ENV='$REMOTE_ENV' ENV_UPLOADED='$ENV_UPLOADED' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

release_dir="$REMOTE_DIR/releases/$COMMIT_SHA"
shared_dir="$REMOTE_DIR/shared"
venv_dir="$REMOTE_DIR/venv"

if command -v apt-get >/dev/null 2>&1; then
  missing_packages=()
  command -v python3 >/dev/null 2>&1 || missing_packages+=(python3)
  dpkg -s python3-venv >/dev/null 2>&1 || missing_packages+=(python3-venv)
  dpkg -s python3-pip >/dev/null 2>&1 || missing_packages+=(python3-pip)
  command -v curl >/dev/null 2>&1 || missing_packages+=(curl)
  if [[ "${#missing_packages[@]}" -gt 0 ]]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y "${missing_packages[@]}"
  fi
fi

mkdir -p "$release_dir" "$shared_dir"
tar -xzf "$REMOTE_ARCHIVE" -C "$release_dir"
rm -f "$REMOTE_ARCHIVE"

if [[ "$ENV_UPLOADED" == "1" ]]; then
  cp "$REMOTE_ENV" "$shared_dir/.env"
  rm -f "$REMOTE_ENV"
fi

if [[ -f "$shared_dir/.env" ]]; then
  cp "$shared_dir/.env" "$release_dir/.env"
elif [[ -f "$release_dir/.env.example" ]]; then
  cp "$release_dir/.env.example" "$release_dir/.env"
fi

python3 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip config set global.index-url "$PIP_INDEX_URL"

requirements_hash="$(sha256sum "$release_dir/requirements.txt" | awk '{print $1}')"
hash_file="$REMOTE_DIR/.requirements.sha256"
installed_hash=""
if [[ -f "$hash_file" ]]; then
  installed_hash="$(cat "$hash_file")"
fi

if [[ ! -x "$venv_dir/bin/uvicorn" || "$requirements_hash" != "$installed_hash" ]]; then
  "$venv_dir/bin/python" -m pip install -U pip
  "$venv_dir/bin/python" -m pip install -r "$release_dir/requirements.txt"
  printf '%s\n' "$requirements_hash" > "$hash_file"
fi

ln -sfn "$release_dir" "$REMOTE_DIR/current"

cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<UNIT
[Unit]
Description=SE3 ${SERVICE_NAME}
After=network.target postgresql.service

[Service]
Type=simple
User=root
WorkingDirectory=${REMOTE_DIR}/current
Environment=PYTHONUNBUFFERED=1
ExecStart=${venv_dir}/bin/uvicorn app.main:app --host 127.0.0.1 --port ${APP_PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:${APP_PORT}/health" >/dev/null; then
    systemctl --no-pager --full status "$SERVICE_NAME"
    exit 0
  fi
  sleep 2
done

journalctl -u "$SERVICE_NAME" -n 120 --no-pager
exit 1
REMOTE_SCRIPT
