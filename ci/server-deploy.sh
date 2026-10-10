#!/usr/bin/env bash
# Installed by an administrator. Forced SSH command; never evaluated from the repository.
set -Eeuo pipefail
service=${1:?service required}
case "$service" in backend|frontend) ;; *) exit 64 ;; esac
repo="ghcr.io/chanhsing1972/kratos-$service"
command=${SSH_ORIGINAL_COMMAND:-}
if [[ ! "$command" =~ ^([0-9a-f]{40})\ (ghcr\.io/chanhsing1972/kratos-(backend|frontend)@sha256:[0-9a-f]{64})$ ]]; then
  echo 'Expected: <40-character commit> <image@sha256:digest>' >&2
  exit 64
fi
revision=${BASH_REMATCH[1]}
image=${BASH_REMATCH[2]}
[[ "$image" == "$repo@sha256:"* ]] || exit 64
IFS= read -r token
[[ -n "$token" ]] || exit 64
cd /srv/se3
state=/srv/se3/deploy
mkdir -p "$state/backups"
chmod 700 "$state" "$state/backups"
# Both repositories share one Compose project; serialize the entire update.
exec 9>"$state/deploy.lock"
flock -w 600 9
runtime=$(mktemp -d "$state/.run-XXXXXXXX")
export DOCKER_CONFIG="$runtime/docker"
mkdir -m 700 "$DOCKER_CONFIG"
compose() {
  docker compose --project-name se3 --project-directory /srv/se3 \
    -f /srv/se3/docker-compose.yml -f "$state/backend.yml" -f "$state/frontend.yml" "$@"
}
changed=false
previous_image=''
cleanup() {
  result=$?
  trap - EXIT
  if [[ $result -ne 0 && "$changed" == true ]]; then
    echo "Deployment failed; restoring previous $service image." >&2
    cp "$runtime/previous.yml" "$state/$service.yml"
    if compose up -d --no-deps --force-recreate --wait --wait-timeout 120 "$service"; then
      echo "Rollback succeeded: $previous_image" >&2
    else
      echo 'Rollback failed; administrator attention required.' >&2
    fi
  fi
  rm -rf "$runtime"
  exit "$result"
}
trap cleanup EXIT
printf '%s\n' "$token" | docker login ghcr.io --username ChanHsing1972 --password-stdin >/dev/null
unset token
# Use the immutable CI-produced digest, never rebuild or git pull on production.
docker pull "$image"
actual=$(docker image inspect "$image" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
[[ "$actual" == "$revision" ]] || { echo 'Image revision mismatch.' >&2; exit 1; }
container=$(compose ps -q "$service")
[[ -n "$container" ]] || { echo 'Existing service is missing.' >&2; exit 1; }
previous_image=$(docker inspect "$container" --format '{{.Image}}')
# Pin the old image as a local tag so rollback is independent of mutable dev tags.
rollback="kratos-$service:rollback-${previous_image#sha256:}"
docker tag "$previous_image" "$rollback"
cp "$state/$service.yml" "$runtime/previous.yml"
printf 'services:\n  %s:\n    image: %s\n' "$service" "$rollback" > "$runtime/previous.yml"
if [[ "$service" == backend ]]; then
  db_name=$(docker exec "$container" python -c 'import os; print(os.environ["PG_DB"])')
  [[ "$db_name" =~ ^[a-zA-Z0-9_]+$ ]] || exit 64
  # Runtime startup may update schemas, so retain a real database backup first.
  runuser -u postgres -- pg_dump --format=custom "$db_name" > "$state/backups/$revision.dump"
  chmod 600 "$state/backups/$revision.dump"
fi
printf 'services:\n  %s:\n    image: %s\n' "$service" "$image" > "$runtime/candidate.yml"
mv "$runtime/candidate.yml" "$state/$service.yml"
changed=true
compose up -d --no-deps --force-recreate --wait --wait-timeout 120 "$service"
# Container health alone is insufficient: check revision and database/route contract.
if [[ "$service" == backend ]]; then
  curl -fsS --max-time 15 http://127.0.0.1:8000/health | python3 -c \
    'import json,sys; d=json.load(sys.stdin); assert d["status"]=="ok" and d["db"] is True and d["revision"]==sys.argv[1], d' "$revision"
  public_path=api/health
else
  curl -fsS --max-time 15 http://127.0.0.1/version.json | python3 -c \
    'import json,sys; d=json.load(sys.stdin); assert d["service"]=="frontend" and d["revision"]==sys.argv[1], d' "$revision"
  public_path=version.json
fi
public_ok=false
for attempt in {1..12}; do
  if curl -fsS --max-time 15 -H 'Cache-Control: no-cache' "https://kratos.qzz.io/$public_path?ci=$revision" | \
      python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["revision"]==sys.argv[1], d' "$revision"; then
    public_ok=true
    break
  fi
  sleep 5
done
[[ "$public_ok" == true ]] || { echo 'Public revision check failed.' >&2; exit 1; }
printf '{"service":"%s","revision":"%s","image":"%s","previous_image":"%s","deployed_at":"%s"}\n' \
  "$service" "$revision" "$image" "$previous_image" "$(date -u +%FT%TZ)" > "$state/$service.json"
changed=false
echo "Deployment verified: $service $revision"
