#!/usr/bin/env bash
set -Eeuo pipefail
: "${DEPLOY_HOST:?Missing DEPLOY_HOST}" "${DEPLOY_USER:?Missing DEPLOY_USER}"
: "${DEPLOY_SSH_KEY:?Missing DEPLOY_SSH_KEY}" "${DEPLOY_KNOWN_HOSTS:?Missing DEPLOY_KNOWN_HOSTS}"
: "${GHCR_TOKEN:?Missing GHCR_TOKEN}" "${DEPLOY_IMAGE:?Missing DEPLOY_IMAGE}" "${GITHUB_SHA:?Missing GITHUB_SHA}"
ssh_dir=$(mktemp -d)
trap 'rm -rf "$ssh_dir"' EXIT
chmod 700 "$ssh_dir"
printf '%s\n' "$DEPLOY_SSH_KEY" > "$ssh_dir/key"
printf '%s\n' "$DEPLOY_KNOWN_HOSTS" > "$ssh_dir/known_hosts"
chmod 600 "$ssh_dir/key" "$ssh_dir/known_hosts"
# The server key is restricted to this service's deployment command.
printf '%s\n' "$GHCR_TOKEN" | ssh -i "$ssh_dir/key" \
  -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$ssh_dir/known_hosts" -o ConnectTimeout=15 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
  "${DEPLOY_USER}@${DEPLOY_HOST}" "$GITHUB_SHA $DEPLOY_IMAGE"
