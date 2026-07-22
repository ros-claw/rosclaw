#!/usr/bin/env bash
set -euo pipefail

repo_root="${ROSCLAW_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$repo_root"
export PATH="$repo_root/.venv/bin:$PATH"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

hub_root=$(mktemp -d /tmp/rosclaw-hub-simforge.XXXXXX)
registry_dir="$hub_root/registry"
mkdir -p "$registry_dir"
touch "$registry_dir/catalog.jsonl"
export ROSCLAW_HOME="$hub_root/home"
export ROSCLAW_HUB_SIGNING_KEY="$repo_root/tests/fixtures/hub_keys/fixture-private.pem"
export ROSCLAW_HUB_SIGNING_KEY_ID=rosclaw-hub-fixture-v1
export ROSCLAW_HUB_TRUST_STORE="$repo_root/tests/fixtures/hub_keys/trust.json"

port="${ROSCLAW_HUB_TEST_PORT:-18787}"
.venv/bin/python tests/fixtures/fake_registry/server.py \
  --directory "$registry_dir" --port "$port" &
registry_pid=$!
trap 'kill "$registry_pid" 2>/dev/null || true' EXIT

for _attempt in 1 2 3 4 5 6 7 8 9 10; do
  if .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/catalog.jsonl', timeout=1).read()" 2>/dev/null; then
    break
  fi
  sleep 0.2
done

registry="http://127.0.0.1:$port"
ref=rosclaw://digital_twin/rosclaw/g1-mujoco-basic@0.5.0
asset=tests/fixtures/hub_assets/digital_twin_valid

rosclaw hub validate "$asset/manifest.yaml"
rosclaw hub verify "$asset"
rosclaw hub login --registry "$registry" --token fake-valid-token --insecure-local
rosclaw hub publish "$asset" --private --sign --registry "$registry" --output "$hub_root/dist"
rosclaw hub sync --registry "$registry" --clear
rosclaw hub search g1-mujoco-basic --type digital_twin
rosclaw hub install "$ref" --dry-run --skip-health
rosclaw hub install "$ref" --yes --skip-health
rosclaw hub list --installed
rosclaw hub uninstall "$ref" --yes
test "$(rosclaw hub list --installed --json)" = "[]"

echo "ROSCLAW_HUB_VERIFY_OK root=$hub_root"
