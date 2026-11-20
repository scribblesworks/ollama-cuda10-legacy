#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
device="${NVIDIA_CDI_DEVICE:-nvidia.com/gpu=0}"
gpu="${NVIDIA_SMI_GPU:-${device#nvidia.com/gpu=}}"
port="${TEST_PORT:-11436}"
image="${OLLAMA_LEGACY_IMAGE:-ollama-cuda10:local}"
candidate=ollama-cuda10-test-$$
docker run -d --rm --name "$candidate" --device "$device" \
    -p "127.0.0.1:$port:11434" -v /var/lib/ollama \
    -e OLLAMA_HOST=0.0.0.0:11434 -e OLLAMA_NO_CLOUD=1 -e OLLAMA_GO_TEMPLATE=1 \
    -e OLLAMA_CONTEXT_LENGTH=512 "$image" >/dev/null
cleanup() {
    local status=$?
    if (( status != 0 )); then
        docker logs --tail 60 "$candidate" >&2 || true
    fi
    docker stop "$candidate" >/dev/null 2>&1 || true
}
trap cleanup EXIT
ready=false
for ((attempt=0; attempt<40; attempt++)); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/api/version" 2>/dev/null; then
        printf '\n'
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]]
for model in qwen2.5:1.5b qwen3:0.6b; do
    docker exec "$candidate" env OLLAMA_HOST=127.0.0.1:11434 ollama pull "$model"
done
python3 tests/api-check.py "$port" qwen2.5:1.5b qwen3:0.6b --gpu "$gpu"
logs=$(docker logs "$candidate" 2>&1)
grep -E 'offloaded [1-9][0-9]*/[0-9]+ layers to GPU' <<< "$logs"
if grep -Ei 'cuda error|falling back to cpu' <<< "$logs"; then
    exit 1
fi
