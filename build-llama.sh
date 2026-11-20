#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/llama.cpp"

: "${CUDA_HOME:=/usr/local/cuda-10.2}"
: "${CUDA_HOST_CXX:=/usr/bin/g++-7}"
arch="${CUDA_ARCH:-30}"
case "$arch" in
    30|32|35|37|50|52|53|60|61|62|70|72|75) ;;
    *) printf 'Unsupported CUDA 10.2 architecture: %s\n' "$arch" >&2; exit 1 ;;
esac
nvcc="$CUDA_HOME/bin/nvcc"
if [[ "$("$nvcc" --version)" != *"release 10.2,"* ]]; then
    printf 'CUDA 10.2 is required.\n' >&2
    exit 1
fi
if (( $# == 0 )); then
    set -- server
fi

# The upstream Jetson recipe also works with nvcc versions that predate
# --forward-unknown-to-host-compiler.
make -j"${JOBS:-4}" LLAMA_CUDA=1 LLAMA_NO_NATIVE=1 CUDA_PATH="$CUDA_HOME" \
    CUDA_DOCKER_ARCH="sm_$arch" LLAMA_CUDA_NVCC="$nvcc" LLAMA_CUDA_CCBIN="$CUDA_HOST_CXX" \
    JETSON_EOL_MODULE_DETECT=1 CUDA_CXXFLAGS=-march=x86-64 \
    LLAMA_CUDA_F16= LLAMA_CUDA_DMMV_F16= LLAMA_CUDA_FORCE_MMQ= \
    CFLAGS="-march=x86-64" HOST_CXXFLAGS=-march=x86-64 \
    LDFLAGS="-L$CUDA_HOME/lib64/stubs" "$@"

"$CUDA_HOME/bin/cuobjdump" --list-elf ggml-cuda.o | grep -F "sm_$arch"
