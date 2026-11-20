# Validation

| Component | Revision |
| --- | --- |
| Runner base | llama.cpp `229ffff872f8ad0d21c997d18ee7a23692ae60a0` |
| Dense Qwen3 backport | llama.cpp b5092, `d3bd7193ba66c15963fd1c59448f22019a8caf6e` |
| Ollama | v0.40.16 |
| CUDA and cuBLAS | 10.2.89 and 10.2.3.254 |

## CUDA compatibility

The runner uses default cuBLAS math for pre-Volta devices. The Kepler path
avoids batched extended GEMM, uses FP32 SGEMM, and guards dequantization
tile widths. Native CPU tuning is disabled. CUDA Graphs, optional FP16
arithmetic, forced MMQ, and flash attention are outside the validated path.

A direct compile probe against llama.cpp
`6f5c9ad21746f69bbb7f3cb295a90e53fe316963` failed. CUDA 10.2 nvcc rejected
C++17. Compiling its CUDA vendor header in C++14 mode failed because
`cuda_bf16.h` was unavailable. The container uses the pinned runner with
the architecture backport.

## Public checkout checks

Applied both patches to the pinned source and ran `CUDA_ARCH=30 ./build-llama.sh`
in an Ubuntu 18.04 build environment with GCC 7 and CUDA 10.2.89.
The server linked, and `cuobjdump` listed an embedded `sm_30` cubin.
The CUDA device probe compiled and returned an empty result on a host
without a visible CUDA device. Adapter smoke checks used simulated devices
to cover discovery, architecture mismatch, single-GPU enforcement, and
explicit CPU execution. Shell and Python syntax checks and
`docker compose config --quiet` passed, including with `.env.example`.
The public Docker image and live GPU inference were outside these local checks.

## Hardware evidence

Earlier builds using these runner patches were tested on a CC 3.0 GPU
with driver 470.256.02. Ollama v0.40.16 generated the expected output
with 24 of 29 model layers offloaded. Each model passed streaming and
non-streaming requests with context 512 and batch 32.

| Model | API | Sampled peak VRAM |
| --- | --- | ---: |
| Qwen2.5 1.5B | Chat | 978 MiB |
| Qwen3 0.6B | Raw completion | 512 MiB |

The Qwen3 build also passed non-streaming completion in a deployed
Docker container. A separate Qwen3 chat prompt answered incorrectly
on both CPU and GPU. The earlier hardware results predate the
generalized GPU discovery and Compose configuration in this checkout.
Other architectures have not been hardware tested.
