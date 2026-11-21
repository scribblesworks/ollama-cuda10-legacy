# Ollama for CUDA 10.2

Run Ollama on older NVIDIA GPUs using a llama.cpp runner built with CUDA 10.2. The image pairs Ollama v0.40.16 with a pinned llama.cpp revision, a legacy CUDA patch, and a dense Qwen3 backport. A small adapter connects the current Ollama API to the older runner.

## How it fits together

```text
       Ollama CLI or API client
                  |
                  | HTTP on 127.0.0.1:11434 by default
                  v
  +----------------------------------------------+
  | Docker container                             |
  |                                              |
  | Ollama server <------> model volume          |
  |       |                                      |
  |       v                                      |
  | Runner adapter (CLI and device discovery)    |
  |       |                                      |
  |       v                                      |
  | Patched llama.cpp (Qwen3 + legacy CUDA)      |
  |       |                                      |
  |       v                                      |
  | CUDA 10.2 runtime + cuBLAS                    |
  +-------|--------------------------------------+
          | GPU access via NVIDIA Container Toolkit (CDI)
          v
     Host NVIDIA driver
          |
          v
       NVIDIA GPU
```

At build time, `CUDA_ARCH` selects the GPU target for llama.cpp. At startup, `NVIDIA_CDI_DEVICE` selects which physical GPU the container can access.

## Requirements

- Linux x86-64, Docker with Compose and CDI support, and NVIDIA Container Toolkit. Use a host driver that supports both your GPU and CUDA 10.2. The compiler and CUDA user-space libraries are installed inside the build image.
- One GPU exposed to the container. Set `CUDA_ARCH` to its compute capability without the dot, such as `30` for CC 3.0. The default is `30`. The build accepts CUDA 10.2's `30`, `32`, `35`, `37`, `50`, `52`, `53`, `60`, `61`, `62`, `70`, `72`, and `75` targets. Hardware validation is limited to CC 3.0.
- A CDI device from `nvidia-ctk cdi list`. See the [NVIDIA CDI setup guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

## Build and run

```bash
cp .env.example .env
# Set CUDA_ARCH and NVIDIA_CDI_DEVICE to match your GPU.
./build.sh
docker compose up -d
docker compose exec -e OLLAMA_HOST=127.0.0.1:11434 ollama ollama pull qwen3:0.6b
```

Compose serves the native Ollama API on loopback port 11434 by default. Set `OLLAMA_PORT` in `.env` if you already run Ollama locally.

```bash
curl -fsS http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:0.6b","prompt":"2 + 2 =","raw":true,"stream":false,"options":{"num_predict":2,"temperature":0}}'
docker compose logs ollama
```

Check the logs for a positive GPU layer-offload count. Set `LEGACY_CUDA_GPU_LAYERS` or use the request option `num_gpu` to adjust offload. Lower the layer count and context length if GPU memory is limited. Ollama runs as root inside the container and uses `/root/.ollama` for state and models, as in the official image. `docker compose down` stops the service and retains the volume.

If you use a separate Compose file with the previous image, mount the same volume or host directory at `/root/.ollama` instead of `/var/lib/ollama`. Remove any old `HOME`, `OLLAMA_MODELS`, or `user: "65534:65534"` overrides. The existing top-level `models/` directory will be picked up at the new mount point.

## Test and limitations

`tests/test-container.sh` starts a temporary container on port 11436 with its own model storage. It tests Qwen2.5 chat and Qwen3 completion, both with and without streaming. It samples GPU memory using `nvidia-smi`. Run it with the same image and CDI device selection as your build. For a host with other GPU workloads, leave enough free VRAM or run the tests between workloads.

```bash
NVIDIA_CDI_DEVICE=nvidia.com/gpu=0 ./tests/test-container.sh
```

The runner was validated with Qwen2.5 1.5B and dense Qwen3 0.6B. Qwen3 MoE, Qwen3.5, tool calling, and general chat quality have not been validated. See [VALIDATION.md](VALIDATION.md) for the pinned revisions, results, and CUDA 10.2 constraints.

The repository includes source patches and build scripts. The build fetches Ollama's published executable and NVIDIA CUDA packages. See [llama.cpp's license](LICENSE.llama.cpp) and [Ollama's license](LICENSE.ollama) for the upstream components.

## Acknowledgments

Thanks to theIvanR for documenting llama.cpp compatibility work on older NVIDIA GPUs. That README was the starting point for this project's CUDA 10.2 adaptation. [theIvanR/llama-on-legacy-gpu](https://github.com/theIvanR/llama-on-legacy-gpu)
