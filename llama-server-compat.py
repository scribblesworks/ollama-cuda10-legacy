#!/usr/bin/env python3
"""Launch the CUDA 10.2 runner using Ollama's text-generation interface."""
import os
from pathlib import Path
import subprocess
import sys


GPU_LAYER_FLAGS = {"-ngl", "--n-gpu-layers", "--gpu-layers"}


def cuda_devices(directory):
    output = subprocess.check_output(
        [str(directory / "cuda-device-info")], text=True, timeout=10
    )
    devices = []
    for line in output.splitlines():
        index, major, minor, total, free, name = line.split("\t", maxsplit=5)
        devices.append((int(index), int(major), int(minor), int(total), int(free), name))
    return devices


def selected_device(directory, arch):
    devices = cuda_devices(directory)
    if len(devices) > 1:
        raise ValueError("Expose exactly one CUDA device to the container")
    if not devices:
        return None
    device = devices[0]
    _, major, minor, _, _, _ = device
    if 10 * major + minor != arch:
        raise ValueError(f"GPU compute capability {major}.{minor} does not match image target sm_{arch}")
    return device


def report_device(directory, arch, list_devices):
    device = selected_device(directory, arch)
    if device is not None:
        index, major, minor, total, free, name = device
        if list_devices:
            print(f"  CUDA{index}: {name} ({total} MiB, {free} MiB free)")
        print(f"Device {index}: {name}, compute capability {major}.{minor}", file=sys.stderr)
    print(f"system_info: CUDA : ARCHS = {arch * 10}", file=sys.stderr)


def translate(args):
    ignored = {"--no-webui", "--offline", "--no-log-prefix", "--no-log-timestamps", "--no-jinja"}
    accepted_values = {
        "--log-verbosity": {"4"},
        "--chat-template": {"chatml"},
        "--keep": {"4"},
        "--load-mode": {"none"},
        "--flash-attn": {"off", "auto"},
        "--cache-type-k": {"f16"},
        "--cache-type-v": {"f16"},
    }
    result = []
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if arg in ignored or arg == "--context-shift":
            continue
        if arg in accepted_values or arg == "-ub":
            if i == len(args):
                raise ValueError(f"{arg} requires a value")
            value = args[i]
            i += 1
            if arg == "-ub":
                if "-b" not in args or value != args[args.index("-b") + 1]:
                    raise ValueError("legacy runner requires matching batch and microbatch sizes")
            elif value not in accepted_values[arg]:
                raise ValueError(f"legacy runner does not support {arg} {value}")
            continue
        result.append(arg)
    if not GPU_LAYER_FLAGS.intersection(result):
        layers = int(os.getenv("LEGACY_CUDA_GPU_LAYERS", "24"))
        if layers < 0:
            raise ValueError("LEGACY_CUDA_GPU_LAYERS must be nonnegative")
        result.extend(("-ngl", str(layers)))
    return result


def main(args):
    directory = Path(__file__).resolve().parent
    arch = int((directory / "cuda-arch").read_text())
    if "--list-devices" in args or "--model" not in args:
        report_device(directory, arch, "--list-devices" in args)
        return
    launch = translate(args)
    for index, value in enumerate(launch):
        if value in GPU_LAYER_FLAGS:
            if index + 1 == len(launch):
                raise ValueError(f"{value} requires a layer count")
            if int(launch[index + 1]) != 0 and selected_device(directory, arch) is None:
                raise ValueError(f"No visible CUDA device for sm_{arch}; use num_gpu=0 for CPU inference")
            break
    runner = directory / "llama-server-legacy"
    os.execv(runner, [str(runner), *launch])


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(f"CUDA 10.2 runner: {exc}")
