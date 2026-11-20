#!/usr/bin/env python3
"""Compare short deterministic Ollama requests and sample CUDA VRAM."""
import argparse
import json
import subprocess
import threading
import time
import urllib.request


def vram_used(gpu):
    reading = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True, timeout=5
    )
    return int(reading.strip())


def check(port, gpu, baseline, model, stream):
    # keep_alive=0 unloads asynchronously after the response.
    for _ in range(40):
        reading = vram_used(gpu)
        if reading <= baseline + 16:
            break
        time.sleep(0.25)
    if reading > baseline + 16:
        raise RuntimeError("GPU memory did not return to its pre-test baseline")
    samples = [reading]
    finished = threading.Event()

    def sample():
        while not finished.wait(0.2):
            samples.append(vram_used(gpu))

    monitor = threading.Thread(target=sample)
    monitor.start()
    qwen3 = model.startswith("qwen3:")
    endpoint = "generate" if qwen3 else "chat"
    body = {
        "model": model,
        "stream": stream,
        "keep_alive": 0,
        "options": {"num_ctx": 512, "num_batch": 32, "num_gpu": 24,
                    "num_predict": 2 if qwen3 else 16, "temperature": 0, "seed": 42},
    }
    if qwen3:
        body.update(prompt="2 + 2 =", raw=True)
    else:
        body.update(messages=[{"role": "user", "content": "What is 2 plus 2? Reply with just the number."}],
                    think=False)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/{endpoint}",
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            records = [json.loads(line) for line in response if line.strip()]
    finally:
        finished.set()
        monitor.join()
    last = records[-1]
    answer = "".join(record.get("response", record.get("message", {}).get("content", "")) for record in records).strip()
    assert last["done"] and answer.rstrip(".") == "4", (model, stream, answer)
    assert max(samples) > reading, (model, stream, "No increase in measured GPU memory")
    print(json.dumps({
        "port": port, "model": model, "endpoint": endpoint, "stream": stream, "answer": answer,
        "tokens": last["eval_count"], "seconds": round(last["total_duration"] / 1e9, 3),
        "baseline_mib": reading, "sampled_peak_mib": max(samples),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID")
    args = parser.parse_args()
    baseline = vram_used(args.gpu)
    for model in args.models:
        for stream in (False, True):
            check(args.port, args.gpu, baseline, model, stream)
