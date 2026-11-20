#!/usr/bin/env python3
"""Extract the published upstream Ollama 0.40.16 Linux amd64 executable."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

digest = "0f9ace102f525080411253638f08b21f7c483d1ee9f49284ee17f89b8e8cee03"
token_url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ollama/ollama:pull"
with urllib.request.urlopen(token_url, timeout=45) as response:
    token = json.load(response)["token"]
request = urllib.request.Request(
    "https://registry-1.docker.io/v2/ollama/ollama/blobs/sha256:" + digest,
    headers={"Authorization": "Bearer " + token},
)
with urllib.request.urlopen(request, timeout=90) as response:
    layer = response.read()
if hashlib.sha256(layer).hexdigest() != digest:
    raise SystemExit("Ollama image layer digest mismatch")
with tarfile.open(fileobj=io.BytesIO(layer), mode="r:gz") as archive:
    executable = archive.extractfile("usr/bin/ollama")
    if executable is None:
        raise SystemExit("Ollama executable not found")
    target = Path(sys.argv[1])
    target.write_bytes(executable.read())
    target.chmod(0o755)
