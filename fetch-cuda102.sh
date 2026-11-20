#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
destination="${1:?Provide an absolute path to extract the CUDA packages into}"
[[ "$destination" = /* ]]
manifest="$PWD/cuda102.sha256"
repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64
mkdir -p "$destination" "$destination/.debs"
cd "$destination/.debs"
while read -r sha file; do
    if ! printf '%s  %s\n' "$sha" "$file" | sha256sum --check --status; then
        curl --fail --silent --show-error --location --retry 3 "$repo/$file" -o "$file"
    fi
done < "$manifest"
sha256sum --check "$manifest"
while read -r sha file; do
    dpkg-deb --extract "$file" "$destination"
done < "$manifest"
