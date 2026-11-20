FROM ubuntu:18.04 AS cuda-build
ARG CUDA_ARCH=30
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git g++ make
WORKDIR /build
COPY cuda102.sha256 fetch-cuda102.sh build-llama.sh cuda-device-info.cpp ./
RUN ./fetch-cuda102.sh / && ln -s cuda-10.2 /usr/local/cuda
COPY patches/ ./patches/
RUN git init llama.cpp \
    && git -C llama.cpp fetch --depth=1 https://github.com/ggml-org/llama.cpp.git 229ffff872f8ad0d21c997d18ee7a23692ae60a0 \
    && git -C llama.cpp checkout --detach FETCH_HEAD \
    && git -C llama.cpp apply /build/patches/llama-cuda10.patch /build/patches/qwen3.patch \
    && CUDA_ARCH="$CUDA_ARCH" ./build-llama.sh \
    && printf '%s\n' "$CUDA_ARCH" > /cuda-arch \
    && g++ -std=c++11 -I/usr/local/cuda/include cuda-device-info.cpp \
        -L/usr/local/cuda/lib64 -lcudart -o /cuda-device-info \
    && mkdir /cuda-runtime \
    && cp -a /usr/local/cuda/lib64/libcudart.so.10.2* /usr/local/cuda/lib64/libcublas.so.10* \
        /usr/local/cuda/lib64/libcublasLt.so.10* /cuda-runtime/

FROM debian:13-slim AS upstream-ollama
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates
COPY fetch-ollama.py /fetch-ollama.py
RUN python3 /fetch-ollama.py /ollama

FROM debian:13-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgomp1 ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -o 65534 -g 65534 /var/lib/ollama
COPY --from=cuda-build /cuda-runtime/ /opt/cuda102/
COPY --from=cuda-build /build/llama.cpp/server /usr/lib/ollama/llama-server-legacy
COPY --from=cuda-build /cuda-arch /cuda-device-info /usr/lib/ollama/
COPY --from=upstream-ollama /ollama /usr/bin/ollama
COPY --chmod=755 llama-server-compat.py /usr/lib/ollama/llama-server
COPY LICENSE.llama.cpp LICENSE.ollama /usr/share/doc/ollama-cuda10/
ENV LD_LIBRARY_PATH=/opt/cuda102 HOME=/var/lib/ollama OLLAMA_MODELS=/var/lib/ollama/models
USER 65534:65534
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
