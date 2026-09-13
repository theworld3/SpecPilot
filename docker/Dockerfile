ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3
FROM ${BASE_IMAGE}

ARG VLLM_COMMIT=da8ec2826898d47bd2fb424bc607052a6dc515c1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git init vllm \
    && cd vllm \
    && git remote add origin https://github.com/vllm-project/vllm.git \
    && git fetch --depth 1 origin ${VLLM_COMMIT} \
    && git checkout FETCH_HEAD

COPY . /opt/specpilot-vllm
RUN python -m pip install --no-cache-dir -e "/opt/specpilot-vllm[gpu]" \
    && cd /opt/vllm \
    && git apply /opt/specpilot-vllm/patches/vllm-da8ec282.patch \
    && VLLM_USE_PRECOMPILED=1 python -m pip install --no-cache-dir -e .

WORKDIR /workspace

