# Ascend 910B baseline. This public image supplies CANN and Python, not PyTorch.
ARG CANN_IMAGE=quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.10
FROM ${CANN_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt constraints.txt ./
# CPU wheels use PyTorch's index; torch-npu supplies the NPU backend.
RUN case "$(uname -m)" in \
      x86_64) python -m pip install --no-cache-dir 'numpy<2' \
        'torch==2.7.1+cpu' 'torchaudio==2.7.1+cpu' \
        --extra-index-url https://download.pytorch.org/whl/cpu ;; \
      aarch64) python -m pip install --no-cache-dir 'numpy<2' \
        'torch==2.7.1+cpu' 'torchaudio==2.7.1' \
        --extra-index-url https://download.pytorch.org/whl/cpu ;; \
      *) echo 'Unsupported CPU architecture' >&2; exit 1 ;; \
    esac && \
    python -m pip install --no-cache-dir --no-deps 'torch-npu==2.7.1.post8' && \
    python -m pip install --no-cache-dir -c constraints.txt -r requirements.txt && \
    python -m pip check
# TorchNPU imports may need mounted driver libraries; check its metadata at build time.
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'from importlib.metadata import version; import torch, torchaudio; assert torch.__version__.split("+")[0] == "2.7.1"; assert torchaudio.__version__.split("+")[0] == "2.7.1"; assert version("torch-npu") == "2.7.1.post8"; print("Torch:", torch.__version__, "torchaudio:", torchaudio.__version__, "TorchNPU:", version("torch-npu"))' && \
    ffmpeg -version >/dev/null
COPY . /app
RUN chmod +x /app/entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "app.py", "--models-dir", "/app/models", "--data-dir", "/app/private-tsb-data", "--port", "8000"]
