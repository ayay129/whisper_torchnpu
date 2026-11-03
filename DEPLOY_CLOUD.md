# 云平台容器部署验证

当前云环境：Ubuntu 22.04 / aarch64，910B2 64GB，平台显示 NPU ID 5，驱动 25.2.1。当前 SSH 环境本身是容器，仅有 `/usr/local/Ascend/driver`，没有 CANN 和 Docker。

直接在当前平台容器安装 CANN 和 Python 环境。使用平台挂载的驱动，驱动和固件由平台管理。[CANN 9.1.0 官方兼容矩阵](https://gitcode.com/cann/release-management/blob/master/9.1.0/release-notes.md) 中，CANN 9.1.X 兼容 Ascend HDK 25.2.X。

| 组件 | 版本 / 安装位置 |
| --- | --- |
| 驱动 | 平台已有 25.2.1 |
| CANN Toolkit + 910B ops | 9.1.0，安装到当前平台容器 |
| Python | 3.10，项目独立 venv |
| PyTorch / torchaudio | 2.7.1，ARM64 CPU wheel + TorchNPU |
| torch-npu | 2.7.1.post8 |
| pyannote.audio | 3.3.2 |
| lightning / pytorch-lightning | 2.5.5 |
| Transformers | 4.46.3 |

以下命令以 root 执行，每一步成功后再进行下一步。当前终端是 sh 时，先进入 bash：

```bash
bash
```

**1. 安装系统依赖并创建 Python 环境**

先上传本项目并进入项目目录，下面路径替换为云端实际位置：

```bash
cd /path/to/whisperx_imonitor
df -h . /tmp /usr/local/Ascend

apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev \
  build-essential cmake gfortran libblas-dev libnuma-dev \
  zlib1g-dev libffi-dev libssl-dev patchelf \
  ffmpeg libsndfile1 wget curl ca-certificates unzip pciutils

python3.10 -m venv .venv
source .venv/bin/activate
export PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
python -m pip install --upgrade pip wheel
python -m pip install 'numpy<2' attrs cython decorator sympy cffi pyyaml \
  pathlib2 psutil 'protobuf==3.20.0' scipy requests absl-py
```

**2. 安装 CANN 9.1.0 Toolkit 和 910B 算子包**

下载地址和安装参数与 [Ascend 官方 CANN 镜像源码](https://github.com/Ascend/cann-container-image/blob/main/cann/9.1.0-910b-ubuntu22.04-py3.10/Dockerfile) 一致。这里只下载软件环境，不下载模型。

```bash
mkdir -p /tmp/whisperx-cann-install
(
  set -e
  cd /tmp/whisperx-cann-install
  wget --header='Referer: https://www.hiascend.com/' \
    'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-aarch64.run'
  wget --header='Referer: https://www.hiascend.com/' \
    'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run'
  bash Ascend-cann-toolkit_9.1.0_linux-aarch64.run --quiet --install --install-for-all
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  bash Ascend-cann-910b-ops_9.1.0_linux-aarch64.run --quiet --install --install-for-all
)

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
```

本项目使用 TorchNPU eager 推理，不使用 MindIE 或 ATB，因此这条原生安装路径不额外安装 NNAL，不需要编译 `.ts` 模型。

**3. 安装项目配套依赖**

当前 Python 应为项目的 `.venv/bin/python`。普通依赖使用华为源；PyTorch ARM64 CPU wheel 使用官方 CPU 源。

```bash
python -m pip install --no-cache-dir 'torch==2.7.1+cpu' 'torchaudio==2.7.1' \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir --no-deps 'torch-npu==2.7.1.post8'
python -m pip install --no-cache-dir -c constraints.txt -r requirements.txt
python -m pip check
```

**4. 先验证 NPU 环境，再加载模型**

平台显示的物理 ID 5 不等于程序应直接使用 `npu:5`。先验证 `npu:0`；只看到一张卡时，程序使用 `--device-id 0`。如果测试失败，保留完整报错及 `npu-smi info -m` 输出定位。

```bash
npu-smi info -t board -i 5
npu-smi info -m
python - <<'PY'
from importlib.metadata import version
import torch
import torch_npu
import torchaudio

print('torch:', torch.__version__)
print('torchaudio:', torchaudio.__version__)
print('torch-npu:', version('torch-npu'))
assert torch.__version__.split('+')[0] == '2.7.1'
assert torchaudio.__version__.split('+')[0] == '2.7.1'
assert version('torch-npu') == '2.7.1.post8'
assert torch.npu.is_available(), 'NPU unavailable'
print('visible NPU count:', torch.npu.device_count())
torch.npu.set_device(0)
x = torch.ones((32, 32), device='npu:0')
y = x @ x
torch.npu.synchronize()
assert torch.allclose(y.cpu(), torch.full((32, 32), 32.0))
print('NPU matmul OK:', torch.npu.get_device_name(0))
PY
```

**5. 拷贝本地模型，启动服务**

模型布局见 [README.md](README.md)。Whisper 使用 Hugging Face 原始权重，Silero 和 pyannote 使用完整本地模型目录。测试音频先选一份含清晰语音的短 WAV。

```bash
python app.py --models-dir ./models --data-dir ./private-tsb-data \
  --device-id 0 --workers 6 --host 0.0.0.0 --port 8000
```

另开终端验证：

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/tsb/alive
curl -sS http://127.0.0.1:8000/transcribe -F 'file=@test.wav'
curl -sS http://127.0.0.1:8000/transcribe \
  -F 'file=@test.wav' -F 'diarize=true' -F 'speaker_embeddings=true'
```

每次新开终端启动服务前，重新进入项目目录，激活 `.venv`、加载 CANN `set_env.sh` 并设置上面的驱动库路径；安装依赖时还需设置华为 pip 源。若云平台重建容器，软件环境能否保留取决于平台策略，请使用平台提供的保存镜像或持久化机制。

当前只核对了版本配套和安装命令，尚未在这台云端容器执行安装或实测模型。
