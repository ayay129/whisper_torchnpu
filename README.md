# whisperx_imonitor

独立的 WhisperX 风格昇腾服务：Silero VAD + Transformers Whisper large-v3 (TorchNPU) + pyannote/WeSpeaker + TSB。无需 MindIE、无需编译 `.ts`。服务只读取本地模型，缺少文件时启动报错，不会自动下载。

## 模型目录

将模型拷贝到项目的 `models/`，或将同样结构的目录挂载至容器 `/app/models`：

```text
models/
├── whisper-large-v3/                  # Hugging Face 格式 Whisper 权重
│   ├── config.json
│   ├── generation_config.json
│   ├── preprocessor_config.json
│   ├── tokenizer.json                 # 或配套 tokenizer 文件
│   ├── tokenizer_config.json
│   └── model.safetensors              # 或 pytorch_model.bin、完整分片
├── silero-vad-6.2.1/                 # whisperx2.0 对应的完整本地仓库
│   ├── hubconf.py
│   └── src/silero_vad/data/silero_vad.jit
├── pyannote-diarization/config.yaml
├── pyannote-segmentation-3.0/pytorch_model.bin
└── pyannote-wespeaker/pytorch_model.bin
```

Silero 的 `hubconf.py` 还会导入同仓库的 `src/silero_vad/` 代码，所以应复制整个 `silero-vad-6.2.1` 目录；pyannote 的三个目录可从原项目复制。程序会将 pyannote 配置里的分段模型和声纹模型路径解析到所提供的 `--models-dir`，不修改原配置。Whisper 必须是 Hugging Face/PyTorch 权重，CTranslate2 的 faster-whisper 权重或 MindIE 编译 `.ts` 不能直接用于这个服务。原服务器的 `whisper_pretrained` 如为完整 Hugging Face 格式，可复制为 `whisper-large-v3`；请同时复制 tokenizer 和模型配置。

## 环境和运行

云平台已提供 SSH 容器、仅挂载驱动且没有 Docker/CANN 时，参考 [云平台容器部署验证](DEPLOY_CLOUD.md) 直接安装；该指引对应当前验证机 910B2 / 驱动 25.2.1 / aarch64。

准备 ffmpeg，以及适配目标设备的 CANN、PyTorch、torchaudio 和 torch-npu。TorchNPU 通过 CANN 调用昇腾算子，依然需要 CANN。`constraints.txt` 固定当前 Docker 构建基线：PyTorch/torchaudio 2.7.1、torch-npu 2.7.1.post8，配套 CANN 9.1.0。其他 CANN 版本需要重新核对配套关系。不要把本项目与昇腾 950 项目的依赖版本直接混用。安装完对应的 torch/torchaudio/torch-npu 后：

```bash
python -m pip install -c constraints.txt -r requirements.txt \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple

python app.py --models-dir ./models --data-dir ./private-tsb-data \
  --device-id 0 --workers 6 --host 0.0.0.0 --port 8000
```

按服务器环境先初始化 CANN、NPU 设备和动态库路径。Silero 和 WeSpeaker fbank/FFT 使用 CPU；Whisper、pyannote 分离和 WeSpeaker 模型尝试使用 NPU。WeSpeaker 的 fbank 计算回退到 CPU 是为了规避已遇到的 complex64 abs 不支持，后续模型算子是否在目标设备运行仍需实测。Silero 阈值 0.7，按约 10 秒合并语音段；与原 faster-whisper 实现的分段、解码和逐段语言结果不保证完全一致。

### Docker

Dockerfile 使用 Ascend 官方公开 CANN 镜像作为基础，默认是 **910B / CANN 9.1.0 / Ubuntu 22.04 / Python 3.10**：

```text
quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.10
```

该镜像供应 CANN 用户态工具包和 Python；Dockerfile 自行安装 ffmpeg、libsndfile1 和下列配套依赖，无需私有基础镜像：

| 组件 | 版本 |
| --- | --- |
| CANN | 9.1.0（含 910B 算子包） |
| PyTorch / torchaudio | 2.7.1 |
| torch-npu 安装包 | 2.7.1.post8（产品版本 26.1.0） |
| pyannote.audio | 3.3.2 |
| lightning / pytorch-lightning | 2.5.5 |
| Transformers | 4.46.3 |

依据：[Ascend 官方配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)、[CANN 9.1.0 官方镜像源码](https://github.com/Ascend/cann-container-image/blob/main/cann/9.1.0-910b-ubuntu22.04-py3.10/Dockerfile)。pyannote 使用 3.3.2，保留原项目的模型格式；固定 Lightning 2.5.5 是为了兼容旧 pyannote checkpoint，避免 PyTorch 2.6 起 `weights_only` 默认行为变化导致加载失败。模型仍只从用户提供的本地目录读取。

当前目标服务器为 **910B3 / 驱动 25.5.1**。已核对 [CANN 9.1.0 官方发布说明](https://gitcode.com/cann/release-management/blob/master/9.1.0/release-notes.md)：兼容矩阵中 CANN 9.1.X 与 Ascend HDK 25.5.X 为 Y，因此可保留当前驱动。说明中的 26.0.RC1 是配套基线，兼容矩阵也支持 25.5.X。

基础 `npu-smi info` 未显示固件版本，部署前在宿主机检查：

```bash
npu-smi info -t board -i 0
```

查看 `Software Version`、`Firmware Version` 和 `Compatibility`；`Compatibility` 应为 `OK`，其他卡也应逐卡核对。命令与字段说明见 [官方版本查询指引](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0030.html)。容器自带新版 CANN，宿主机不必安装同版 toolkit；保持宿主机现有环境，通过下方驱动目录挂载使用。版本矩阵兼容仍需经过容器内的真实 NPU 推理验证。

```bash
docker build -t whisperx-imonitor .
```

同一个官方镜像也由 DockerHub 提供，可按服务器网络切换镜像地址，版本保持一致：

```bash
docker build \
  --build-arg CANN_IMAGE=ascendai/cann:9.1.0-910b-ubuntu22.04-py3.10 \
  -t whisperx-imonitor .
```

通用依赖和 torch-npu 使用华为 PyPI 源。两种 CPU 架构的 torch 都使用 `2.7.1+cpu` wheel，Docker 构建通过阿里云 PyTorch wheel 镜像 `https://mirrors.aliyun.com/pytorch-wheels/cpu/` 获取，避免安装 CUDA 版 PyTorch；torchaudio 的 x86_64 wheel 是 `2.7.1+cpu`，ARM64 wheel 是 `2.7.1`。CPU wheel 配合 torch-npu，Whisper 和声纹模型仍在 NPU 运行。不要仅替换基础镜像为其他 CANN 版本，而继续沿用这组固定的 PyTorch/TorchNPU 版本。

`entrypoint.sh` 会在启动应用前加载容器内的 `/usr/local/Ascend/ascend-toolkit/set_env.sh`，再执行 CMD。构建步骤会检查依赖、导入版本和 ffmpeg；仅在构建时的导入检查设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，避免无驱动挂载时自动加载 TorchNPU。运行应用时正常启用 NPU。

`models/` 不会打进镜像，运行时只读挂载；TSB 结果目录应独立持久化。以下为 910B 单卡示例，宿主机设备节点必须存在：

```bash
mkdir -p private-tsb-data

docker run -d --name whisperx-imonitor \
  --ipc=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/private-tsb-data:/app/private-tsb-data" \
  -p 8000:8000 \
  whisperx-imonitor

docker logs -f whisperx-imonitor
```

只挂载宿主机驱动，不用宿主机的整个 `/usr/local/Ascend` 覆盖容器目录，否则会把镜像内配套的 CANN 覆盖掉。其他昇腾型号需要使用对应芯片的镜像及设备配置。实际镜像构建、驱动与 NPU 推理需要在目标服务器验证。

## 接口

同一服务端口提供：

- `GET /health`：健康检查。
- `POST /transcribe`：`file` 上传普通音视频，支持 `diarize`、`min_speakers`、`max_speakers`、`speaker_embeddings`；也兼容原 CDR 的 `sample`、`label`、`sgmfile` PCM 表单。两种表单不要混用。
- `GET /tsb/alive`、`/tsb/sessionapply`、`/tsb/workerstart`、`/tsb/tsbrestart`。
- `POST /tsb/transcribe`、`/tsb/getresult`。

上传 WAV/MP4 示例：

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F 'file=@test.wav' \
  -F 'diarize=true' \
  -F 'speaker_embeddings=true'
```

TSB 流程：先调用 `/tsb/sessionapply`，使用返回的 `workerid`；依次以 `dir=0` 和 `dir=1` 上传同一个 `cdrid` 的 PCM，每路最后一片 `continue=0`；随后使用 `workerid` 和形如 `[12345]` 的 `cdrid` 表单轮询 `/tsb/getresult`。`combo` 输出转录和声纹，`vponly` 仅输出声纹。单个 worker 只绑定一个 `cdrid`。`completed=true` 代表至少一路有结果，需分别检查 `l` 和 `r`。

原 CDR 的批量测试脚本也可直接使用：

```bash
bash wav_to_pcm.sh /path/to/wav_directory
bash transcribe_pcm.sh /path/to/pcm_directory http://127.0.0.1:8000
```

TSB 每侧完成后结果保存 24 小时；在保留期内支持重复查询，包括 worker 释放后。SQLite 与 PCM 缓存位于 `--data-dir`，请挂载持久卷并限制目录权限。一个 worker 对应一个会话；6 是逻辑名额，底层推理串行。断开连接的未完成会话需要调用 workerstart 释放。当前重启接口只重置会话状态，不重建共享 NPU 模型；服务异常退出后未完成会话需要重传，遗留的临时音频需停服后清理。

## 测试与限制

```bash
python -m unittest discover -s tests -v
```

自动化测试使用模拟推理验证普通上传、CDR、TSB、声纹编码、时间偏移和结果持久化；**不验证真实音频的声纹准确率、两小时内存、NPU 算子及吞吐**。必须在目标昇腾机器上进行实机验证，检查转录效果、语音漏检、声纹兼容性、并发和 OOM 恢复。有效语音太短可能导致声纹失败，TSB 会返回致命错误并需重启 worker。模式、出错文案及历史结果查询与客户 PDF 的细节需联调确认。
