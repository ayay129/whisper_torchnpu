"""Offline Silero + TorchNPU Whisper + pyannote inference."""
from __future__ import annotations

import base64
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def speech_chunks(timestamps, max_duration=10.0):
    """Group Silero speech intervals without losing their original offsets."""
    merged = []
    for item in timestamps:
        start, end = int(item['start']), int(item['end'])
        if end <= start:
            continue
        if merged and end - merged[-1][0] <= max_duration * SAMPLE_RATE:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def use_cpu_fbank_on_npu(model):
    original = model.compute_fbank

    def compute_fbank(waveforms):
        if waveforms.device.type in ('npu', 'privateuseone'):
            # Ascend complex64 abs is unsupported in some CANN versions.
            return original(waveforms.cpu()).to(waveforms.device)
        return original(waveforms)

    model.compute_fbank = compute_fbank


class WhisperRuntime:
    def __init__(self, model_dir='models', device_id=0, language=None):
        root = Path(model_dir).resolve()
        whisper = root / 'whisper-large-v3'
        silero = root / 'silero-vad-6.2.1'
        diarization = root / 'pyannote-diarization' / 'config.yaml'
        required = [whisper / 'config.json', whisper / 'preprocessor_config.json',
                    silero / 'hubconf.py', silero / 'src/silero_vad/data/silero_vad.jit',
                    diarization, root / 'pyannote-segmentation-3.0/pytorch_model.bin',
                    root / 'pyannote-wespeaker/pytorch_model.bin']
        missing = [str(path) for path in required if not path.is_file()]
        if not any((whisper / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json')):
            missing.append(str(whisper / '<Hugging Face model weights>'))
        if missing:
            raise FileNotFoundError('Missing local models: ' + ', '.join(missing))

        # The copied pyannote configuration contains project-relative model paths.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        import torch_npu  # noqa: F401
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
        from pyannote.audio import Pipeline
        from pyannote.audio.models.embedding.wespeaker import BaseWeSpeakerResNet
        import yaml

        if not torch.npu.is_available():
            raise RuntimeError('TorchNPU is installed but no Ascend NPU is available')
        if not 0 <= device_id < torch.npu.device_count():
            raise RuntimeError(f'NPU device {device_id} is not visible; visible device count: {torch.npu.device_count()}')
        self.torch = torch
        self.device = torch.device(f'npu:{device_id}')
        torch.npu.set_device(self.device)
        self.lock = threading.RLock()
        self.language = language
        self.language_codes = TO_LANGUAGE_CODE
        self.vad, utils = torch.hub.load(str(silero), 'silero_vad', source='local', force_reload=False, onnx=False)
        self.speech_timestamps = utils[0]
        processor = AutoProcessor.from_pretrained(str(whisper), local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(str(whisper), torch_dtype=torch.float16, attn_implementation='eager', local_files_only=True)
        model.eval().to(self.device)
        self.asr = pipeline('automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
                            feature_extractor=processor.feature_extractor, device=self.device, torch_dtype=torch.float16)
        with diarization.open(encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
        params = config['pipeline']['params']
        params['segmentation'] = str(root / 'pyannote-segmentation-3.0/pytorch_model.bin')
        params['embedding'] = str(root / 'pyannote-wespeaker/pytorch_model.bin')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', encoding='utf-8') as local_config:
            yaml.safe_dump(config, local_config)
            local_config.flush()
            self.diarization = Pipeline.from_pretrained(local_config.name).to(self.device)
        embedding_model = getattr(getattr(self.diarization, '_embedding', None), 'model_', None)
        if isinstance(embedding_model, BaseWeSpeakerResNet):
            use_cpu_fbank_on_npu(embedding_model)

    def transcribe(self, audio, language=None):
        with self.lock:
            self.torch.npu.set_device(self.device)
            timestamps = self.speech_timestamps(self.torch.from_numpy(audio), self.vad, sampling_rate=SAMPLE_RATE,
                                                threshold=0.7, max_speech_duration_s=10, return_seconds=False)
            segments = []
            for start, end in speech_chunks(timestamps):
                used_language = language or self.language
                kwargs = {'return_timestamps': True, 'generate_kwargs': {'task': 'transcribe'}}
                if used_language:
                    kwargs['generate_kwargs']['language'] = used_language
                else:
                    kwargs['return_language'] = True
                with self.torch.inference_mode():
                    result = self.asr(audio[start:end], **kwargs)
                text = str(result.get('text', '')).strip()
                if not text:
                    continue
                detected = used_language or result.get('language')
                if not detected:
                    detected = next((chunk.get('language') for chunk in result.get('chunks', []) if chunk.get('language')), None)
                if detected:
                    detected = str(detected).lower().removeprefix('<|').removesuffix('|>')
                    detected = self.language_codes.get(detected, detected)
                segments.append({'start': round(start / SAMPLE_RATE, 3), 'end': round(end / SAMPLE_RATE, 3),
                                 'text': text, 'language': detected})
            return {'segments': segments, 'language': language or self.language or next((s['language'] for s in segments if s['language']), None)}

    def speakers(self, audio, *, embeddings=True, min_speakers=None, max_speakers=None):
        with self.lock:
            self.torch.npu.set_device(self.device)
            output = self.diarization({'waveform': self.torch.from_numpy(audio[None, :]), 'sample_rate': SAMPLE_RATE},
                                      min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=embeddings)
        annotation, vectors = output if embeddings else (output, None)
        labels = annotation.labels()
        turns = [(float(segment.start), float(segment.end), speaker) for segment, _, speaker in annotation.itertracks(yield_label=True)]
        if vectors is not None and (len(vectors) != len(labels) or not np.isfinite(vectors).all()):
            raise ValueError('Invalid speaker embeddings')
        return turns, {name: np.asarray(vector, dtype='<f4') for name, vector in zip(labels, vectors)} if vectors is not None else {}


def speaker_for_segment(segment, turns):
    overlaps = {}
    for start, end, speaker in turns:
        overlap = max(0.0, min(segment['end'], end) - max(segment['start'], start))
        overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
    if overlaps and max(overlaps.values()) > 0:
        return max(overlaps, key=overlaps.get)
    return None


def encode_vector(vector):
    return base64.b64encode(np.asarray(vector, dtype='<f4').tobytes()).decode('ascii')
