"""Decode local media and the CDR raw PCM input format."""
import subprocess
from pathlib import Path

import numpy as np


def decode_pcm(payload, rate):
    if not payload:
        raise ValueError('Audio file is empty')
    if len(payload) % 2:
        raise ValueError('PCM byte length must be even for 16-bit audio')
    if rate not in (8000, 16000):
        raise ValueError('Only 8000 or 16000 sample rate is supported')
    audio = np.frombuffer(payload, dtype='<i2').astype(np.float32) / 32768.0
    if rate == 8000:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, 2, 1).astype(np.float32)
    return audio


def decode_media(path):
    command = ['ffmpeg', '-nostdin', '-threads', '1', '-i', str(Path(path)), '-f', 's16le',
               '-ac', '1', '-acodec', 'pcm_s16le', '-ar', '16000', '-']
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise ValueError('ffmpeg is not installed') from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError('Unable to decode audio or video') from exc
    return decode_pcm(result.stdout, 16000)
