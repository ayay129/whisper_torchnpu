#!/usr/bin/env python3
"""Independent WhisperX-style TorchNPU service with file, CDR and TSB APIs."""
from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import File, Form, HTTPException, UploadFile

from audio import decode_media, decode_pcm
from runtime import WhisperRuntime, encode_vector, speaker_for_segment
from tsb_api import TSBBackend, create_app as create_tsb_app, parse_input
from tsb_core import TSBError, TSBService

logger = logging.getLogger(__name__)


def create_app(runtime, data_dir, workers=6):
    service = TSBService(TSBBackend(runtime), data_dir, workers=workers)
    app = create_tsb_app(service)

    @app.on_event('shutdown')
    def shutdown():
        service.close()

    @app.get('/health')
    def health():
        return {'status': 'ok', 'backend': 'torchnpu-whisperx-imonitor'}

    @app.post('/transcribe')
    def transcribe(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                   label: Optional[str] = Form(None), sgmfile: Optional[UploadFile] = File(None),
                   diarize: bool = Form(False), min_speakers: Optional[int] = Form(None),
                   max_speakers: Optional[int] = Form(None), speaker_embeddings: bool = Form(False)):
        if file is not None and any(value is not None for value in (sample, label, sgmfile)):
            raise HTTPException(400, 'Use either file or CDR fields')
        if file is not None:
            if speaker_embeddings and not diarize:
                raise HTTPException(400, 'speaker_embeddings requires diarize=true.')
            suffix = Path(file.filename or '').suffix
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
                    shutil.copyfileobj(file.file, temp)
                    temp.flush()
                    if temp.tell() == 0:
                        raise HTTPException(400, 'Uploaded file is empty.')
                    audio = decode_media(temp.name)
                with runtime.lock:
                    result = runtime.transcribe(audio)
                    if diarize:
                        turns, vectors = runtime.speakers(audio, embeddings=speaker_embeddings,
                                                          min_speakers=min_speakers, max_speakers=max_speakers)
                        if speaker_embeddings:
                            result['speaker_embeddings'] = {name: encode_vector(vector) for name, vector in vectors.items()}
                        for segment in result['segments']:
                            speaker = speaker_for_segment(segment, turns)
                            if speaker_embeddings and speaker is not None:
                                segment['speaker'] = speaker
                            segment['text'] = f"{speaker or ''} || {segment['text'].strip()}"
                return result
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                logger.exception('File transcription failed')
                raise HTTPException(500, f'{type(exc).__name__}: {exc}') from exc
            finally:
                file.file.close()
        if sample is None or label is None or sgmfile is None:
            raise HTTPException(400, 'file or sample, label and sgmfile are required')
        try:
            meta, label_meta = parse_input(sample, label)
            payload = sgmfile.file.read()
            audio = decode_pcm(payload, meta['rate'])
            result = runtime.transcribe(audio)
            offset = float(label_meta['start'])
            response = {'sample': {key: meta[key] for key in ('cdrid', 'dir', 'seqno', 'continue')},
                        'label': {'time': label_meta['time'], 'start': label_meta['start']}}
            if result['segments']:
                response['segments'] = [{'text': item['text'], 'start': round(offset + item['start'], 3),
                                         'end': round(offset + item['end'], 3)} for item in result['segments']]
            if result['language']:
                response['language'] = result['language']
            return response
        except TSBError as exc:
            raise HTTPException(exc.code, exc.message) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception('CDR transcription failed')
            raise HTTPException(500, f'{type(exc).__name__}: {exc}') from exc
        finally:
            sgmfile.file.close()

    return app


def main():
    parser = argparse.ArgumentParser(description='Offline WhisperX-style TorchNPU and TSB service')
    parser.add_argument('--models-dir', default='models', help='Local models directory')
    parser.add_argument('--data-dir', default='private-tsb-data', help='TSB audio and SQLite directory')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--language', default=None, help='Optional Whisper language code')
    parser.add_argument('--workers', type=int, default=6, help='Logical TSB worker count')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    logging.basicConfig(level=logging.INFO)
    runtime = WhisperRuntime(args.models_dir, args.device_id, args.language)
    import uvicorn
    uvicorn.run(create_app(runtime, args.data_dir, args.workers), host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
