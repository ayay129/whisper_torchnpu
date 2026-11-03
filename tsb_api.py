#!/usr/bin/env python3
"""TSB adapter for the MindIE Whisper runtime; run as a single server process."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from tsb_core import TSBError, TSBService


def parse_object(name, raw):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        raise TSBError(f'{name} must be valid JSON')
    if not isinstance(value, dict):
        raise TSBError(f'{name} must be a JSON object')
    return value


def parse_input(raw_sample, raw_label):
    sample, label = parse_object('sample', raw_sample), parse_object('label', raw_label)
    for key in ('fmt', 'channel', 'rate', 'bit', 'size', 'cdrid', 'dir', 'seqno', 'continue'):
        if key not in sample:
            raise TSBError(f'sample.{key} is required')
        if type(sample[key]) is not int:
            raise TSBError(f'sample.{key} must be an integer')
    for key, allowed in {'fmt': (1,), 'channel': (1,), 'rate': (8000, 16000), 'bit': (16,), 'dir': (0, 1), 'continue': (0, 1)}.items():
        if sample[key] not in allowed:
            raise TSBError(f'sample.{key} is invalid')
    if sample['size'] <= 0 or sample['seqno'] < 0:
        raise TSBError('sample.size or sample.seqno is invalid')
    if not -(2 ** 63) <= sample['cdrid'] < 2 ** 63:
        raise TSBError('sample.cdrid is invalid')
    for key in ('time', 'start'):
        if key not in label:
            raise TSBError(f'label.{key} is required')
    if type(label['start']) not in (int, float) or not math.isfinite(label['start']):
        raise TSBError('label.start must be a number')
    if 'timestamp' in sample and (type(sample['timestamp']) not in (int, float) or not math.isfinite(sample['timestamp']) or sample['timestamp'] < 0):
        raise TSBError('sample.timestamp must be a number')
    return sample, label


def worker_number(raw):
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise TSBError('workerid is required')


def create_app(service, cdr_app=None):
    app = FastAPI(title='MindIE WhisperX TSB')

    @app.exception_handler(TSBError)
    async def tsb_error(_request, exc):
        return JSONResponse(status_code=exc.code, content={'code': exc.code, 'msg': exc.message})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        if not request.url.path.startswith('/tsb/'):
            return await request_validation_exception_handler(request, exc)
        return JSONResponse(status_code=400, content={'code': 400, 'msg': 'Invalid request'})

    @app.get('/tsb/alive')
    def alive():
        return {'code': 200, 'msg': 'Success', 'workercount': len(service.workers), 'mode': 'combo'}

    @app.get('/tsb/tsbrestart')
    def restart_all():
        for workerid in range(len(service.workers)):
            service.restart(workerid)
        return {'code': 200, 'msg': 'Success'}

    @app.get('/tsb/workerstart')
    def restart(workerid: str = ''):
        service.restart(worker_number(workerid))
        return {'code': 200, 'msg': 'Success'}

    @app.get('/tsb/sessionapply')
    def apply(priority: int = 0):
        return {'code': 200, 'msg': 'Success', 'workerid': str(service.apply(priority))}

    @app.post('/tsb/transcribe')
    async def transcribe(workerid: str = Form(''), sample: str = Form(''), label: str = Form(''), sgmfile: UploadFile = File(...), mode: str = Form('combo')):
        meta, label_meta = parse_input(sample, label)
        if meta['size'] > service.capacity:
            raise TSBError('Resources is busy')
        try:
            payload = await sgmfile.read(meta['size'] + 1)
        finally:
            await sgmfile.close()
        block = await run_in_threadpool(service.submit, worker_number(workerid), meta, label_meta, payload, mode)
        return {'code': 200, 'msg': 'Success', 'block': block}

    @app.post('/tsb/getresult')
    def getresult(workerid: str = Form(''), cdrid: str = Form('')):
        try:
            ids = json.loads(cdrid)
        except ValueError:
            raise TSBError('cdrid must be a JSON array')
        if not isinstance(ids, list) or not ids or any(type(i) is not int or not -(2 ** 63) <= i < 2 ** 63 for i in ids):
            raise TSBError('cdrid must be a JSON array of integers')
        return service.get(worker_number(workerid), ids)

    if cdr_app is not None:
        app.include_router(cdr_app.router)
    return app



class TSBBackend:
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, parts, mode):
        import time
        from audio import decode_pcm
        from runtime import encode_vector, speaker_for_segment

        started = time.perf_counter()
        waveforms, ranges, cursor = [], [], 0.0
        for path, rate, offset in parts:
            with open(path, 'rb') as pcm:
                audio = decode_pcm(pcm.read(), rate)
            end = cursor + len(audio) / 16000
            waveforms.append(audio)
            ranges.append((cursor, end, offset))
            cursor = end
        joined = np.concatenate(waveforms)
        with self.runtime.lock:
            turns, vectors = self.runtime.speakers(joined, embeddings=True)
            if (not vectors and mode == 'vponly') or (vectors and sum(end - start for start, end, _ in turns) < 5):
                raise ValueError('Effective audio duration is too short')
            speaker_ids = {name: i for i, name in enumerate(vectors)}
            speakers = [{'speaker_id': speaker_ids[name], 'vpstr': encode_vector(vector)} for name, vector in vectors.items()]
            segments = []
            if mode == 'combo':
                for audio, (start, _, offset) in zip(waveforms, ranges):
                    result = self.runtime.transcribe(audio)
                    for segment in result['segments']:
                        lookup = {**segment, 'start': segment['start'] + start, 'end': segment['end'] + start}
                        speaker = speaker_for_segment(lookup, turns)
                        segments.append({**segment, 'start': round(offset + segment['start'], 3),
                                         'end': round(offset + segment['end'], 3), 'speaker_id': speaker_ids.get(speaker)})
        return {'rtf': (time.perf_counter() - started) / cursor if cursor else 0,
                'segments': segments, 'speaker_count': len(speakers), 'speakers': speakers}
