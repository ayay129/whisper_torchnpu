import base64
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import create_app
from audio import decode_media, decode_pcm
from runtime import SAMPLE_RATE, WhisperRuntime, speech_chunks, use_cpu_fbank_on_npu
from tsb_api import TSBBackend


class FakeRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self.calls = 0

    def transcribe(self, audio, language=None):
        self.calls += 1
        return {'segments': [{'start': 0.25, 'end': 1.25, 'text': 'hello', 'language': 'en'}], 'language': 'en'}

    def speakers(self, audio, **kwargs):
        return [(0.0, len(audio) / SAMPLE_RATE, 'SPEAKER_00')], {'SPEAKER_00': np.array([1, 2], dtype='<f4')} if kwargs.get('embeddings') else {}


def sample(direction, cdrid=42):
    return {'fmt': 1, 'channel': 1, 'rate': 16000, 'bit': 16, 'size': 6 * SAMPLE_RATE * 2,
            'cdrid': cdrid, 'dir': direction, 'seqno': 0, 'continue': 0}


class RuntimeTest(unittest.TestCase):
    def test_tsb_rtf_is_audio_duration_divided_by_processing_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.pcm'
            path.write_bytes(b'\0\0' * (6 * SAMPLE_RATE))
            with patch('time.perf_counter', side_effect=[100.0, 102.0]):
                result = TSBBackend(FakeRuntime())([(path, SAMPLE_RATE, 0)], 'vponly')
            self.assertEqual(result['rtf'], 3.0)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required')
    def test_actual_media_decoding_and_8khz_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=1', str(path)], check=True)
            audio = decode_media(path)
            self.assertEqual(audio.shape, (SAMPLE_RATE,))
            self.assertGreater(float(np.max(np.abs(audio))), 0.01)
        self.assertEqual(decode_pcm(b'\0\0' * 8000, 8000).shape, (SAMPLE_RATE,))

    def test_missing_models_fail_without_network_or_gpu_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'Missing local models'):
                WhisperRuntime(directory)

    def test_silero_segments_keep_timestamps_and_language(self):
        runtime = WhisperRuntime.__new__(WhisperRuntime)
        runtime.lock = threading.RLock()
        runtime.vad = object()
        runtime.language = None
        runtime.language_codes = {'english': 'en'}
        runtime.device = 'npu:0'
        class Torch:
            class npu:
                @staticmethod
                def set_device(device): pass
            @staticmethod
            def from_numpy(value): return value
            @staticmethod
            def inference_mode(): return nullcontext()
        runtime.torch = Torch
        runtime.speech_timestamps = lambda *args, **kwargs: [
            {'start': SAMPLE_RATE, 'end': 2 * SAMPLE_RATE},
            {'start': 3 * SAMPLE_RATE, 'end': 4 * SAMPLE_RATE},
            {'start': 12 * SAMPLE_RATE, 'end': 13 * SAMPLE_RATE},
        ]
        runtime.asr = lambda audio, **kwargs: {'text': 'hello', 'chunks': [{'language': 'english'}]}
        result = runtime.transcribe(np.zeros(14 * SAMPLE_RATE, dtype=np.float32))
        self.assertEqual([(s['start'], s['end']) for s in result['segments']], [(1, 4), (12, 13)])
        self.assertEqual([s['language'] for s in result['segments']], ['en', 'en'])
        self.assertEqual(speech_chunks([]), [])

    def test_npu_fbank_uses_cpu_without_changing_cpu_behavior(self):
        calls = []
        class Tensor:
            def __init__(self, device): self.device = type('Device', (), {'type': device})()
            def cpu(self): calls.append('cpu'); return Tensor('cpu')
            def to(self, device): calls.append(('to', device.type)); return Tensor(device.type)
        class Model:
            def compute_fbank(self, tensor): calls.append(('fbank', tensor.device.type)); return tensor
        model = Model()
        use_cpu_fbank_on_npu(model)
        self.assertEqual(model.compute_fbank(Tensor('npu')).device.type, 'npu')
        self.assertEqual(calls, ['cpu', ('fbank', 'cpu'), ('to', 'npu')])
        calls.clear()
        self.assertEqual(model.compute_fbank(Tensor('cpu')).device.type, 'cpu')
        self.assertEqual(calls, [('fbank', 'cpu')])


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = FakeRuntime()
        self.client = TestClient(create_app(self.runtime, self.temp.name, workers=1))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_file_and_cdr_protocols(self):
        with patch('app.decode_media', return_value=np.zeros(SAMPLE_RATE, dtype=np.float32)):
            result = self.client.post('/transcribe', files={'file': ('test.wav', b'wave', 'audio/wav')},
                                      data={'diarize': 'true', 'speaker_embeddings': 'true'})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['segments'][0]['speaker'], 'SPEAKER_00')
        self.assertEqual(result.json()['segments'][0]['text'], 'SPEAKER_00 || hello')
        self.assertEqual(base64.b64decode(result.json()['speaker_embeddings']['SPEAKER_00']), np.array([1, 2], dtype='<f4').tobytes())
        pcm = b'\0\0' * (6 * SAMPLE_RATE)
        result = self.client.post('/transcribe', data={'sample': json.dumps(sample(0)), 'label': '{"time":"today","start":2}'}, files={'sgmfile': ('test.pcm', pcm)})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['segments'][0], {'text': 'hello', 'start': 2.25, 'end': 3.25})
        self.assertEqual(self.client.post('/transcribe', data={}).status_code, 400)

    def test_missing_speaker_returns_null(self):
        with patch.object(self.runtime, 'speakers', return_value=([], {})):
            with patch('app.decode_media', return_value=np.zeros(SAMPLE_RATE, dtype=np.float32)):
                response = self.client.post('/transcribe', files={'file': ('test.wav', b'wave', 'audio/wav')},
                                            data={'diarize': 'true', 'speaker_embeddings': 'true'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(response.json()['segments'][0]['speaker'])
            path = Path(self.temp.name) / 'test.pcm'
            path.write_bytes(b'\0\0' * (6 * SAMPLE_RATE))
            result = TSBBackend(self.runtime)([(path, SAMPLE_RATE, 0)], 'combo')
            self.assertIsNone(result['segments'][0]['speaker_id'])
            self.assertEqual(result['speaker_count'], 0)
            with self.assertRaisesRegex(ValueError, 'too short'):
                TSBBackend(self.runtime)([(path, SAMPLE_RATE, 0)], 'vponly')
        with patch.object(self.runtime, 'speakers', return_value=(
                [(1.5, 6.5, 'SPEAKER_00')], {'SPEAKER_00': np.array([1, 2], dtype='<f4')})):
            result = TSBBackend(self.runtime)([(path, SAMPLE_RATE, 0)], 'combo')
            self.assertIsNone(result['segments'][0]['speaker_id'])
            self.assertEqual(result['speaker_count'], 1)

    def test_rejected_fragment_logs_call_identifiers(self):
        worker = self.client.get('/tsb/sessionapply').json()['workerid']
        metadata = {**sample(1, 12345), 'seqno': 2, 'continue': 1}
        with self.assertLogs('tsb_api', level='WARNING') as captured:
            response = self.client.post('/tsb/transcribe', data={'workerid': worker,
                'sample': json.dumps(metadata), 'label': '{"time":"today","start":0}'},
                files={'sgmfile': ('test.pcm', b'\0\0' * (6 * SAMPLE_RATE))})
        self.assertEqual(response.status_code, 400)
        self.assertIn('workerid=0 cdrid=12345 dir=1 seqno=2 continue=1', captured.output[0])

    def test_tsb_status_endpoint(self):
        worker = self.client.get('/tsb/sessionapply').json()['workerid']
        metadata = {**sample(0, 1793776767146000), 'continue': 1}
        response = self.client.post('/tsb/transcribe', data={'workerid': worker,
            'sample': json.dumps(metadata), 'label': '{"time":"today","start":0}'},
            files={'sgmfile': ('test.pcm', b'\0\0' * (6 * SAMPLE_RATE))})
        self.assertEqual(response.status_code, 200)
        status = self.client.get('/tsb/status', params={'workerid': worker, 'cdrid': '1793776767146000'})
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json(), {'code': 200, 'workerid': worker, 'cdrid': 1793776767146000,
            'l': {'status': 'receiving', 'received_fragments': 1, 'next_seqno': 1},
            'r': {'status': 'not_received', 'received_fragments': 0}})
        self.assertEqual(self.client.get('/tsb/status', params={'workerid': worker, 'cdrid': 'abc'}).status_code, 400)
        self.assertEqual(self.client.get('/tsb/status', params={'workerid': worker, 'cdrid': str(2 ** 63)}).status_code, 400)

    def test_tsb_both_sides_and_repeated_result(self):
        for mode in ('combo', 'vponly'):
            with self.subTest(mode=mode):
                cdrid = 42 if mode == 'combo' else 43
                worker = self.client.get('/tsb/sessionapply').json()['workerid']
                before = self.runtime.calls
                for direction in (0, 1):
                    response = self.client.post('/tsb/transcribe', data={'workerid': worker, 'mode': mode,
                        'sample': json.dumps(sample(direction, cdrid)), 'label': '{"time":"today","start":0}'},
                        files={'sgmfile': ('test.pcm', b'\0\0' * (6 * SAMPLE_RATE))})
                    self.assertEqual(response.status_code, 200, response.text)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    result = self.client.post('/tsb/getresult', data={'workerid': worker, 'cdrid': f'[{cdrid}]'}).json()
                    if result.get('data') and 'r' in result['data'][0]:
                        break
                    time.sleep(.01)
                item = result['data'][0]
                self.assertEqual(set(item), {'cdrid', 'l', 'r'})
                self.assertEqual(item['l']['speaker_count'], 1)
                if mode == 'combo':
                    self.assertEqual(item['l']['segments'][0]['speaker_id'], 0)
                    self.assertEqual(item['l']['segments'][0]['language'], 'en')
                else:
                    self.assertEqual(item['l']['segments'], [])
                    self.assertEqual(self.runtime.calls, before)
                self.assertEqual(result, self.client.post('/tsb/getresult', data={'workerid': worker, 'cdrid': f'[{cdrid}]'}).json())

    def test_combo_returns_short_speech_and_missing_speaker_for_both_sides(self):
        vector = np.array([1, 2], dtype='<f4')
        with patch.object(self.runtime, 'speakers', side_effect=[
                ([(0.0, 4.0, 'SPEAKER_00')], {'SPEAKER_00': vector}),
                ([], {})]):
            worker = self.client.get('/tsb/sessionapply').json()['workerid']
            for direction in (0, 1):
                response = self.client.post('/tsb/transcribe', data={
                    'workerid': worker, 'sample': json.dumps(sample(direction, 46)),
                    'label': '{"time":"today","start":0}'},
                    files={'sgmfile': ('test.pcm', b'\0\0' * (6 * SAMPLE_RATE))})
                self.assertEqual(response.status_code, 200, response.text)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                response = self.client.post('/tsb/getresult', data={'workerid': worker, 'cdrid': '[46]'})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                if result.get('data') and 'r' in result['data'][0]:
                    break
                time.sleep(.01)
            item = result['data'][0]
            self.assertEqual(item['l']['segments'][0]['text'], 'hello')
            self.assertEqual(item['l']['segments'][0]['speaker_id'], 0)
            self.assertEqual(item['l']['speaker_count'], 1)
            self.assertEqual(base64.b64decode(item['l']['speakers'][0]['vpstr']), vector.tobytes())
            self.assertEqual(item['r']['segments'][0]['text'], 'hello')
            self.assertIsNone(item['r']['segments'][0]['speaker_id'])
            self.assertEqual(item['r']['speaker_count'], 0)
            self.assertEqual(item['r']['speakers'], [])

    def test_tsb_preserves_fragment_offsets(self):
        worker = self.client.get('/tsb/sessionapply').json()['workerid']
        pcm = b'\0\0' * (6 * SAMPLE_RATE)
        for direction, seqno, continued, timestamp in ((0, 0, 1, 0), (0, 1, 0, 10), (1, 0, 0, 0)):
            metadata = {**sample(direction, 44), 'seqno': seqno, 'continue': continued, 'timestamp': timestamp}
            response = self.client.post('/tsb/transcribe', data={'workerid': worker, 'sample': json.dumps(metadata),
                'label': '{"time":"today","start":0}'}, files={'sgmfile': ('test.pcm', pcm)})
            self.assertEqual(response.status_code, 200, response.text)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.client.post('/tsb/getresult', data={'workerid': worker, 'cdrid': '[44]'}).json()
            if result.get('data') and 'r' in result['data'][0]:
                break
            time.sleep(.01)
        self.assertEqual([s['start'] for s in result['data'][0]['l']['segments']], [0.25, 10.25])


if __name__ == '__main__':
    unittest.main()
