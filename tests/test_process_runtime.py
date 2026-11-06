import json
import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app import create_app
import app
import process_runtime


class FakeRuntime:
    def __init__(self, device_id):
        self.device_id = device_id

    def transcribe(self, audio, language=None):
        return {'pid': os.getpid(), 'segments': [{'start': 0.0, 'end': 1.0, 'text': 'hello'}], 'language': language or 'en'}

    def speakers(self, audio, **options):
        name = f'SPEAKER_{os.getpid()}'
        return [(0.0, 1.0, name)], {name: [1.0, 2.0]} if options.get('embeddings') else {}


class BarrierBackend:
    def __init__(self, directory, device_id):
        self.directory = Path(directory)
        self.device_id = device_id

    def __call__(self, parts, mode):
        (self.directory / str(os.getpid())).touch()
        deadline = time.monotonic() + 5
        while len(list(self.directory.iterdir())) < 2:
            if time.monotonic() >= deadline:
                raise TimeoutError('inference processes did not run concurrently')
            time.sleep(.01)
        return {'pid': os.getpid(), 'device_id': self.device_id, 'mode': mode, 'segments': []}


def fake_initialize(model_dir, device_id, language):
    process_runtime._runtime = FakeRuntime(device_id)
    process_runtime._backend = BarrierBackend(model_dir, device_id)


def fake_initialize_failure(model_dir, device_id, language):
    marker = Path(model_dir) / 'initialized'
    if marker.exists():
        raise RuntimeError('simulated model initialization failure')
    marker.write_text(str(os.getpid()))
    fake_initialize(model_dir, device_id, language)


class ProcessRuntimeTest(unittest.TestCase):
    def test_main_uses_per_device_counts_for_workers_and_backends(self):
        runtime = MagicMock()
        with patch.dict(os.environ, {'ASR_WORKER_DEVICES': '0:2,1:1'}), \
             patch.object(sys, 'argv', ['app.py', '--workers', '9']), \
             patch('app.ProcessRuntimes', return_value=runtime) as factory, \
             patch('app.create_app', return_value='application') as create, \
             patch('uvicorn.run') as run:
            app.main()
        self.assertEqual(factory.call_args.args[1], [0, 0, 1])
        self.assertEqual(create.call_args.args[2], 3)
        self.assertEqual(len(create.call_args.args[3]), 3)
        run.assert_called_once()
        runtime.close.assert_called_once()

    def test_parse_per_device_worker_counts(self):
        self.assertEqual(process_runtime.parse_worker_devices('0:2,1:3'), [0, 0, 1, 1, 1])
        for invalid in ('', '0:0', '0:1,0:2', '-1:2', '0:1,', '0:abc', '0:1, 1:2'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                process_runtime.parse_worker_devices(invalid)

    def test_spawned_processes_are_resident_bound_and_run_concurrently(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(process_runtime, '_initialize_worker', fake_initialize):
                runtimes = process_runtime.ProcessRuntimes(directory, [0, 1], None)
            try:
                self.assertEqual(len(set(runtimes.pids)), 2)
                self.assertNotIn(os.getpid(), runtimes.pids)
                self.assertEqual(runtimes.transcribe([1], 'en')['pid'], runtimes.pids[0])
                self.assertEqual(runtimes.speakers([1], embeddings=True)[0][0][2], f'SPEAKER_{runtimes.pids[0]}')
                with ThreadPoolExecutor(max_workers=2) as threads:
                    futures = [threads.submit(runtimes.tsb, i, [], 'combo') for i in range(2)]
                    results = [future.result(timeout=10) for future in futures]
                self.assertEqual([item['pid'] for item in results], runtimes.pids)
                self.assertEqual([item['device_id'] for item in results], [0, 1])
            finally:
                runtimes.close()

    def test_failed_initialization_stops_already_started_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(process_runtime, '_initialize_worker', fake_initialize_failure):
                with self.assertRaises(BrokenProcessPool):
                    process_runtime.ProcessRuntimes(directory, [0, 0], None)
            pid = int((Path(directory) / 'initialized').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_tsb_api_routes_each_worker_to_its_own_process(self):
        with tempfile.TemporaryDirectory() as directory:
            markers = Path(directory) / 'markers'
            markers.mkdir()
            with patch.object(process_runtime, '_initialize_worker', fake_initialize):
                runtimes = process_runtime.ProcessRuntimes(str(markers), [0, 0], None)
            try:
                backends = [partial(runtimes.tsb, i) for i in range(2)]
                with TestClient(create_app(runtimes, Path(directory) / 'data', workers=2, worker_backends=backends)) as client:
                    with patch('app.decode_media', return_value=[0.0] * 16000):
                        response = client.post('/transcribe', files={'file': ('test.wav', b'wave')},
                                               data={'diarize': 'true', 'speaker_embeddings': 'true'})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()['segments'][0]['speaker'], f'SPEAKER_{runtimes.pids[0]}')
                    metadata = {'fmt': 1, 'channel': 1, 'rate': 16000, 'bit': 16, 'size': 4,
                                'cdrid': 789, 'dir': 0, 'seqno': 0, 'continue': 0}
                    response = client.post('/transcribe', data={'sample': json.dumps(metadata),
                        'label': '{"time":"today","start":0}'}, files={'sgmfile': ('test.pcm', b'1234')})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()['segments'][0]['text'], 'hello')
                    for workerid, cdrid in ((0, 123), (1, 456)):
                        self.assertEqual(client.get('/tsb/sessionapply').json()['workerid'], str(workerid))
                        metadata = {'fmt': 1, 'channel': 1, 'rate': 16000, 'bit': 16, 'size': 4,
                                    'cdrid': cdrid, 'dir': 0, 'seqno': 0, 'continue': 0}
                        response = client.post('/tsb/transcribe', data={'workerid': str(workerid),
                            'sample': json.dumps(metadata), 'label': '{"time":"today","start":0}'},
                            files={'sgmfile': ('test.pcm', b'1234')})
                        self.assertEqual(response.status_code, 200, response.text)
                    for workerid, cdrid in ((0, 123), (1, 456)):
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            response = client.post('/tsb/getresult', data={'workerid': str(workerid), 'cdrid': f'[{cdrid}]'})
                            self.assertEqual(response.status_code, 200, response.text)
                            result = response.json()
                            if result['completed']:
                                break
                            time.sleep(.01)
                        self.assertTrue(result['completed'])
                        self.assertEqual(result['data'][0]['l']['pid'], runtimes.pids[workerid])
            finally:
                runtimes.close()


if __name__ == '__main__':
    unittest.main()
