"""One spawned, model-owning inference process per TSB worker."""
import logging
import multiprocessing
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)
_runtime = None
_backend = None


def parse_worker_devices(value):
    devices = []
    seen = set()
    for item in value.split(','):
        if not re.fullmatch(r'\d+:\d+', item):
            raise ValueError('worker devices must use DEVICE:COUNT pairs, e.g. 0:10,1:8')
        device, count = map(int, item.split(':'))
        if device in seen or count < 1:
            raise ValueError('worker device IDs must be unique and counts must be positive')
        seen.add(device)
        devices.extend([device] * count)
    return devices


def _initialize_worker(model_dir, device_id, language):
    global _runtime, _backend
    from runtime import WhisperRuntime
    from tsb_api import TSBBackend
    _runtime = WhisperRuntime(model_dir, device_id, language)
    _backend = TSBBackend(_runtime)


def _ready():
    return os.getpid()


def _transcribe(audio, language):
    return _runtime.transcribe(audio, language)


def _speakers(audio, options):
    return _runtime.speakers(audio, **options)


def _tsb(parts, mode):
    return _backend(parts, mode)


class ProcessRuntimes:
    def __init__(self, model_dir, device_ids, language):
        self.lock = threading.RLock()
        self.executors = []
        self.pids = []
        context = multiprocessing.get_context('spawn')
        try:
            for workerid, device_id in enumerate(device_ids):
                executor = ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_initialize_worker,
                                               initargs=(model_dir, device_id, language))
                self.executors.append(executor)
                pid = executor.submit(_ready).result()
                self.pids.append(pid)
                logger.info('Inference process ready workerid=%s pid=%s device=%s', workerid, pid, device_id)
        except BaseException:
            self.close()
            raise

    def transcribe(self, audio, language=None):
        return self.executors[0].submit(_transcribe, audio, language).result()

    def speakers(self, audio, **options):
        return self.executors[0].submit(_speakers, audio, options).result()

    def tsb(self, workerid, parts, mode):
        return self.executors[workerid].submit(_tsb, parts, mode).result()

    def close(self):
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)
