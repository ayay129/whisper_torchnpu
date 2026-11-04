"""One spawned, model-owning inference process per TSB worker."""
import logging
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)
_runtime = None
_backend = None


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
    def __init__(self, model_dir, device_id, language, workers):
        self.lock = threading.RLock()
        self.executors = []
        self.pids = []
        context = multiprocessing.get_context('spawn')
        try:
            for workerid in range(workers):
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
