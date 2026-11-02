"""TSB session management and durable results, independent of the inference backend."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


class TSBError(Exception):
    def __init__(self, message, code=400):
        self.message, self.code = message, code
        super().__init__(message)


class TSBService:
    def __init__(self, backend, directory, workers=6, capacity=1_152_000_000, retention=86400, wait_seconds=30):
        self.backend = backend
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.root / 'audio'
        self.audio_dir.mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / 'results.sqlite3', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS results (worker INTEGER, cdrid INTEGER, direction INTEGER, result TEXT, expires REAL, PRIMARY KEY(worker,cdrid,direction))')
        self.db.commit()
        self.capacity, self.retention, self.wait_seconds = capacity, retention, wait_seconds
        self.condition = threading.Condition(threading.RLock())
        self.priority_waiters = 0
        self.workers = [self._empty(0) for _ in range(workers)]
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tsb-inference')

    @staticmethod
    def _empty(generation):
        return {'generation': generation, 'applied': False, 'cdrid': None, 'sides': {}, 'error': None, 'bytes': 0, 'mode': None}

    def _cleanup_results(self):
        self.db.execute('DELETE FROM results WHERE expires <= ?', (time.time(),))
        self.db.commit()

    def apply(self, priority=0):
        deadline = time.monotonic() + (self.wait_seconds if priority > 0 else 0)
        with self.condition:
            if priority > 0:
                self.priority_waiters += 1
            try:
                while True:
                    if priority > 0 or not self.priority_waiters:
                        for i, worker in enumerate(self.workers):
                            if not worker['applied']:
                                worker['applied'] = True
                                return i
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TSBError('Resources is busy', 429)
                    self.condition.wait(remaining)
            finally:
                if priority > 0:
                    self.priority_waiters -= 1
                    self.condition.notify_all()

    def _worker(self, workerid):
        if not isinstance(workerid, int) or not 0 <= workerid < len(self.workers):
            raise TSBError('workerid is required')
        worker = self.workers[workerid]
        if not worker['applied']:
            raise TSBError('workerid is not applied')
        if worker['error']:
            raise TSBError(worker['error'])
        return worker

    def restart(self, workerid):
        with self.condition:
            if not isinstance(workerid, int) or not 0 <= workerid < len(self.workers):
                raise TSBError('workerid is required')
            old = self.workers[workerid]
            self.workers[workerid] = self._empty(old['generation'] + 1)
            for side in old['sides'].values():
                shutil.rmtree(side['directory'], ignore_errors=True)
            self.condition.notify_all()

    def submit(self, workerid, sample, label, payload, mode):
        if mode not in ('combo', 'vponly'):
            raise TSBError('mode must be combo or vponly')
        if not payload:
            raise TSBError('Audio file is empty')
        if len(payload) % 2:
            raise TSBError('PCM byte length must be even for 16-bit audio')
        if len(payload) != sample['size']:
            raise TSBError('sample.size does not match audio size')
        with self.condition:
            worker = self._worker(workerid)
            cdrid, direction = sample['cdrid'], sample['dir']
            if worker['cdrid'] is not None and worker['cdrid'] != cdrid:
                raise TSBError('Workerid is applied')
            if worker['mode'] is not None and worker['mode'] != mode:
                raise TSBError('mode must be consistent for a session')
            self._cleanup_results()
            if worker['cdrid'] is None and self.db.execute('SELECT 1 FROM results WHERE worker=? AND cdrid=?', (workerid, cdrid)).fetchone():
                raise TSBError('cdrid already exists for this worker')
            side = worker['sides'].get(direction)
            if side and (side['finished'] or sample['seqno'] != len(side['parts'])):
                raise TSBError('sample.seqno must be consecutive')
            if not side and sample['seqno'] != 0:
                raise TSBError('sample.seqno must be consecutive')
            if worker['bytes'] + len(payload) > self.capacity:
                raise TSBError('Resources is busy')
            if side is None:
                side = {'directory': tempfile.mkdtemp(prefix='tsb-', dir=self.audio_dir), 'parts': [], 'finished': False, 'done': False}
                worker['sides'][direction] = side
            path = Path(side['directory']) / str(sample['seqno'])
            path.write_bytes(payload)
            side['parts'].append((str(path), sample['rate'], sample.get('timestamp', label['start'])))
            worker['cdrid'], worker['mode'] = cdrid, mode
            worker['bytes'] += len(payload)
            if sample['continue'] == 0:
                side['finished'] = True
                self.executor.submit(self._run, workerid, worker['generation'], cdrid, direction, list(side['parts']), mode)
            block = len(worker['sides']) == 2 and all(s['finished'] for s in worker['sides'].values()) and worker['bytes'] > self.capacity / 2
            return block

    def _run(self, workerid, generation, cdrid, direction, parts, mode):
        try:
            with self.condition:
                worker = self.workers[workerid]
                if worker['generation'] != generation or not worker['applied'] or worker['error']:
                    return
            result = self.backend(parts, mode)
        except Exception as exc:
            logger.exception('TSB inference failed: workerid=%s cdrid=%s dir=%s', workerid, cdrid, direction)
            msg = str(exc)
            if 'out of memory' in msg.lower():
                msg = 'GPU memory is insufficient'
            elif 'effective audio duration is too short' in msg.lower():
                msg = 'Effective speech duration is less than 5 seconds'
            else:
                msg = 'Inference failed'
            with self.condition:
                worker = self.workers[workerid]
                if worker['generation'] == generation and worker['applied']:
                    worker['error'] = msg
            return
        with self.condition:
            worker = self.workers[workerid]
            if worker['generation'] != generation or not worker['applied'] or worker['error']:
                return
            try:
                self.db.execute('INSERT OR REPLACE INTO results VALUES (?,?,?,?,?)', (workerid, cdrid, direction, json.dumps(result, ensure_ascii=False), time.time() + self.retention))
                self.db.commit()
            except Exception:
                logger.exception('Unable to save TSB result')
                self.db.rollback()
                worker['error'] = 'Unable to save result'
                return
            side = worker['sides'][direction]
            side['done'] = True
            shutil.rmtree(side['directory'], ignore_errors=True)
            if len(worker['sides']) == 2 and all(s['done'] for s in worker['sides'].values()):
                self.workers[workerid] = self._empty(generation + 1)
                self.condition.notify_all()

    def get(self, workerid, cdrids):
        with self.condition:
            if not isinstance(workerid, int) or not 0 <= workerid < len(self.workers):
                raise TSBError('workerid is required')
            worker = self.workers[workerid]
            if worker['applied'] and worker['error'] and worker['cdrid'] in cdrids:
                raise TSBError(worker['error'])
            self._cleanup_results()
            data = []
            for cdrid in cdrids:
                rows = self.db.execute('SELECT direction,result FROM results WHERE worker=? AND cdrid=?', (workerid, cdrid)).fetchall()
                if rows:
                    item = {'cdrid': cdrid}
                    item.update(('l' if direction == 0 else 'r', json.loads(result)) for direction, result in rows)
                    data.append(item)
            if not data and (not worker['applied'] or worker['cdrid'] not in (None, *cdrids)):
                raise TSBError('workerid is not applied')
            return {'code': 200, 'msg': 'Success', 'completed': bool(data), **({'data': data} if data else {})}

    def close(self):
        self.executor.shutdown(wait=True)
        self.db.close()
