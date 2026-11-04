import tempfile
import threading
import time
import unittest

from tsb_core import TSBError, TSBService


def sample(direction, seq=0, continued=0, cdrid=123):
    return {'cdrid': cdrid, 'dir': direction, 'seqno': seq, 'continue': continued, 'rate': 16000, 'size': 4}


class TSBServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = TSBService(lambda parts, mode: {'segments': [], 'speaker_count': 1, 'speakers': [{'speaker_id': 0, 'vpstr': 'abc'}]}, self.temp.name, workers=1, capacity=16, retention=60, wait_seconds=1)

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def wait_until(self, fn):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if fn():
                return
            time.sleep(.01)
        self.fail('timed out')

    def test_session_persists_both_sides_and_results_can_be_retrieved_again(self):
        self.assertEqual(self.service.apply(), 0)
        self.service.submit(0, sample(0), {'start': 0}, b'1234', 'combo')
        self.service.submit(0, sample(1), {'start': 0}, b'1234', 'combo')
        self.wait_until(lambda: not self.service.workers[0]['applied'])
        result = self.service.get(0, [123])
        self.assertEqual(set(result['data'][0]), {'cdrid', 'l', 'r'})
        self.assertEqual(result, self.service.get(0, [123]))
        self.service.close()
        self.service = TSBService(lambda *a: None, self.temp.name, workers=1)
        self.assertEqual(result, self.service.get(0, [123]))
        self.service.db.execute('UPDATE results SET expires = 0')
        self.service.db.commit()
        with self.assertRaisesRegex(TSBError, 'workerid is not applied'):
            self.service.get(0, [123])

    def test_worker_exclusive_fragments_and_capacity(self):
        self.service.apply()
        with self.assertRaises(TSBError) as busy:
            self.service.apply()
        self.assertEqual(busy.exception.code, 429)
        self.service.submit(0, sample(0, continued=1), {'start': 0}, b'1234', 'combo')
        with self.assertRaisesRegex(TSBError, 'Workerid is applied'):
            self.service.submit(0, sample(1, cdrid=456), {'start': 0}, b'1234', 'combo')
        with self.assertRaisesRegex(TSBError, 'consecutive'):
            self.service.submit(0, sample(0, seq=2), {'start': 0}, b'1234', 'combo')
        with self.assertRaisesRegex(TSBError, 'size does not match'):
            self.service.submit(0, sample(0, seq=1), {'start': 0}, b'12', 'combo')

    def test_restart_discards_in_flight_result(self):
        started, release = threading.Event(), threading.Event()
        def backend(parts, mode):
            started.set()
            release.wait(2)
            return {'segments': []}
        self.service.backend = backend
        self.service.apply()
        self.service.submit(0, sample(0), {'start': 0}, b'1234', 'vponly')
        self.assertTrue(started.wait(1))
        self.service.restart(0)
        release.set()
        self.service.close()
        self.service = TSBService(lambda *a: None, self.temp.name, workers=1)
        with self.assertRaisesRegex(TSBError, 'workerid is not applied'):
            self.service.get(0, [123])

    def test_inference_error_is_fatal_until_restart(self):
        def fail(*args):
            raise RuntimeError('out of memory')
        self.service.backend = fail
        self.service.apply()
        self.service.submit(0, sample(0), {'start': 0}, b'1234', 'combo')
        self.wait_until(lambda: self.service.workers[0]['error'] is not None)
        with self.assertRaisesRegex(TSBError, 'GPU memory is insufficient'):
            self.service.get(0, [123])
        self.service.restart(0)
        self.assertEqual(self.service.apply(), 0)

    def test_trace_shows_missing_side_and_completed_session(self):
        with self.assertLogs('tsb_core', level='INFO') as captured:
            self.service.apply()
            self.service.submit(0, sample(0), {'start': 0}, b'1234', 'combo')
            self.wait_until(lambda: self.service.get(0, [123]).get('completed'))
            self.service.submit(0, sample(1), {'start': 0}, b'1234', 'combo')
            self.wait_until(lambda: not self.service.workers[0]['applied'])
            self.service.get(0, [123])
        output = '\n'.join(captured.output)
        self.assertIn('fragment_received workerid=0 cdrid=123 generation=0 dir=0 seqno=0 continue=0 size=4', output)
        self.assertIn('queued workerid=0 cdrid=123 generation=0 dir=1 fragments=1 size=4', output)
        self.assertIn('processing_started workerid=0 cdrid=123 generation=0 dir=1', output)
        self.assertIn('result_saved workerid=0 cdrid=123 generation=0 dir=1', output)
        self.assertIn('result_queried workerid=0 cdrid=123 returned=l', output)
        self.assertIn('worker_released workerid=0 cdrid=123', output)

    def test_priority_waiter_receives_released_worker(self):
        self.service.apply()
        allocated = []
        waiter = threading.Thread(target=lambda: allocated.append(self.service.apply(priority=1)))
        waiter.start()
        self.wait_until(lambda: self.service.priority_waiters == 1)
        self.service.restart(0)
        waiter.join(2)
        self.assertEqual(allocated, [0])


if __name__ == '__main__':
    unittest.main()
