import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import replay_pcm_cases as replay


class ReplayCasesTest(unittest.TestCase):
    def test_missing_full_channels_uses_filename_offsets_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / '008.fmt').write_text(json.dumps({'fmt': 1, 'channel': 2, 'bit': 16,
                                                       'rate': 16000, 'seqnol': 1, 'seqnor': 1}))
            (case / '008.0.9.L.pcm').write_bytes(b'\x01\x02' * 20)
            (case / '008.0.22.R.pcm').write_bytes(b'\x03\x04' * 20)
            warnings = io.StringIO()
            with redirect_stderr(warnings):
                rate, sides = replay.fragments(case)
            self.assertEqual(rate, 16000)
            self.assertEqual(sides['L'][0][2], 9.0)
            self.assertEqual(sides['R'][0][2], 22.0)
            self.assertIn('approximate timestamps from filenames', warnings.getvalue())

    def test_one_case_uploads_left_then_right_and_saves_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / '008_case'
            case.mkdir()
            (case / '008.fmt').write_text(json.dumps({'fmt': 1, 'channel': 2, 'bit': 16,
                                                       'rate': 16000, 'seqnol': 2, 'seqnor': 1}))
            left1, left2, right = b'\x01\x02' * 20, b'\x03\x04' * 20, b'\x05\x06' * 20
            (case / '008.L.pcm').write_bytes(b'\0\0' * 16000 + left1 + b'\0\0' * 8000 + left2)
            (case / '008.R.pcm').write_bytes(b'\0\0' * 8000 + right)
            for name, data in [('008.0.1.L.pcm', left1), ('008.1.1.L.pcm', left2), ('008.0.0.R.pcm', right)]:
                (case / name).write_bytes(data)
            submitted, polls = [], []

            def request(url, fields=None, file=None):
                if url.endswith('/sessionapply'):
                    return {'code': 200, 'workerid': '2'}
                if url.endswith('/transcribe'):
                    submitted.append((json.loads(fields['sample']), file.name))
                    return {'code': 200}
                polls.append(1)
                return {'code': 200, 'data': [{'l': {}, **({'r': {}} if len(polls) == 2 else {})}]}

            with patch.object(replay, 'request', side_effect=request), patch.object(replay.time, 'sleep'):
                replay.replay(case, 'http://test', 5, root)
            self.assertEqual([item[0]['dir'] for item in submitted], [0, 0, 1])
            self.assertEqual([item[0]['continue'] for item in submitted], [1, 0, 0])
            self.assertEqual([item[0]['timestamp'] for item in submitted], [1.0, 1.50125, 0.5])
            self.assertEqual(len(polls), 2)
            self.assertEqual(set(json.loads((root / '008_case.json').read_text())['data'][0]), {'l', 'r'})


if __name__ == '__main__':
    unittest.main()
