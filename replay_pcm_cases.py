#!/usr/bin/env python3
"""Replay one PCM case, or all cases in a directory, through the TSB API."""
import argparse
import datetime
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def fragments(directory):
    fmt_files = list(directory.glob('*.fmt'))
    if len(fmt_files) != 1:
        raise ValueError(f'{directory}: expected one .fmt file')
    fmt = json.loads(fmt_files[0].read_text(encoding='utf-8'))
    if (fmt.get('fmt'), fmt.get('channel'), fmt.get('bit'), fmt.get('rate')) != (1, 2, 16, 16000):
        raise ValueError(f'{directory}: expected stereo 16-bit 16 kHz PCM')
    prefix = fmt_files[0].stem
    result = {}
    for side, count_key in (('L', 'seqnol'), ('R', 'seqnor')):
        full_path = directory / f'{prefix}.{side}.pcm'
        whole = full_path.read_bytes() if full_path.is_file() else None
        if whole is None:
            print(f'WARNING: {full_path} missing; using approximate timestamps from filenames', file=sys.stderr, flush=True)
        pattern = re.compile(rf'{re.escape(prefix)}\.(\d+)\.(\d+)\.{side}\.pcm')
        files = sorted((int(m.group(1)), int(m.group(2)), p) for p in directory.glob(f'*.{side}.pcm')
                       if (m := pattern.fullmatch(p.name)))
        if [seq for seq, _, _ in files] != list(range(fmt[count_key])) or not files:
            raise ValueError(f'{directory}: invalid {side} fragment sequence')
        result[side] = []
        for seq, named_offset, path in files:
            data = path.read_bytes()
            if not data or len(data) % 2:
                raise ValueError(f'{path}: empty or invalid 16-bit PCM')
            if whole is None:
                timestamp = float(named_offset)
            else:
                offset = whole.find(data)
                if offset < 0 or offset % 2 or whole.find(data, offset + 2) >= 0:
                    raise ValueError(f'{path}: fragment has no unique aligned match in full channel')
                timestamp = offset / (fmt['rate'] * 2)
            result[side].append((path, seq, timestamp))
    return fmt['rate'], result


def request(url, fields=None, file=None):
    if fields is None:
        req = urllib.request.Request(url)
    else:
        boundary = uuid.uuid4().hex
        body = bytearray()
        for name, value in fields.items():
            body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        if file is not None:
            body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="sgmfile"; filename="audio.pcm"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
            body.extend(file.read_bytes())
            body.extend(b'\r\n')
        body.extend(f'--{boundary}--\r\n'.encode())
        req = urllib.request.Request(url, data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode("utf-8", errors="replace")}') from exc
    if result.get('code') != 200:
        raise RuntimeError(f'API error: {result}')
    return result


def replay(directory, base, timeout, output):
    rate, sides = fragments(directory)
    worker = request(base + '/tsb/sessionapply')['workerid']
    cdrid = time.time_ns() // 1000
    prefix = next(directory.glob('*.fmt')).stem
    metadata = {'workerid': worker, 'cdrid': cdrid,
                'timestamp_source': {side: 'full_channel' if (directory / f'{prefix}.{side}.pcm').is_file()
                                     else 'approximate_filename' for side in ('L', 'R')}}
    (output / f'{directory.name}.session.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'{directory.name}: worker={worker}, cdrid={cdrid}', flush=True)
    for direction, side in enumerate(('L', 'R')):
        parts = sides[side]
        for i, (path, seq, offset) in enumerate(parts):
            sample = {'fmt': 1, 'channel': 1, 'rate': rate, 'bit': 16, 'size': path.stat().st_size,
                      'cdrid': cdrid, 'dir': direction, 'seqno': seq,
                      'continue': int(i != len(parts) - 1), 'timestamp': offset}
            label = {'time': datetime.datetime.now().strftime('%Y%m%d%H%M%S'), 'start': offset}
            request(base + '/tsb/transcribe', {'workerid': worker, 'mode': 'combo',
                    'sample': json.dumps(sample), 'label': json.dumps(label)}, path)
            print(f'  uploaded {path.name}: {offset:.3f}s', flush=True)
    partial = output / f'{directory.name}.partial.json'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = request(base + '/tsb/getresult', {'workerid': worker, 'cdrid': json.dumps([cdrid])})
        partial.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if any('l' in item and 'r' in item for item in result.get('data', [])):
            partial.rename(output / f'{directory.name}.json')
            return
        time.sleep(2)
    raise TimeoutError(f'no complete result after {timeout}s; worker={worker} cdrid={cdrid}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('base_url', nargs='?', default='http://127.0.0.1:8000')
    parser.add_argument('--timeout', type=int, default=3600, help='result wait time in seconds per case')
    args = parser.parse_args()
    if not args.cases.is_dir() or args.timeout <= 0:
        parser.error('case directory must exist and timeout must be positive')
    if list(args.cases.glob('*.fmt')):
        directories = [args.cases]
        output_parent = args.cases.parent
    else:
        directories = sorted(path for path in args.cases.iterdir() if path.is_dir() and list(path.glob('*.fmt')))
        output_parent = args.cases
    if not directories:
        parser.error('no case directories with .fmt files')
    output = Path(tempfile.mkdtemp(prefix='replay-results-', dir=output_parent))
    print(f'Results: {output}', flush=True)
    for directory in directories:
        try:
            replay(directory, args.base_url.rstrip('/'), args.timeout, output)
            print(f'Saved: {output / (directory.name + ".json")}', flush=True)
        except Exception as exc:
            print(f'ERROR: {directory.name}: {exc}; session details are in {output}', file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
