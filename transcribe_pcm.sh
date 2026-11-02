#!/usr/bin/env bash
# Recursively send raw PCM files to the CDR transcription endpoint.
set -u

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] || [ ! -d "$1" ]; then
  echo "Usage: bash $0 PCM_DIRECTORY [BASE_URL]" >&2
  exit 2
fi
for cmd in curl python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing command: $cmd" >&2
    exit 2
  fi
done
base_url="${2:-http://127.0.0.1:8000}"
base_url="${base_url%/}"
cdrid=$(python3 -c 'import time; print(time.time_ns() // 1000)') || exit 1
failed=0

while IFS= read -r -d '' pcm_file; do
  json_file="${pcm_file%.*}.json"
  if [ -e "$json_file" ]; then
    echo "Already exists, skipped: $json_file"
    continue
  fi
  pcm_bytes=$(wc -c < "$pcm_file" | tr -d '[:space:]')
  if [ "$pcm_bytes" -eq 0 ] || [ "$((pcm_bytes % 2))" -ne 0 ]; then
    echo "Empty or invalid 16-bit PCM file: $pcm_file" >&2
    failed=1
    continue
  fi
  case "$pcm_file" in *.R.pcm) dir=1 ;; *) dir=0 ;; esac
  cdrid=$((cdrid + 1))
  label_time=$(date '+%Y%m%d%H%M%S')
  tmp_file=$(mktemp "${json_file}.tmp.XXXXXX") || { failed=1; continue; }
  if ! status=$(curl -sS --output "$tmp_file" --write-out '%{http_code}' \
    -X POST "$base_url/transcribe" \
    -F "sample={\"fmt\":1,\"channel\":1,\"rate\":16000,\"bit\":16,\"size\":${pcm_bytes},\"cdrid\":${cdrid},\"dir\":${dir},\"seqno\":0,\"continue\":0};type=application/json" \
    -F "label={\"time\":\"${label_time}\",\"start\":0};type=application/json" \
    -F 'sgmfile=@-;filename=audio.pcm;type=application/octet-stream' < "$pcm_file"); then
    echo "Request failed: $pcm_file" >&2
    rm -f "$tmp_file"
    failed=1
    continue
  fi
  if ! python3 -m json.tool < "$tmp_file" >/dev/null 2>&1; then
    echo "Invalid JSON response (HTTP $status): $pcm_file" >&2
    rm -f "$tmp_file"
    failed=1
    continue
  fi
  if ! mv -n "$tmp_file" "$json_file" || [ -e "$tmp_file" ]; then
    echo "Output already exists or could not be created: $json_file" >&2
    rm -f "$tmp_file"
    failed=1
    continue
  fi
  echo "Saved: $json_file (HTTP $status)"
  case "$status" in 2??) ;; *) failed=1 ;; esac
done < <(find "$1" -type f -iname '*.pcm' -print0)

exit "$failed"
