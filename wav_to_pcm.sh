#!/usr/bin/env bash
# Recursively convert WAV files into 16 kHz, signed 16-bit mono PCM.
set -u

if [ "$#" -ne 1 ] || [ ! -d "$1" ]; then
  echo "Usage: bash $0 WAV_DIRECTORY" >&2
  exit 2
fi
for cmd in ffmpeg ffprobe; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing command: $cmd" >&2
    exit 2
  fi
done

failed=0
convert_channel() {
  local wav_file="$1" pcm_file="$2" channel_filter="$3" tmp_file
  if [ -e "$pcm_file" ]; then
    echo "Already exists, skipped: $pcm_file"
    return 0
  fi
  tmp_file=$(mktemp "${pcm_file}.tmp.XXXXXX") || return 1
  if ffmpeg -nostdin -v error -y -i "$wav_file" -map 0:a:0 \
      -af "$channel_filter" -ar 16000 -acodec pcm_s16le -f s16le "$tmp_file"; then
    if mv -n "$tmp_file" "$pcm_file" && [ ! -e "$tmp_file" ]; then
      echo "Created: $pcm_file"
      return 0
    fi
    echo "Output already exists or could not be created: $pcm_file" >&2
  else
    echo "Conversion failed: $wav_file" >&2
  fi
  rm -f "$tmp_file"
  return 1
}

while IFS= read -r -d '' wav_file; do
  channels=$(ffprobe -v error -select_streams a:0 -show_entries stream=channels \
    -of default=noprint_wrappers=1:nokey=1 "$wav_file") || channels=''
  base="${wav_file%.*}"
  case "$channels" in
    1)
      echo "Mono: $wav_file"
      convert_channel "$wav_file" "$base.pcm" 'pan=mono|c0=c0' || failed=1
      ;;
    2)
      echo "Stereo: $wav_file"
      convert_channel "$wav_file" "$base.L.pcm" 'pan=mono|c0=c0' || failed=1
      convert_channel "$wav_file" "$base.R.pcm" 'pan=mono|c0=c1' || failed=1
      ;;
    *)
      echo "Unsupported or unreadable channel count ($channels): $wav_file" >&2
      failed=1
      ;;
  esac
done < <(find "$1" -type f -iname '*.wav' -print0)

exit "$failed"
