#!/usr/bin/env bash
set -euo pipefail

part=$1
range_start=$2
range_end=$3
url='https://drive.usercontent.google.com/download?id=18Fd3vGvj0Dz8rPlxROxugjZaF8Z4jf7g&export=download&confirm=t'
expected=$((range_end - range_start + 1))

while true; do
  current=$(stat -c %s "$part" 2>/dev/null || echo 0)
  if (( current == expected )); then
    break
  fi
  if (( current > expected )); then
    echo "oversized part: $part ($current > $expected)" >&2
    exit 1
  fi

  next_start=$((range_start + current))
  next_file="${part}.next"
  curl -fsSL -r "${next_start}-${range_end}" -o "$next_file" "$url" || true
  received=$(stat -c %s "$next_file" 2>/dev/null || echo 0)
  if (( received > expected - current )); then
    echo "server returned too many bytes for $part" >&2
    exit 1
  fi
  if (( received > 0 )); then
    dd if="$next_file" of="$part" bs=4M oflag=append conv=notrunc status=none
  else
    sleep 1
  fi
  rm -f "$next_file"
done

echo "$part complete: $expected bytes"
