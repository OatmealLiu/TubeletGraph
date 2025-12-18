#!/usr/bin/env bash

# Convert each .mp4 in the local ./videos folder to JPG frames (30 fps),
# saved as 0000000.jpg, 0000001.jpg, ... in a per-video output folder.

set -euo pipefail

# Resolve the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEOS_DIR="$SCRIPT_DIR/videos"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is not installed or not in PATH." >&2
  exit 1
fi

if [ ! -d "$VIDEOS_DIR" ]; then
  echo "Error: videos directory not found at: $VIDEOS_DIR" >&2
  exit 1
fi

shopt -s nullglob
mp4_files=("$VIDEOS_DIR"/*.mp4)

if [ "${#mp4_files[@]}" -eq 0 ]; then
  echo "No .mp4 files found in $VIDEOS_DIR" >&2
  exit 0
fi

for video in "${mp4_files[@]}"; do
  basename="$(basename "$video" .mp4)"
  out_dir="$SCRIPT_DIR/$basename"

  echo "Processing: $video"
  echo "Output directory: $out_dir"

  mkdir -p "$out_dir"

  # -vf fps=30: sample at 30 frames per second
  # -start_number 0: index output files starting from 0000000.jpg
  ffmpeg -i "$video" -vf fps=30 -start_number 0 "$out_dir/%07d.jpg"
done

echo "Done."


