#!/usr/bin/env bash
# PocketTTS Voice Registration (Linux / macOS)
# Converts an MP3 to 24 kHz mono WAV and registers it with the PocketTTS server.

set -euo pipefail

echo "========================================"
echo "   PocketTTS Voice Registration"
echo "========================================"
echo

# Check for ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "[ERROR] ffmpeg not found. Install it and add it to PATH." >&2
    echo "  sudo apt install ffmpeg      # Debian/Ubuntu"
    echo "  brew install ffmpeg          # macOS"
    exit 1
fi

# Check for curl
if ! command -v curl &>/dev/null; then
    echo "[ERROR] curl not found. Install curl." >&2
    exit 1
fi

read -rp "Enter MP3 file path: " mp3_path
read -rp "Enter voice name: " voice_name

# Strip surrounding quotes
mp3_path="${mp3_path#\"}"
mp3_path="${mp3_path%\"}"
voice_name="${voice_name#\"}"
voice_name="${voice_name%\"}"

# Validate MP3 exists
if [[ ! -f "$mp3_path" ]]; then
    echo "[ERROR] MP3 file not found: $mp3_path" >&2
    exit 1
fi

# Derive output WAV path (same dir as mp3, with _converted.wav suffix)
mp3_dir="$(dirname "$mp3_path")"
mp3_base="$(basename "$mp3_path" .mp3)"
wav_path="$mp3_dir/${mp3_base}_converted.wav"

echo
echo "[1/2] Converting MP3 to WAV..."
echo

ffmpeg -y -i "$mp3_path" -ac 1 -ar 24000 -c:a pcm_s16le "$wav_path"

echo "[OK] WAV conversion complete: $wav_path"
echo "[2/2] Uploading voice..."

# Read WAV and base64-encode it, then POST as JSON
b64=$(base64 -w0 "$wav_path")
payload=$(printf '{"name":"%s","wav_b64":"%s"}' "$voice_name" "$b64")

response=$(curl -s -w "\n%{http_code}" \
    -X POST http://127.0.0.1:8080/v1/audio/voices \
    -H "Content-Type: application/json" \
    -d "$payload" \
    --max-time 180)

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
    echo "[SUCCESS] Voice registered!"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
else
    echo "[ERROR] Registration failed (HTTP $http_code)." >&2
    echo "$body" >&2
    exit 1
fi

echo
