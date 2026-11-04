@echo off
title PocketTTS Voice Registration
color 0A

echo ========================================
echo       PocketTTS Voice Registration
echo ========================================
echo.

set /p "MP3_PATH=Enter MP3 file path: "
set /p "VOICE_NAME=Enter voice name: "

set "MP3_PATH=%MP3_PATH:"=%"

echo.
echo [1/2] Converting MP3 to WAV...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$ErrorActionPreference='Stop'; ^
$mp3='%MP3_PATH%'; ^
$name='%VOICE_NAME%'; ^
if (!(Test-Path -LiteralPath $mp3 -PathType Leaf)) { throw 'MP3 file not found' }; ^
$wav=[IO.Path]::ChangeExtension($mp3,'_converted.wav'); ^
$ffmpeg=Get-Command ffmpeg -ErrorAction SilentlyContinue; ^
if (!$ffmpeg) { throw 'FFmpeg not found. Install FFmpeg and add it to PATH.' }; ^
& $ffmpeg.Source -y -i $mp3 -ac 1 -ar 24000 -c:a pcm_s16le $wav; ^
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg conversion failed' }; ^
Write-Host '[OK] WAV conversion complete' -ForegroundColor Green; ^
Write-Host '[2/2] Uploading voice...'; ^
$bytes=[IO.File]::ReadAllBytes($wav); ^
$payload=@{name=$name;wav_b64=[Convert]::ToBase64String($bytes)} | ConvertTo-Json -Compress; ^
$response=Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/audio/voices' -Method Post -ContentType 'application/json' -Body $payload -TimeoutSec 180; ^
Write-Host '[SUCCESS] Voice registered!' -ForegroundColor Green; ^
$response | ConvertTo-Json -Depth 10"

if errorlevel 1 (
    echo.
    echo [ERROR] Registration failed.
)

echo.
pause
