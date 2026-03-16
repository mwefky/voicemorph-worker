"""
RunPod Serverless Handler for Voice Morph.
Receives audio URL + preset → converts voice → returns result URL.
"""
import os
import sys
import time
import tempfile
import traceback

import runpod
import requests
import boto3
import torch
import numpy as np

# Add seed-vc to path
sys.path.insert(0, "/app/seed-vc")

# ─── Globals (loaded once, reused across requests) ───────────────────────────
converter = None

PRESET_DIR = "/app/presets"
PRESETS = {
    "female_01": os.path.join(PRESET_DIR, "female_01.wav"),
    "female_02": os.path.join(PRESET_DIR, "female_02.wav"),
}

QUALITY_MAP = {
    "fast": 10,
    "balanced": 25,
    "best": 50,
}

# R2 config (env vars set on RunPod endpoint)
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "voicemorph-audio")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def get_converter():
    global converter
    if converter is None:
        print("Loading models...")
        start = time.time()
        from voice_converter_serverless import VoiceConverter
        converter = VoiceConverter(fp16=True)
        converter.load_models()
        print(f"Models loaded in {time.time() - start:.1f}s")
    return converter


def download_file(url: str, dest: str):
    """Download a file from URL to local path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_result(local_path: str, key: str) -> str:
    """Upload result to R2 and return public URL."""
    s3 = get_s3_client()
    s3.upload_file(
        local_path, R2_BUCKET, key,
        ExtraArgs={"ContentType": "audio/wav"}
    )
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL}/{key}"
    return f"r2://{R2_BUCKET}/{key}"


def handler(event):
    """
    RunPod handler function.

    Input:
        audioUrl: str — URL to source audio file
        presetId: str — voice preset ID (female_01, female_02)
        quality: str — "fast", "balanced", or "best"
        jobId: str — unique job identifier for output naming

    Output:
        outputUrl: str — URL to converted audio
        duration: float — audio duration in seconds
        processingTime: float — time taken in seconds
    """
    try:
        inp = event.get("input", {})

        # Health check / test mode
        if inp.get("test"):
            return {
                "status": "ok",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu",
                "cuda_available": torch.cuda.is_available(),
                "presets_available": list(PRESETS.keys()),
            }

        audio_url = inp.get("audioUrl")
        preset_id = inp.get("presetId", "female_01")
        quality = inp.get("quality", "balanced")
        job_id = inp.get("jobId", f"job-{int(time.time())}")

        if not audio_url:
            return {"error": "Missing audioUrl"}

        # Validate preset
        preset_path = PRESETS.get(preset_id)
        if not preset_path or not os.path.exists(preset_path):
            return {"error": f"Unknown preset: {preset_id}"}

        diffusion_steps = QUALITY_MAP.get(quality, 25)

        # Download source audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_src:
            src_path = tmp_src.name
        print(f"Downloading source audio from {audio_url[:80]}...")
        download_file(audio_url, src_path)

        # Preprocess (lazy-load models on first real request)
        print("Preprocessing audio...")
        vc = get_converter()
        print(f"Model sr={vc.sr}, hop_length={vc.hop_length}")
        from audio_processor import preprocess_audio
        import soundfile as sf
        audio_data, sr = preprocess_audio(src_path, sr=vc.sr)
        preprocessed_path = f"/tmp/preprocessed_{job_id}.wav"
        print(f"Saving preprocessed audio: shape={audio_data.shape}, sr={sr}, path={preprocessed_path}")
        sf.write(preprocessed_path, audio_data, samplerate=int(sr))

        # Convert
        print(f"Converting with preset={preset_id}, steps={diffusion_steps}...")
        start = time.time()
        import torchaudio
        result_audio, out_sr = vc.convert(
            preprocessed_path,
            preset_path,
            diffusion_steps=diffusion_steps,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
        )
        processing_time = time.time() - start
        duration = len(result_audio) / out_sr
        print(f"Conversion done: {duration:.1f}s audio in {processing_time:.1f}s")

        # Save result
        output_path = f"/tmp/output_{job_id}.wav"
        wave_tensor = torch.tensor(result_audio)[None, :].float()
        torchaudio.save(output_path, wave_tensor, out_sr)

        # Upload to R2
        output_key = f"output/{job_id}.wav"
        output_url = upload_result(output_path, output_key)

        # Cleanup temp files
        for p in [src_path, preprocessed_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)

        return {
            "outputUrl": output_url,
            "duration": round(duration, 2),
            "processingTime": round(processing_time, 2),
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── RunPod Entry Point ─────────────────────────────────────────────────────

import signal
import threading

def _log_system_info():
    """Log system info for debugging."""
    print(f"Python: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_mem / 1e9:.1f} GB")
    print(f"R2 configured: {bool(R2_ACCOUNT_ID and R2_ACCESS_KEY)}")
    print(f"Presets: {[k for k in PRESETS if os.path.exists(PRESETS[k])]}")
    import importlib
    print(f"RunPod version: {importlib.import_module('runpod').__version__}")

print("=" * 60)
print("VoiceMorph Worker Starting")
print("=" * 60)

try:
    _log_system_info()
except Exception as e:
    print(f"System info error: {e}")

# Pre-load models in background so handler starts immediately
def _preload():
    try:
        print("Pre-loading models in background...")
        get_converter()
        print("Models pre-loaded successfully!")
    except Exception as e:
        print(f"WARNING: Model pre-load failed: {e}")
        traceback.print_exc()

preload_thread = threading.Thread(target=_preload, daemon=True)
preload_thread.start()

# CRITICAL: Start handler immediately (don't wait for model loading)
print("Starting RunPod serverless handler...")
sys.stdout.flush()
sys.stderr.flush()
runpod.serverless.start({"handler": handler})
