"""
Pre-download all models during Docker build so they are baked into the image.
Run this script once during `docker build` to avoid runtime downloads.
"""
import os
from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = "/app/models"


def download_dit_checkpoint():
    """Download Seed-VC DiT checkpoint and config."""
    dest = os.path.join(MODELS_DIR, "seed-vc")
    os.makedirs(dest, exist_ok=True)

    dit_ckpt = hf_hub_download(
        repo_id="Plachta/Seed-VC",
        filename="DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        cache_dir=dest,
    )
    dit_config = hf_hub_download(
        repo_id="Plachta/Seed-VC",
        filename="config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        cache_dir=dest,
    )
    print(f"DiT checkpoint: {dit_ckpt}")
    print(f"DiT config:     {dit_config}")
    return dit_ckpt, dit_config


def download_campplus():
    """Download CAMPPlus speaker embedding model."""
    dest = os.path.join(MODELS_DIR, "campplus")
    os.makedirs(dest, exist_ok=True)

    ckpt = hf_hub_download(
        repo_id="funasr/campplus",
        filename="campplus_cn_common.bin",
        cache_dir=dest,
    )
    print(f"CAMPPlus checkpoint: {ckpt}")
    return ckpt


def download_bigvgan():
    """Download BigVGAN v2 vocoder."""
    dest = os.path.join(MODELS_DIR, "bigvgan")
    os.makedirs(dest, exist_ok=True)

    snapshot_download(
        repo_id="nvidia/bigvgan_v2_22khz_80band_256x",
        cache_dir=dest,
        allow_patterns=["*.json", "*.pt", "*.bin"],
    )
    print(f"BigVGAN vocoder downloaded to: {dest}")


def download_whisper():
    """Download Whisper small speech tokenizer."""
    dest = os.path.join(MODELS_DIR, "whisper")
    os.makedirs(dest, exist_ok=True)

    snapshot_download(
        repo_id="openai/whisper-small",
        cache_dir=dest,
    )
    print(f"Whisper tokenizer downloaded to: {dest}")


if __name__ == "__main__":
    print("Downloading all models to bake into Docker image...")
    download_dit_checkpoint()
    download_campplus()
    download_bigvgan()
    download_whisper()
    print("All models downloaded successfully.")
