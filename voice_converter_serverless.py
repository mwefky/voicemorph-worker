"""
Voice conversion engine wrapping Seed-VC for RunPod serverless.
CUDA-only. Loads models from local baked paths (/app/models/).
"""
import os
import sys
import time
import numpy as np
import torch
import torchaudio
import librosa
import yaml
from types import SimpleNamespace

# Add the worker directory to path so we can import seed-vc modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


class VoiceConverter:
    """Wraps Seed-VC for zero-shot voice conversion. CUDA-only serverless version."""

    def __init__(self, fp16: bool = True):
        self.device = torch.device("cuda")
        self.fp16 = fp16
        self.model = None
        self.semantic_fn = None
        self.vocoder_fn = None
        self.campplus_model = None
        self.mel_fn = None
        self.mel_fn_args = None
        self.sr = None
        self.hop_length = None
        self.loaded = False

    def load_models(self, progress_callback=None):
        """Load all Seed-VC models from baked local paths. Call once at startup."""
        if self.loaded:
            return

        from modules.commons import build_model, load_checkpoint, recursive_munch
        from hf_utils import load_custom_model_from_hf

        # --- Override HF cache to use baked model paths ---
        seed_vc_cache = os.path.join(MODELS_DIR, "seed-vc")
        campplus_cache = os.path.join(MODELS_DIR, "campplus")
        bigvgan_cache = os.path.join(MODELS_DIR, "bigvgan")
        whisper_cache = os.path.join(MODELS_DIR, "whisper")

        os.environ["HF_HUB_CACHE"] = seed_vc_cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = seed_vc_cache

        if progress_callback:
            progress_callback(0.1, "Loading DiT model...")

        # Load DiT checkpoint (speech conversion model) from local cache
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        )

        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = "DiT"
        model = build_model(model_params, stage="DiT")
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr = config["preprocess_params"]["sr"]

        model, _, _, _ = load_checkpoint(
            model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.model = model

        if progress_callback:
            progress_callback(0.4, "Loading speaker embedding model...")

        # Load CAMPPlus (speaker embedding)
        from modules.campplus.DTDNN import CAMPPlus

        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(
            torch.load(campplus_ckpt_path, map_location="cpu")
        )
        campplus_model.eval()
        campplus_model.to(self.device)
        self.campplus_model = campplus_model

        if progress_callback:
            progress_callback(0.6, "Loading vocoder...")

        # Load BigVGAN vocoder — on CUDA (supports iFFT unlike MPS)
        from modules.bigvgan import bigvgan

        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            bigvgan_name, use_cuda_kernel=False
        )
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(self.device)
        self.vocoder_fn = bigvgan_model

        if progress_callback:
            progress_callback(0.8, "Loading speech tokenizer...")

        # Load Whisper speech tokenizer
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(
            whisper_name, torch_dtype=torch.float16
        ).to(self.device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        device = self.device

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori

        self.semantic_fn = semantic_fn

        # Mel spectrogram function
        self.mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None
            if config["preprocess_params"]["spect_params"].get("fmax", "None")
            == "None"
            else 8000,
            "center": False,
        }
        from modules.audio import mel_spectrogram

        self.mel_fn = lambda x: mel_spectrogram(x, **self.mel_fn_args)

        self.loaded = True
        if progress_callback:
            progress_callback(1.0, "Models loaded!")

    @staticmethod
    def _crossfade(chunk1, chunk2, overlap):
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = (
                chunk2[:overlap] * fade_in[: len(chunk2)]
                + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
            )
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    @torch.no_grad()
    @torch.inference_mode()
    def convert(
        self,
        source_path: str,
        reference_path: str,
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
    ) -> tuple[np.ndarray, int]:
        """
        Convert source voice to sound like reference voice.

        Args:
            source_path: Path to source audio (your voice)
            reference_path: Path to reference audio (target voice)
            diffusion_steps: Quality vs speed (10=fast, 25=balanced, 50-100=best)
            length_adjust: Speed adjustment (<1.0=faster, >1.0=slower)
            inference_cfg_rate: Guidance strength (0.0-1.0)

        Returns:
            (audio_array, sample_rate)
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        sr = self.sr
        hop_length = self.hop_length
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * hop_length
        max_context_window = sr // hop_length * 30

        # Load audio
        source_audio = librosa.load(source_path, sr=sr)[0]
        ref_audio = librosa.load(reference_path, sr=sr)[0]

        source_audio = (
            torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
        )
        ref_audio = (
            torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(self.device)
        )

        # Extract semantic features from source
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = self.semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:
                    chunk = converted_waves_16k[
                        :, traversed_time : traversed_time + 16000 * 30
                    ]
                else:
                    chunk = torch.cat(
                        [
                            buffer,
                            converted_waves_16k[
                                :,
                                traversed_time : traversed_time
                                + 16000 * (30 - overlapping_time),
                            ],
                        ],
                        dim=-1,
                    )
                S_alt = self.semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time :])
                buffer = chunk[:, -16000 * overlapping_time :]
                traversed_time += (
                    30 * 16000
                    if traversed_time == 0
                    else chunk.size(-1) - 16000 * overlapping_time
                )
            S_alt = torch.cat(S_alt_list, dim=1)

        # Extract semantic features from reference
        ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        S_ori = self.semantic_fn(ori_waves_16k)

        # Mel spectrograms
        mel = self.mel_fn(source_audio.float()).to(self.device)
        mel2 = self.mel_fn(ref_audio.float()).to(self.device)

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(
            mel.device
        )
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        # Speaker embedding from reference
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.cpu(), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0).to(self.device))

        # Length regulation
        cond, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=None
        )
        prompt_condition, _, codes, commitment_loss, codebook_loss = (
            self.model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
            )
        )

        # Generate chunk by chunk with crossfade
        max_source_window = max_context_window - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []

        while processed_frames < cond.size(1):
            chunk_cond = cond[
                :, processed_frames : processed_frames + max_source_window
            ]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            # Run diffusion on CUDA
            vc_target = self.model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]

            # Vocoder on CUDA (supports iFFT)
            vc_wave = self.vocoder_fn(vc_target.float()).squeeze()
            if vc_wave.ndim == 0:
                vc_wave = vc_wave.unsqueeze(0)
            vc_wave = vc_wave[None, :]

            if processed_frames == 0:
                if is_last_chunk:
                    generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                    break
                generated_wave_chunks.append(
                    vc_wave[0, :-overlap_wave_len].cpu().numpy()
                )
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last_chunk:
                output_wave = self._crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                break
            else:
                output_wave = self._crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len

        result = np.concatenate(generated_wave_chunks)
        return result, sr

    def convert_and_save(
        self,
        source_path: str,
        reference_path: str,
        output_path: str,
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
    ) -> str:
        """Convert and save to file. Returns output path."""
        audio, sr = self.convert(
            source_path,
            reference_path,
            diffusion_steps,
            length_adjust,
            inference_cfg_rate,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        wave_tensor = torch.tensor(audio)[None, :].float()
        torchaudio.save(output_path, wave_tensor.cpu(), sr)
        return output_path
