import torch
import torchaudio


def resample(audio, sr, target_sr=24000, mono=True):
    # audio: torch.Tensor (C, T)

    # ---- mono ----
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)  # (1, T)

    # ---- resample ----
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    return audio, target_sr