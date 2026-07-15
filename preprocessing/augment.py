import random
import torch


def random_pitch_shift(waveform, sr, min_steps=-2, max_steps=2):
    """Apply a random pitch shift to waveform by resampling it.

    The audio is sped up or slowed down by a small factor, which changes pitch.
    The returned pitch shift steps are used to transpose the chord-beat tensor.
    """
    if waveform.dim() != 2:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[-1] < 2:
        return waveform, 0

    steps = random.randint(min_steps, max_steps)
    if steps == 0:
        return waveform, 0

    rate = 2 ** (steps / 12)
    new_length = max(1, int(waveform.shape[-1] / rate))
    resampled = torch.nn.functional.interpolate(
        waveform.unsqueeze(1),
        size=new_length,
        mode="linear",
        align_corners=False,
    ).squeeze(1)

    if resampled.shape[-1] < waveform.shape[-1]:
        pad_len = waveform.shape[-1] - resampled.shape[-1]
        resampled = torch.nn.functional.pad(resampled, (0, pad_len))
    else:
        resampled = resampled[..., : waveform.shape[-1]]

    return resampled, steps


def transpose_chord_representation(chord_representation, pitch_shift_steps):
    """Transpose the 12-bin chord representation by a number of semitones."""
    if pitch_shift_steps == 0:
        return chord_representation

    if chord_representation.dim() != 2:
        return chord_representation

    if chord_representation.shape[1] < 13:
        return chord_representation

    transposed = chord_representation.clone()
    pitch_classes = chord_representation[:, :12]
    shifted = torch.zeros_like(pitch_classes)

    for semitone in range(12):
        src_idx = (semitone - pitch_shift_steps) % 12
        shifted[:, semitone] = pitch_classes[:, src_idx]

    transposed[:, :12] = shifted
    return transposed
