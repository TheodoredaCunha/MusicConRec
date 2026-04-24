import torch
import torch.nn.functional as F

def nt_xent(z_audio, z_chord, temperature=0.07):
    """
    z_audio: (B, D)
    z_chord: (B, D)
    """

    # Normalize (VERY IMPORTANT)
    z_audio = F.normalize(z_audio, dim=1)
    z_chord = F.normalize(z_chord, dim=1)

    batch_size = z_audio.size(0)

    # Similarity matrix (B x B)
    logits = torch.matmul(z_audio, z_chord.T) / temperature

    # Labels: diagonal is positive
    labels = torch.arange(batch_size, device=z_audio.device)

    # Cross entropy loss (audio -> chord)
    loss_a2c = F.cross_entropy(logits, labels)

    # Cross entropy loss (chord -> audio)
    loss_c2a = F.cross_entropy(logits.T, labels)

    return (loss_a2c + loss_c2a) / 2