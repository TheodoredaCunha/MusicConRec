import torch
import torch.nn.functional as F


def nt_xent(z_audio, z_chord, temperature=0.07):
    z_audio = F.normalize(z_audio, dim=1)
    z_chord = F.normalize(z_chord, dim=1)

    batch_size = z_audio.size(0)
    logits = torch.matmul(z_audio, z_chord.T) / temperature
    labels = torch.arange(batch_size, device=z_audio.device)

    loss_a2c = F.cross_entropy(logits, labels)
    loss_c2a = F.cross_entropy(logits.T, labels)
    return (loss_a2c + loss_c2a) / 2
