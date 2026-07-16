import torch
import torch.nn.functional as F


def nt_xent(z_audio, z_chord, temperature=0.07):
    """
    z_audio: (B, D)
    z_chord: (B, D)
    """

    z_audio = F.normalize(z_audio, dim=1)
    z_chord = F.normalize(z_chord, dim=1)

    batch_size = z_audio.size(0)
    logits = torch.matmul(z_audio, z_chord.T) / temperature
    labels = torch.arange(batch_size, device=z_audio.device)

    loss_a2c = F.cross_entropy(logits, labels)
    loss_c2a = F.cross_entropy(logits.T, labels)

    return (loss_a2c + loss_c2a) / 2


def moco_contrastive_loss(
    query_audio,
    query_chord,
    key_audio,
    key_chord,
    queue_audio,
    queue_chord,
    temperature=0.07,
):
    """MoCo-style contrastive loss over current batch and a momentum queue."""

    q_a = F.normalize(query_audio, dim=1)
    q_c = F.normalize(query_chord, dim=1)
    k_a = F.normalize(key_audio, dim=1)
    k_c = F.normalize(key_chord, dim=1)
    queue_a = F.normalize(queue_audio, dim=1)
    queue_c = F.normalize(queue_chord, dim=1)

    batch_size = q_a.size(0)

    logits_a2c = torch.matmul(q_a, torch.cat([k_c, queue_c], dim=0).T) / temperature
    logits_c2a = torch.matmul(q_c, torch.cat([k_a, queue_a], dim=0).T) / temperature

    targets = torch.arange(batch_size, device=q_a.device)
    loss_a2c = F.cross_entropy(logits_a2c, targets)
    loss_c2a = F.cross_entropy(logits_c2a, targets)

    return (loss_a2c + loss_c2a) / 2