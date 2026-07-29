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
    z_audio: torch.Tensor,      # query embeddings, audio tower  [B, D], normalized
    z_chord: torch.Tensor,      # query embeddings, chord tower  [B, D], normalized
    k_audio: torch.Tensor,      # key embeddings,   audio tower  [B, D], normalized (momentum encoder)
    k_chord: torch.Tensor,      # key embeddings,   chord tower  [B, D], normalized (momentum encoder)
    queue_audio: torch.Tensor,  # [K, D] negative queue of past audio keys
    queue_chord: torch.Tensor,  # [K, D] negative queue of past chord keys
    temperature: float = 0.07,
):
    batch_size = z_audio.shape[0]
    device = z_audio.device
 
    # Direction 1: audio query -> chord key (positive) + queue_chord (negatives)
    l_pos_a = torch.einsum("nc,nc->n", [z_audio, k_chord]).unsqueeze(-1)   # [B, 1]
    l_neg_a = z_audio @ queue_chord.T                                     # [B, K]
    logits_a = torch.cat([l_pos_a, l_neg_a], dim=1) / temperature         # [B, 1+K]
    labels_a = torch.zeros(batch_size, dtype=torch.long, device=device)   # positive is index 0
    loss_a = F.cross_entropy(logits_a, labels_a)
 
    # Direction 2: chord query -> audio key (positive) + queue_audio (negatives)
    l_pos_c = torch.einsum("nc,nc->n", [z_chord, k_audio]).unsqueeze(-1)
    l_neg_c = z_chord @ queue_audio.T
    logits_c = torch.cat([l_pos_c, l_neg_c], dim=1) / temperature
    labels_c = torch.zeros(batch_size, dtype=torch.long, device=device)
    loss_c = F.cross_entropy(logits_c, labels_c)
 
    return (loss_a + loss_c) / 2