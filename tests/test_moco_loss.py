import torch

from loss.ntxent import moco_contrastive_loss


def test_moco_contrastive_loss_is_finite():
    query_audio = torch.randn(4, 8)
    query_chord = torch.randn(4, 8)
    key_audio = torch.randn(4, 8)
    key_chord = torch.randn(4, 8)
    queue_audio = torch.randn(8, 8)
    queue_chord = torch.randn(8, 8)

    loss = moco_contrastive_loss(
        query_audio,
        query_chord,
        key_audio,
        key_chord,
        queue_audio,
        queue_chord,
        temperature=0.1,
    )

    assert torch.isfinite(loss)
    assert loss.ndim == 0
