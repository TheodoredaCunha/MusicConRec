import torch
import torch.nn.functional as F
import torchaudio


def _stft_loss(x, y, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length).to(x.device)

    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    Y = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    X_mag = X.abs()
    Y_mag = Y.abs()

    sc_loss = torch.norm(Y_mag - X_mag, p="fro") / torch.norm(Y_mag, p="fro")
    log_loss = F.l1_loss(torch.log(Y_mag + 1e-7), torch.log(X_mag + 1e-7))

    return sc_loss + log_loss


def multi_scale_stft_loss(x, y):
    losses = []
    configs = [
        (1024, 256, 1024),
        (512, 128, 512),
        (256, 64, 256),
    ]

    for n_fft, hop, win in configs:
        losses.append(_stft_loss(x, y, n_fft, hop, win))

    return sum(losses) / len(losses)
