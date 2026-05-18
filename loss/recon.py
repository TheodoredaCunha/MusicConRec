import torch
import torch.nn.functional as F
import torchaudio


def _stft_loss(x, y, n_fft, hop_length, win_length):
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

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

    # magnitude
    X_mag = X.abs().clamp(min=1e-9)
    Y_mag = Y.abs().clamp(min=1e-9)

    # spectral convergence
    sc_loss = torch.norm(Y_mag - X_mag, p="fro") / (torch.norm(Y_mag, p="fro") + 1e-9)

    # log magnitude loss
    log_loss = F.l1_loss(torch.log(Y_mag), torch.log(X_mag))

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


