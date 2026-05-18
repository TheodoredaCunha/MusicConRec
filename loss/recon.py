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

    # spectral convergence (more stable)
    sc_num = ((Y_mag - X_mag) ** 2).mean().sqrt()
    sc_den = (Y_mag ** 2).mean().sqrt() + 1e-7
    sc_loss = sc_num / sc_den

    # log magnitude loss
    log_loss = F.l1_loss(
        torch.log1p(Y_mag),
        torch.log1p(X_mag)
    )

    # print("sc_loss:", sc_loss.item())
    # print("log_loss:", log_loss.item())
    # print("X_mag max:", X_mag.max().item())
    # print("Y_mag max:", Y_mag.max().item())

    return sc_loss + log_loss


def multi_scale_stft_loss(x, y):
    losses = []
    
    configs = [
        (256, 64, 256),
    ]
    
    for n_fft, hop, win in configs:
        losses.append(_stft_loss(x, y, n_fft, hop, win))
    
    return sum(losses) / len(losses)


