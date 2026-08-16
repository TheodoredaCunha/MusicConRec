import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import json
import os
import random
import numpy as np

from dataset import MusicBenchDataset, collate_fn
from model.model import MusicConRec
from loss.ntxent import moco_contrastive_loss
from loss.recon import multi_scale_stft_loss

import warnings



def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    import traceback
    traceback.print_stack()
    print(f"{filename}:{lineno}: {category.__name__}: {message}")

warnings.showwarning = warn_with_traceback

warnings.filterwarnings(
    "ignore",
    message=".*TorchCodec.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder.*"
)




def get_env_paths():
    is_sagemaker = "SM_MODEL_DIR" in os.environ

    if is_sagemaker:
        print("Running in SageMaker")

        train_dir = os.environ.get("SM_CHANNEL_TRAIN")
        val_dir = os.environ.get("SM_CHANNEL_VALIDATION")
        model_dir = os.environ.get("SM_MODEL_DIR")
        log_dir = "/opt/ml/output/tensorboard"

    else:
        print("Running locally")

        train_dir = "./dataset"
        val_dir = "./dataset"
        model_dir = "./outputs"
        log_dir = "./runs"

        os.makedirs(model_dir, exist_ok=True)

    os.makedirs(log_dir, exist_ok=True)

    return train_dir, val_dir, model_dir, log_dir


def build_scheduler(optimizer, hp):
    scheduler_cfg = hp.get("lr_scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "none").lower()

    if scheduler_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.1)
        )

    if scheduler_type == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_cfg.get("milestones", [10, 20, 30]),
            gamma=scheduler_cfg.get("gamma", 0.1)
        )

    if scheduler_type == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", hp.get("epochs", 50)),
            eta_min=scheduler_cfg.get("eta_min", 0.0)
        )

    if scheduler_type == "reducelronplateau":
        kwargs = {
            "mode": scheduler_cfg.get("mode", "min"),
            "factor": scheduler_cfg.get("factor", 0.1),
            "patience": scheduler_cfg.get("patience", 3),
            "threshold": scheduler_cfg.get("threshold", 1e-4),
            "min_lr": scheduler_cfg.get("min_lr", 0.0),
            "cooldown": scheduler_cfg.get("cooldown", 0),
        }
        if "verbose" in torch.optim.lr_scheduler.ReduceLROnPlateau.__init__.__code__.co_varnames:
            kwargs["verbose"] = scheduler_cfg.get("verbose", False)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # try:
    #     torch.use_deterministic_algorithms(True, warn_only=True)
    # except Exception:
    #     pass


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32 - 1)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_detect_anomaly(True)

    # =========================
    # HYPERPARAMETERS
    # =========================
    with open("training_hyperparam.json", "r") as f:
        hp = json.load(f)

    seed = int(hp.get("seed", 42))
    set_seed(seed)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    # =========================
    # PATHS (AUTO SWITCH)
    # =========================
    train_dir, val_dir, model_dir, log_dir = get_env_paths()

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    print("Training using MoCo-style contrastive loss with queue size:", hp.get("queue_size", 4096))
    print("Momentum (EMA) coefficient:", hp.get("moco_momentum", 0.99))
    print("Train dir:", train_dir)
    print("Val dir:", val_dir)
    print("Model dir:", model_dir)

    # =========================
    # DATASETS
    # =========================
    train_dataset = MusicBenchDataset(hp['traindata_dir'], train_dir, augment=True)
    val_dataset = MusicBenchDataset(hp['valdata_dir'], val_dir, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=hp["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=dl_generator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hp["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=hp["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=dl_generator,
        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================
    model = MusicConRec(
        momentum=0.999,
        train_encodec=True
    ).to(device)

    print("Unfreezing Encodec backbone to allow reconstruction loss to improve during joint training")

    # Only the ONLINE (query) modules should ever be in the optimizer — the
    # momentum (key) modules (encodec_k, code_embedding_k, audio_pool_k,
    # audio_proj_k, chord_encoder_k) are updated exclusively via EMA in
    # update_momentum_encoders() and must never receive a gradient step.
    momentum_module_names = {"encodec_k", "code_embedding_k", "audio_pool_k",
                              "audio_proj_k", "chord_encoder_k"}

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]


    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-4,
        weight_decay=1e-4
    )


    scheduler = build_scheduler(optimizer, hp)

    # =========================
    # TRAIN LOOP
    # =========================
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(hp["epochs"]):
        print("epoch ", epoch)

        model.train()
        train_loss = 0
        train_contrastive = 0
        train_recon = 0

        for audio, chord_beats in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            audio = audio.to(device)
            chord_beats = chord_beats.to(device)

            # Query encoders (grad-tracked) run first.
            output = model(audio, chord_beats)

            x_recon = output['x_recon']
            z_audio = output['z_audio']     # query embedding, audio tower
            z_chord = output['z_chord']     # query embedding, chord/beat tower

            # Momentum (key) encoders: EMA-update them from the current query
            # encoder weights, then forward the SAME batch through them.
            model.update_momentum_encoders()
            k_audio, k_chord = model.forward_momentum(audio, chord_beats)

            # Symmetric InfoNCE: queries vs. queue of the OTHER modality's keys,
            # positives are the in-batch (query, key) pairs from the same sample.
            contrastive_loss = moco_contrastive_loss(
                z_audio,
                z_chord,
                k_audio,
                k_chord,
                model.queue_audio,
                model.queue_chord,
                temperature=hp["ntxent_temperature"],
            )

            x = audio.squeeze(1)
            x_hat = x_recon.squeeze(1)

            recon_loss = multi_scale_stft_loss(x, x_hat)

            loss = (
                hp["lambda_contrastive"] * contrastive_loss +
                hp["lambda_recon"] * recon_loss
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print("LOSS IS NAN or INF")
                print("contrastive", contrastive_loss.item())
                print("recon", recon_loss.item())
                exit(1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp.get("max_grad_norm", 1.0))
            optimizer.step()

            # Enqueue the KEY embeddings only now, after backward()/step() have
            # both finished using the queue's pre-update values. Doing this
            # earlier causes: "one of the variables needed for gradient
            # computation has been modified by an inplace operation" — autograd
            # needs the queue's original contents to compute the gradient of
            # the matmul against it during backward(), so mutating it beforehand
            # invalidates the saved values.
            model.update_moco_queue(k_audio, k_chord)

            train_loss += loss.item()
            train_contrastive += contrastive_loss.item()
            train_recon += recon_loss.item()

        train_loss /= len(train_loader)
        train_contrastive /= len(train_loader)
        train_recon /= len(train_loader)

        writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        writer.add_scalar("Loss/Train_Contrastive", train_contrastive, epoch)
        writer.add_scalar("Loss/Train_Reconstruction", train_recon, epoch)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0
        val_contrastive = 0
        val_recon = 0

        with torch.no_grad():
            for audio, chord_beats in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                audio = audio.to(device)
                chord_beats = chord_beats.to(device)

                output = model(audio, chord_beats)
                x_recon = output['x_recon']
                z_audio = output['z_audio']
                z_chord = output['z_chord']

                # No momentum update and no queue mutation during eval —
                # score against the queue's current (training-time) state only.
                k_audio, k_chord = model.forward_momentum(audio, chord_beats)

                contrastive_loss = moco_contrastive_loss(
                    z_audio,
                    z_chord,
                    k_audio,
                    k_chord,
                    model.queue_audio,
                    model.queue_chord,
                    temperature=hp["ntxent_temperature"],
                )

                recon_loss = multi_scale_stft_loss(
                    audio.squeeze(1),
                    x_recon.squeeze(1)
                )

                loss = (
                    hp["lambda_contrastive"] * contrastive_loss +
                    hp["lambda_recon"] * recon_loss
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print("VALIDATION LOSS IS NAN or INF")
                    print("contrastive", contrastive_loss.item())
                    print("recon", recon_loss.item())
                    exit(1)

                val_loss += loss.item()
                val_contrastive += contrastive_loss.item()
                val_recon += recon_loss.item()

        val_loss /= len(val_loader)
        val_contrastive /= len(val_loader)
        val_recon /= len(val_loader)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Val_Contrastive", val_contrastive, epoch)
        writer.add_scalar("Loss/Val_Reconstruction", val_recon, epoch)
        writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        train_recon_weighted = hp["lambda_recon"] * train_recon
        val_recon_weighted = hp["lambda_recon"] * val_recon
        print(
            f"Epoch {epoch} | Train Total: {train_loss:.4f} | "
            f"Train Contrastive: {train_contrastive:.4f} | "
            f"Train λ*Recon: {train_recon_weighted:.4f} | "
            f"Val Total: {val_loss:.4f} | "
            f"Val Contrastive: {val_contrastive:.4f} | "
            f"Val λ*Recon: {val_recon_weighted:.4f}"
        )

        # =========================
        # CHECKPOINT SAVING
        # =========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    writer.close()
    final_model_path = os.path.join(model_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model (validation) saved to: {os.path.join(model_dir, 'best_model.pth')}")
    print(f"Best validation loss: {best_val_loss:.4f}")