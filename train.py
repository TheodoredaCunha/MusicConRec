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
from training_phase import get_training_phase_schedule

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
    phase_schedule = get_training_phase_schedule(hp)
    print(
        "Alternating training schedule:",
        phase_schedule["phase_block_epochs"],
        "epochs per phase block",
        phase_schedule["phase_order"],
    )
    print(
        "Per-task caps:",
        phase_schedule["max_phase_epochs"]["encoder_pretrain"],
        "recon max,",
        phase_schedule["max_phase_epochs"]["contrastive_only"],
        "contrastive max",
    )

    model = MusicConRec(
        momentum=hp.get("moco_momentum", 0.999),
        train_encodec=False
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = build_scheduler(optimizer, hp)

    patience = int(hp.get("early_stopping_patience", 5))
    phase_order = phase_schedule["phase_order"]
    phase_block_epochs = phase_schedule["phase_block_epochs"]
    max_phase_epochs = phase_schedule["max_phase_epochs"]

    best_phase_metrics = {
        "encoder_pretrain": float("inf"),
        "contrastive_only": float("inf"),
    }
    phase_patience_counters = {
        "encoder_pretrain": 0,
        "contrastive_only": 0,
    }
    phase_epoch_counts = {
        "encoder_pretrain": 0,
        "contrastive_only": 0,
    }
    best_val_loss = float("inf")
    global_epoch = 0

    while True:
        phase_changed = False
        for phase in phase_order:
            if phase_epoch_counts[phase] >= max_phase_epochs[phase]:
                continue

            print(f"Starting {phase} phase block (epoch {global_epoch})")

            if phase == "encoder_pretrain":
                for p in model.encodec.parameters():
                    p.requires_grad = True
                for p in model.code_embedding.parameters():
                    p.requires_grad = True
                for p in model.audio_pool.parameters():
                    p.requires_grad = True
                for p in model.audio_proj.parameters():
                    p.requires_grad = True
                for p in model.chord_encoder.parameters():
                    p.requires_grad = True
            else:
                for p in model.encodec.parameters():
                    p.requires_grad = False
                for p in model.code_embedding.parameters():
                    p.requires_grad = True
                for p in model.audio_pool.parameters():
                    p.requires_grad = True
                for p in model.audio_proj.parameters():
                    p.requires_grad = True
                for p in model.chord_encoder.parameters():
                    p.requires_grad = True

            for block_index in range(phase_block_epochs):
                if phase_epoch_counts[phase] >= max_phase_epochs[phase]:
                    break

                epoch = global_epoch
                print(f"epoch {epoch} | phase={phase}")

                model.train()
                train_loss = 0
                train_contrastive = 0
                train_recon = 0

                for audio, chord_beats in tqdm(train_loader, desc=f"Train {phase} Epoch {phase_epoch_counts[phase]}"):
                    audio = audio.to(device)
                    chord_beats = chord_beats.to(device)

                    output = model(audio, chord_beats)
                    x_recon = output['x_recon']
                    z_audio = output['z_audio']
                    z_chord = output['z_chord']

                    model.update_momentum_encoders()
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

                    x = audio.squeeze(1)
                    x_hat = x_recon.squeeze(1)
                    recon_loss = multi_scale_stft_loss(x, x_hat)

                    if phase == "encoder_pretrain":
                        loss = hp["lambda_recon"] * recon_loss
                    else:
                        loss = hp["lambda_contrastive"] * contrastive_loss

                    if torch.isnan(loss) or torch.isinf(loss):
                        print("LOSS IS NAN or INF")
                        print("contrastive", contrastive_loss.item())
                        print("recon", recon_loss.item())
                        exit(1)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp.get("max_grad_norm", 1.0))
                    optimizer.step()
                    model.update_moco_queue(k_audio, k_chord)

                    train_loss += loss.item()
                    train_contrastive += contrastive_loss.item()
                    train_recon += recon_loss.item()

                train_loss /= len(train_loader)
                train_contrastive /= len(train_loader)
                train_recon /= len(train_loader)

                writer.add_scalar("Loss/Train_Total", train_loss, global_epoch)
                writer.add_scalar("Loss/Train_Contrastive", train_contrastive, global_epoch)
                writer.add_scalar("Loss/Train_Reconstruction", train_recon, global_epoch)

                model.eval()
                val_loss = 0
                val_contrastive = 0
                val_recon = 0

                with torch.no_grad():
                    for audio, chord_beats in tqdm(val_loader, desc=f"Val {phase} Epoch {phase_epoch_counts[phase]}"):
                        audio = audio.to(device)
                        chord_beats = chord_beats.to(device)

                        output = model(audio, chord_beats)
                        x_recon = output['x_recon']
                        z_audio = output['z_audio']
                        z_chord = output['z_chord']

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
                        recon_loss = multi_scale_stft_loss(audio.squeeze(1), x_recon.squeeze(1))

                        if phase == "encoder_pretrain":
                            loss = hp["lambda_recon"] * recon_loss
                        else:
                            loss = hp["lambda_contrastive"] * contrastive_loss

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

                writer.add_scalar("Loss/Val_Total", val_loss, global_epoch)
                writer.add_scalar("Loss/Val_Contrastive", val_contrastive, global_epoch)
                writer.add_scalar("Loss/Val_Reconstruction", val_recon, global_epoch)
                writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]["lr"], global_epoch)

                train_recon_weighted = hp["lambda_recon"] * train_recon
                val_recon_weighted = hp["lambda_recon"] * val_recon
                print(
                    f"Epoch {epoch} | phase={phase} | Train Total: {train_loss:.4f} | "
                    f"Train Contrastive: {train_contrastive:.4f} | "
                    f"Train λ*Recon: {train_recon_weighted:.4f} | "
                    f"Val Total: {val_loss:.4f} | "
                    f"Val Contrastive: {val_contrastive:.4f} | "
                    f"Val λ*Recon: {val_recon_weighted:.4f}"
                )

                metric_to_watch = val_recon if phase == "encoder_pretrain" else val_contrastive
                if metric_to_watch < best_phase_metrics[phase]:
                    best_phase_metrics[phase] = metric_to_watch
                    phase_patience_counters[phase] = 0
                    best_model_path = os.path.join(model_dir, f"best_model_{phase}.pth")
                    torch.save(model.state_dict(), best_model_path)
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                    print(f"✓ Best {phase} model saved ({metric_to_watch:.4f})")
                else:
                    phase_patience_counters[phase] += 1

                phase_epoch_counts[phase] += 1
                global_epoch += 1

                if phase_patience_counters[phase] >= patience:
                    print(f"Early stopping triggered for {phase} after {phase_epoch_counts[phase]} epochs")
                    phase_changed = True
                    break

            if phase_patience_counters[phase] >= patience:
                print(f"{phase} phase completed early due to patience")
                phase_changed = True
                break

            if phase_epoch_counts[phase] >= max_phase_epochs[phase]:
                print(f"Reached max epochs for {phase} ({max_phase_epochs[phase]})")
                phase_changed = True

        if not phase_changed:
            break

        if all(phase_epoch_counts[p] >= max_phase_epochs[p] for p in phase_order):
            break

        phase_changed = False

    writer.close()
    final_model_path = os.path.join(model_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model (validation) saved to: {os.path.join(model_dir, 'best_model.pth')}")
    print(f"Best validation loss: {best_val_loss:.4f}")