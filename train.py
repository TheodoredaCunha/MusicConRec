import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import json
import os

from dataset import MusicBenchDataset, collate_fn
from model.model import MusicConRec 
from loss.ntxent import nt_xent
from loss.recon import multi_scale_stft_loss


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

        train_dir = "C:/temp"
        val_dir = "C:/temp"
        model_dir = "./outputs"
        log_dir = "./runs"

        os.makedirs(model_dir, exist_ok=True)

    os.makedirs(log_dir, exist_ok=True)

    return train_dir, val_dir, model_dir, log_dir


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # HYPERPARAMETERS
    # =========================
    with open("training_hyperparam.json", "r") as f:
        hp = json.load(f)

    # =========================
    # PATHS (AUTO SWITCH)
    # =========================
    train_dir, val_dir, model_dir, log_dir = get_env_paths()

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    print("Train dir:", train_dir)
    print("Val dir:", val_dir)
    print("Model dir:", model_dir)


    # =========================
    # DATASETS
    # =========================
    train_dataset = MusicBenchDataset(hp['traindata_dir'], train_dir)
    val_dataset = MusicBenchDataset(hp['valdata_dir'], val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hp["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================
    model = MusicConRec().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp["learning_rate"]
    )

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(hp["epochs"]):
        print("epoch ", epoch)

        model.train()
        train_loss = 0
        train_contrastive = 0
        train_recon = 0

        for audio, chord_beats in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            audio = audio.to(device)
            chord_beats = chord_beats.to(device)

            output = model(audio, chord_beats)

            x_recon = output['x_recon']
            z_audio = output['z_audio']
            z_chord = output['z_chord']

            contrastive_loss = nt_xent(
                z_audio,
                z_chord,
                hp["ntxent_temperature"]
            )

            recon_loss = multi_scale_stft_loss(
                audio.squeeze(1),
                x_recon.squeeze(1)
            )

            loss = (
                hp["lambda_contrastive"] * contrastive_loss +
                hp["lambda_recon"] * recon_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_contrastive += contrastive_loss.item()
            train_recon += recon_loss.item()

        # averages
        train_loss /= len(train_loader)
        train_contrastive /= len(train_loader)
        train_recon /= len(train_loader)

        # 🔥 LOG EVERYTHING
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

                contrastive_loss = nt_xent(
                    z_audio,
                    z_chord,
                    hp["ntxent_temperature"]
                )

                recon_loss = multi_scale_stft_loss(
                    audio.squeeze(1),
                    x_recon.squeeze(1)
                )

                loss = (
                    hp["lambda_contrastive"] * contrastive_loss +
                    hp["lambda_recon"] * recon_loss
                )

                val_loss += loss.item()
                val_contrastive += contrastive_loss.item()
                val_recon += recon_loss.item()

        # averages
        val_loss /= len(val_loader)
        val_contrastive /= len(val_loader)
        val_recon /= len(val_loader)

        # 🔥 LOG
        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Val_Contrastive", val_contrastive, epoch)
        writer.add_scalar("Loss/Val_Reconstruction", val_recon, epoch)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    writer.close()
    # =========================
    # SAVE MODEL
    # =========================
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    print("Model saved to:", model_path)