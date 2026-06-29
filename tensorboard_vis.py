import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


RUN_NAME = "20260519-042801"
RUN_PATH = os.path.join("runs", RUN_NAME)


def plot_train_val(train_tag, val_tag, title):
    ea = EventAccumulator(RUN_PATH)
    ea.Reload()

    train_events = ea.Scalars(train_tag)
    val_events = ea.Scalars(val_tag)

    train_steps = [e.step for e in train_events]
    train_values = [e.value for e in train_events]

    val_steps = [e.step for e in val_events]
    val_values = [e.value for e in val_events]

    plt.figure(figsize=(16, 8))

    # Train = blue
    plt.plot(
        train_steps,
        train_values,
        color="blue",
        marker="o",
        linewidth=2,
        label="Train"
    )

    # Validation = orange
    plt.plot(
        val_steps,
        val_values,
        color="orange",
        marker="o",
        linewidth=2,
        label="Validation"
    )

    # annotate every point clearly
    for x, y in zip(train_steps, train_values):
        plt.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="blue"
        )

    for x, y in zip(val_steps, val_values):
        plt.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8,
            color="orange"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_best_run():
    plot_train_val(
        "Loss/Train_Total",
        "Loss/Val_Total",
        "Total Loss (Train vs Validation)"
    )

    plot_train_val(
        "Loss/Train_Contrastive",
        "Loss/Val_Contrastive",
        "Contrastive Loss (Train vs Validation)"
    )

    plot_train_val(
        "Loss/Train_Reconstruction",
        "Loss/Val_Reconstruction",
        "Reconstruction Loss (Train vs Validation)"
    )

