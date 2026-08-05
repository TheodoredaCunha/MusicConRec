def get_training_phase_schedule(hp):
    total_epochs = int(hp.get("epochs", 50))
    encoder_pretrain_epochs = int(
        hp.get("encoder_pretrain_epochs", max(total_epochs // 2, 1))
    )
    contrastive_only_epochs = int(
        hp.get("contrastive_only_epochs", max(total_epochs - encoder_pretrain_epochs, 0))
    )

    if encoder_pretrain_epochs + contrastive_only_epochs < total_epochs:
        contrastive_only_epochs = total_epochs - encoder_pretrain_epochs

    if encoder_pretrain_epochs + contrastive_only_epochs > total_epochs:
        contrastive_only_epochs = max(total_epochs - encoder_pretrain_epochs, 0)

    phases = ["encoder_pretrain"] * encoder_pretrain_epochs + [
        "contrastive_only"
    ] * contrastive_only_epochs
    phases = phases[:total_epochs]

    if len(phases) < total_epochs:
        phases.extend(["contrastive_only"] * (total_epochs - len(phases)))

    return {
        "total_epochs": total_epochs,
        "encoder_pretrain_epochs": encoder_pretrain_epochs,
        "contrastive_only_epochs": contrastive_only_epochs,
        "phases": phases,
    }
