def get_training_phase_schedule(hp):
    total_epochs = int(hp.get("epochs", 300))
    phase_block_epochs = int(hp.get("alternating_phase_epochs", 30))
    max_phase_epochs = {
        "encoder_pretrain": int(hp.get("max_recon_epochs", 300)),
        "contrastive_only": int(hp.get("max_contrastive_epochs", 300)),
    }

    phase_order = ["encoder_pretrain", "contrastive_only"]

    return {
        "total_epochs": total_epochs,
        "phase_block_epochs": phase_block_epochs,
        "phase_order": phase_order,
        "max_phase_epochs": max_phase_epochs,
    }
