from training_phase import get_training_phase_schedule


def test_training_phase_schedule_uses_alternating_30_epoch_blocks():
    hp = {
        "epochs": 300,
        "alternating_phase_epochs": 30,
        "max_phase_epochs": 300,
    }

    schedule = get_training_phase_schedule(hp)

    assert schedule["phase_block_epochs"] == 30
    assert schedule["phase_order"] == ["encoder_pretrain", "contrastive_only"]
    assert schedule["max_phase_epochs"]["encoder_pretrain"] == 300
    assert schedule["max_phase_epochs"]["contrastive_only"] == 300
