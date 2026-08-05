from training_phase import get_training_phase_schedule


def test_training_phase_schedule_splits_encoder_and_contrastive_phases():
    hp = {
        "epochs": 50,
        "encoder_pretrain_epochs": 10,
        "contrastive_only_epochs": 40,
    }

    schedule = get_training_phase_schedule(hp)

    assert schedule["total_epochs"] == 50
    assert schedule["encoder_pretrain_epochs"] == 10
    assert schedule["contrastive_only_epochs"] == 40
    assert schedule["phases"][:10] == ["encoder_pretrain"] * 10
    assert schedule["phases"][10:] == ["contrastive_only"] * 40
