from summarize_improved_validation_sweep import rank_candidates


def test_rank_candidates_requires_all_predeclared_validation_gates():
    rows = [
        {"candidate_id": 0, "slug": "control", "validation_patient_prostate_dice": 0.70, "validation_worst_patient_prostate_dice": 0.60, "validation_prostate_false_positive_voxels": 100, "validation_prostate_false_negative_voxels": 100, "peak_train_vram_mb": 1000.0, "training_seconds": 100.0},
        {"candidate_id": 1, "slug": "accepted", "validation_patient_prostate_dice": 0.715, "validation_worst_patient_prostate_dice": 0.59, "validation_prostate_false_positive_voxels": 90, "validation_prostate_false_negative_voxels": 90, "peak_train_vram_mb": 1300.0, "training_seconds": 110.0},
        {"candidate_id": 2, "slug": "small_gain", "validation_patient_prostate_dice": 0.709, "validation_worst_patient_prostate_dice": 0.61, "validation_prostate_false_positive_voxels": 1, "validation_prostate_false_negative_voxels": 1, "peak_train_vram_mb": 900.0, "training_seconds": 90.0},
        {"candidate_id": 3, "slug": "vram_high", "validation_patient_prostate_dice": 0.72, "validation_worst_patient_prostate_dice": 0.61, "validation_prostate_false_positive_voxels": 1, "validation_prostate_false_negative_voxels": 1, "peak_train_vram_mb": 1351.0, "training_seconds": 90.0},
    ]

    ranked = rank_candidates(rows)

    assert ranked[0]["slug"] == "accepted"
    assert ranked[0]["accepted"] is True
    assert ranked[1]["slug"] == "small_gain"
    assert ranked[1]["accepted"] is False
    assert ranked[2]["slug"] == "vram_high"
    assert ranked[2]["accepted"] is False
