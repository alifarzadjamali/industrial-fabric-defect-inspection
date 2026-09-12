from pathlib import Path

import pandas as pd


def test_enhanced_protocol_lock_is_complete_and_leakage_safe() -> None:
    lock = pd.read_csv(Path("configs/splits/enhanced_protocol.csv"))
    assert len(lock) == 247
    assert lock["image_id"].is_unique
    assert set(lock["split"]) == {"train", "validation", "final_test"}
    assert (lock["split"] == "final_test").sum() == 38
    assert (lock.loc[lock["split"] == "final_test", "previous_split"] == "train").all()
    assert (
        lock.loc[lock["previous_split"] == "validation", "split"] == "validation"
    ).all()
