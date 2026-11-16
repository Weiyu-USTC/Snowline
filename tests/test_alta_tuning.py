"""Leakage, stopping and checkpoint safeguards for the bounded search."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from alta_radar_models import RadarConfig, make_model, encode_inputs, encode_targets, load_bundle
from tune_alta_radar import Stopper, fixed_score, train_run, probe_indices, sampling_probabilities


def test_stopping_requires_patience_and_minimum_budget():
    s = Stopper(patience=3, min_steps=500)
    assert not s.update(1., 100)
    assert not s.update(.999, 200)
    assert not s.update(.998, 300)
    assert not s.update(.997, 400)
    assert s.update(.996, 500)
    assert not s.update(.98, 600)


def test_probe_is_fixed_and_not_weighted():
    a = probe_indices(352)
    assert len(a) == 32 and len(set(a)) == 32
    np.testing.assert_array_equal(a, probe_indices(352))
    y = np.zeros((4, 2, 32, 32), np.float32)
    y[0] = 2.
    encoded, mask = encode_targets(y)
    p, audit = sampling_probabilities(encoded, mask)
    np.testing.assert_allclose(p, [.775, .075, .075, .075])
    assert audit["wet_case_count"] == 1


def test_fixed_evaluation_preserves_rng_and_training_state():
    torch.set_num_threads(2)
    torch.manual_seed(21)
    model = make_model(RadarConfig(name="dgmr", base_channels=4, input_frames=3, output_frames=2, linear_rate_head=True))
    x = encode_inputs(np.ones((2, 3, 32, 32), np.float32)*.1)
    y, mask = encode_targets(np.ones((2, 2, 32, 32), np.float32)*.2)
    model.train()
    model(x)
    weights = {k: v.clone() for k,v in model.state_dict().items()}
    before = torch.get_rng_state().clone()
    result = fixed_score(model, x, y, mask)
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in weights.items())
    assert result == fixed_score(model, x, y, mask)


def test_small_actual_training_has_separate_final_and_best_states():
    rng = np.random.default_rng(11)
    x = rng.exponential(.2, (4, 3, 32, 32)).astype(np.float32)
    y = rng.exponential(.3, (4, 2, 32, 32)).astype(np.float32)
    ids = np.array(["train_a", "train_b", "train_c", "train_d"])
    data = (x, y, x[:2], y[:2], ids, np.array(["validation_a", "validation_b"]))
    # Use a project-relative temporary folder because result provenance is ROOT-relative.
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="tuning-smoke-", dir=root/"outputs/alta-nowcast") as temporary:
        folder = Path(temporary)
        for name in ["convlstm", "dgmr", "nowcastnet"]:
            d = folder/name
            r = train_run(name, 3e-4, 71, data, d, max_steps=4, eval_every=2,
                          warmup=2, min_steps=4, patience=3, base_channels=4)
            assert r["status"] == "complete" and r["completed_updates"] == 4
            assert (d/"best.pt").exists() and (d/"final.pt").exists()
            assert (d/"best.optimizer.pt").exists() and (d/"final.optimizer.pt").exists()
            assert load_bundle(d/"best.pt")["best_step"] == r["best_step"]
            assert load_bundle(d/"final.pt")["step"] == 4
            resume = torch.load(d/"final.optimizer.pt", weights_only=True)
            assert resume["optimizer"]["state"]
            history = json.loads((d/"history.json").read_text())
            assert sum(row["interval_updates"] for row in history) == 4
            assert all(np.isfinite(row["fixed_train_probe_mse"]) for row in history)
            if name == "nowcastnet":
                assert history[0]["phase"] == "evolution"
                assert not history[0]["validation_eligible_for_selection"]
                assert r["best_step"] > 2


if __name__ == "__main__":
    test_stopping_requires_patience_and_minimum_budget()
    test_probe_is_fixed_and_not_weighted()
    test_fixed_evaluation_preserves_rng_and_training_state()
    test_small_actual_training_has_separate_final_and_best_states()
    print("Four tuning safeguards passed.")
