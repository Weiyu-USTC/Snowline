"""Evaluate frozen selected Alta checkpoints on a common physical rate loss.

This creates a post-training diagnostic, never fits or selects model weights.
The all-grid quality mask and two-member evaluation match _validation_score.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from alta_radar_models import encode_inputs, encode_targets, load_bundle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/alta-nowcast"
OUT = SOURCE / "learning-review"
NAMES = ("convlstm", "dgmr", "nowcastnet")
LABELS = {"convlstm": "ConvLSTM", "dgmr": "DGMR adaptation", "nowcastnet": "NowcastNet adaptation"}
SEED = 98123
BATCH_SIZE = 2
MEMBERS = 2


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def rate_errors(model, x, y, index):
    model.eval()
    squared_sum = absolute_sum = denominator = 0.0
    started = time.monotonic()
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        for start in range(0, len(index), BATCH_SIZE):
            batch = index[start:start+BATCH_SIZE]
            bx = encode_inputs(x[batch])
            by, bm = encode_targets(y[batch])
            # Deliberately preserve the exact validation aggregation and cap.
            prediction = torch.stack([
                torch.expm1(model(bx)).clamp(0, 128) for _ in range(MEMBERS)
            ]).mean(0)
            error = prediction - torch.expm1(by)
            squared_sum += float(((error**2) * bm).sum())
            absolute_sum += float((error.abs() * bm).sum())
            denominator += float(bm.sum())
    if denominator <= 0:
        raise ValueError("No valid verification pixels")
    return {
        "rate_mse_mm_h_squared": squared_sum / denominator,
        "rate_mae_mm_h": absolute_sum / denominator,
        "valid_rate_pixels": int(denominator),
        "squared_error_sum": squared_sum,
        "absolute_error_sum": absolute_sum,
        "evaluation_seconds": time.monotonic() - started,
    }


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    array_path = SOURCE / "learning_arrays.npz"
    with np.load(array_path) as data:
        x, y = data["x"], data["y"]
        ids = data["case_id"].astype(str)
        split = data["split"].astype(str)
        eligible = data["eligible"].astype(bool)
    train = np.flatnonzero(eligible & (split == "train"))
    development = np.flatnonzero(eligible & (split == "validation"))
    test = np.flatnonzero(eligible & (split == "test"))
    groups = {
        "train": train,
        "development_selection": development[::4],
        "development_all": development,
        "test": test,
    }
    assert [len(groups[k]) for k in groups] == [352, 42, 167, 180]
    assert not (set(ids[train]) & set(ids[development]))
    assert not (set(ids[train]) & set(ids[test]))
    assert not (set(ids[development]) & set(ids[test]))
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Post-training common-loss diagnostic of already selected frozen weights; no retraining or selection",
        "learning_arrays_sha256": digest(array_path),
        "implementation_sha256": digest(ROOT / "src/alta_radar_models.py"),
        "diagnostic_source_sha256": digest(Path(__file__)),
        "seed": SEED,
        "ensemble_members": MEMBERS,
        "batch_size": BATCH_SIZE,
        "cpu_threads": 2,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "prediction": "Mean of two physical expm1(model(encoded_input)) predictions, each clipped to [0,128] mm/hour",
        "truth": "expm1(log1p(clipped target rate)); finite nonnegative target mask from encode_targets",
        "aggregation": "Sum of squared/absolute errors over all valid pixels in all20 output frames and all64x64 grid cells, divided by valid pixel count",
        "weights": "Natural eligible-case frequency; no wet-case oversampling, adversarial terms, log-space stabilizer, or maxpool regularizer",
        "split_note": "The42 checkpoint-selection cases are every fourth of167 development cases; they are not a separate independent split",
        "metric_difference": "This is rate MSE in (mm/hour)^2. Existing short_summary.csv evaluates accumulated liquid mm at1/2/3h, excludes grid borders, and averages per-case errors with eight-member predictions",
        "rng_note": "Seed reset separately for every split and model, matching checkpoint-selection validation noise; finite two-member stochastic prediction noise remains",
        "case_ids": {key: ids[index].tolist() for key, index in groups.items()},
        "case_counts": {key: len(index) for key, index in groups.items()},
        "models": {},
    }
    rows = []
    for name in NAMES:
        checkpoint = SOURCE / "models/selected" / (name + ".pt")
        before_hash = digest(checkpoint)
        bundle = load_bundle(checkpoint)
        selected_step = bundle["best_step"]
        selection_expected = next(
            row["validation_mse_mm_h_squared"]
            for row in bundle["training_log"] if row["step"] == selected_step
        )
        audit["models"][name] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": before_hash,
            "selected_step": selected_step,
            "saved_selected_development_mse": selection_expected,
            "config": vars(bundle["config"]),
        }
        for group, index in groups.items():
            errors = rate_errors(bundle["model"], x, y, index)
            row = {"model_id": name, "model": LABELS[name], "split": group,
                   "cases": len(index), "selected_step": selected_step, **errors}
            if group == "development_selection":
                delta = errors["rate_mse_mm_h_squared"] - selection_expected
                audit["models"][name]["reproduced_development_mse_difference"] = delta
                if not np.isclose(errors["rate_mse_mm_h_squared"], selection_expected, rtol=2e-5, atol=1e-7):
                    raise AssertionError(f"{name} validation MSE not reproduced: {errors} vs{selection_expected}")
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in {"squared_error_sum", "absolute_error_sum"}}), flush=True)
        audit["models"][name]["checkpoint_unchanged"] = digest(checkpoint) == before_hash
        assert audit["models"][name]["checkpoint_unchanged"]
        del bundle
    audit["elapsed_seconds"] = time.monotonic() - started
    audit["validation_reproduction_passed"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "checkpoint_losses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "checkpoint_loss_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({"status": "complete", "elapsed_seconds": audit["elapsed_seconds"], "output": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
