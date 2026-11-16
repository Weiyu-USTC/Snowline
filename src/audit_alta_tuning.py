"""Integrity audit of completed bounded tuning; no new scores or fitting."""
import json
from pathlib import Path
import math

import numpy as np
import torch

import tune_alta_radar as tune


def main():
    out = tune.OUT
    plan = json.loads((out/"plan.json").read_text())
    choice = json.loads((out/"final_selection.json").read_text())
    assert tune.digest(out/"plan.json") == choice["plan_sha256"]
    assert tune.digest(tune.ROOT/"src/tune_alta_radar.py") == plan["training_source_sha256"]
    assert tune.digest(out/"tune_alta_radar.executed.py") == plan["training_source_sha256"]
    assert tune.digest(tune.ROOT/"src/alta_radar_models.py") == plan["architecture_source_sha256"]
    assert tune.digest(out/"alta_radar_models.executed.py") == plan["architecture_source_sha256"]
    assert tune.digest(tune.BASE/"learning_arrays.npz") == plan["learning_arrays_sha256"]
    assert tune.original_hashes() == plan["original_selected_sha256"]
    audit = {"plan_sha256": choice["plan_sha256"], "original_checkpoints_unchanged": True,
             "frozen_numerical_source_unchanged": True, "test_evaluation_performed": False,
             "runs": [], "families": {}}
    for name in tune.NAMES:
        records = []
        for result_path in sorted((out/name).glob("*/result.json")):
            directory = result_path.parent
            result = json.loads(result_path.read_text())
            logs = json.loads((directory/"history.json").read_text())
            selected_logs = [r for r in logs if r["validation_eligible_for_selection"]]
            best_row = min(selected_logs, key=lambda r: r["validation_mse_mm_h_squared"])
            assert result["status"] == "complete"
            assert result["best_step"] == best_row["step"]
            assert result["best_validation_mse"] == best_row["validation_mse_mm_h_squared"]
            assert result["completed_updates"] == logs[-1]["step"]
            assert 1500 <= result["completed_updates"] <= 3000
            assert sum(r["interval_updates"] for r in logs) == result["completed_updates"]
            assert sum(result["training_draw_counts"]) == 2*result["completed_updates"]
            assert all(math.isfinite(r[k]) for r in logs for k in
                       ["validation_mse_mm_h_squared", "fixed_train_probe_mse",
                        "interval_mean_objective", "last_gradient_norm_before_clipping"])
            sample = json.loads((directory/"sampling.json").read_text())
            assert sample["case_ids"] == plan["training_cases"]
            assert sample["validation_ids"] == plan["validation_cases"]
            assert sample["probe_ids"] == plan["fixed_naturally_sampled_training_probe_cases"]
            for kind, step in [("best", result["best_step"]), ("final", result["completed_updates"])]:
                assert tune.digest(directory/f"{kind}.pt") == result[f"{kind}_checkpoint_sha256"]
                meta = json.loads((directory/f"{kind}.pt.json").read_text())
                assert meta["step"] == step
                assert meta["config"]["seed"] == result["seed"]
                assert meta["config"]["learning_rate"] == result["initial_learning_rate"]
                saved = torch.load(directory/f"{kind}.optimizer.pt", map_location="cpu", weights_only=True)
                assert saved["optimizer"]["state"]
                assert saved["torch_rng_state"].numel() > 0
                assert saved["numpy_data_rng_state"]["bit_generator"] == "PCG64"
                if name != "convlstm":
                    assert saved["discriminator_optimizer"]["state"]
                    assert saved["discriminator"]
            records.append(result)
            audit["runs"].append({"model": name, "seed": result["seed"],
                "initial_learning_rate": result["initial_learning_rate"],
                "completed_updates": result["completed_updates"], "best_step": result["best_step"],
                "best_validation_mse": result["best_validation_mse"], "all_checks_passed": True})
        assert len(records) == 5
        screen = [r for r in records if r["seed"] == tune.SEEDS[0]]
        assert sorted(r["initial_learning_rate"] for r in screen) == list(tune.RATES)
        winning = min(screen, key=lambda r: (r["best_validation_mse"], r["initial_learning_rate"]))
        rate = winning["initial_learning_rate"]
        replicates = [r for r in records if r["initial_learning_rate"] == rate]
        assert sorted(r["seed"] for r in replicates) == list(tune.SEEDS)
        representative = sorted(replicates, key=lambda r: (r["best_validation_mse"], r["seed"]))[1]
        model_choice = choice["models"][name]
        values = [r["best_validation_mse"] for r in replicates]
        assert model_choice["selected_initial_lr"] == rate
        assert model_choice["representative_seed"] == representative["seed"]
        assert model_choice["representative_checkpoint_sha256"] == representative["best_checkpoint_sha256"]
        assert np.isclose(model_choice["mean_best_validation_mse"], np.mean(values), rtol=0, atol=1e-15)
        assert np.isclose(model_choice["std_best_validation_mse"], np.std(values, ddof=1), rtol=0, atol=1e-15)
        audit["families"][name] = {"selected_initial_lr": rate, "representative_seed": representative["seed"],
                                  "mean_best_validation_mse": float(np.mean(values)), "all_checks_passed": True}
    assert len(audit["runs"]) == 15
    winner = min(audit["families"], key=lambda n: audit["families"][n]["mean_best_validation_mse"])
    assert winner == choice["winner_by_mean_validation"]
    audit["winner_by_mean_validation"] = winner
    audit["total_optimizer_updates"] = sum(r["completed_updates"] for r in audit["runs"])
    audit["all_checks_passed"] = True
    tune.atomic_json(out/"integrity_audit.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ["runs", "families"]}, indent=2))


if __name__ == "__main__":
    main()
