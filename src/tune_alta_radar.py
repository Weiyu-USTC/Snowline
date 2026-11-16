"""Development-only, bounded learning-rate search for compact Alta nowcasters.

The historical holdout has already been viewed in earlier experiments. It is
never used here; subsequent holdout reporting is a repeated benchmark, not a
new pristine test. Original selected checkpoints are immutable inputs.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from alta_radar_models import (
    RadarConfig, RadarDiscriminator, architecture_manifest, encode_inputs,
    encode_targets, ensemble_pool_loss, make_model, physical_mean_mse,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/alta-nowcast"
OUT = BASE / "tuning-20261107"
NAMES = ("convlstm", "dgmr", "nowcastnet")
RATES = (1e-4, 3e-4, 1e-3)
SEEDS = (20261105, 20261106, 20261107)
PROBE_SEED = 818202611
VALIDATION_NOISE_SEED = 98123
PROBE_NOISE_SEED = 731202611


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def load_training_data():
    with np.load(BASE / "learning_arrays.npz") as z:
        split = z["split"].astype(str)
        eligible = z["eligible"].astype(bool)
        ids = z["case_id"].astype(str)
        winter = z["winter"]
        training = np.flatnonzero(eligible & (split == "train"))
        development = np.flatnonzero(eligible & (split == "validation"))[::4]
        assert set(winter[training]) == {2020, 2021}
        assert set(winter[development]) == {2022}
        assert not set(ids[training]) & set(ids[development])
        x = z["x"]
        y = z["y"]
        result = (x[training], y[training], x[development], y[development],
                  ids[training], ids[development])
    return result


def probe_indices(count, size=32):
    return np.sort(np.random.default_rng(PROBE_SEED).choice(count, min(size, count), replace=False))


def original_hashes():
    return {n: digest(BASE / f"models/selected/{n}.pt") for n in NAMES}


def freeze_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        assert plan["training_source_sha256"] == digest(Path(__file__))
        assert plan["architecture_source_sha256"] == digest(ROOT / "src/alta_radar_models.py")
        assert plan["original_selected_sha256"] == original_hashes()
        return plan
    x, y, vx, vy, ids, vids = load_training_data()
    p = probe_indices(len(x))
    plan = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Bounded validation-only learning-rate tuning; no test-based selection",
        "models": list(NAMES), "initial_learning_rates": list(RATES),
        "screening_seed": SEEDS[0], "replication_seeds": list(SEEDS[1:]),
        "max_updates": 3000, "minimum_updates_before_stopping": 1500,
        "batch_size": 2, "base_channels": 8, "threads_per_worker": 2,
        "maximum_concurrent_workers": 3, "ensemble_training_members": 2,
        "evaluation_frequency_updates": 100,
        "validation_noise_seed": VALIDATION_NOISE_SEED,
        "probe_noise_seed": PROBE_NOISE_SEED, "probe_selection_seed": PROBE_SEED,
        "data_rng_seed": "model seed + 1000000, separate NumPy generator",
        "training_cases": ids.tolist(), "validation_cases": vids.tolist(),
        "fixed_naturally_sampled_training_probe_cases": ids[p].tolist(),
        "training_count": len(ids), "validation_count": len(vids), "probe_count": len(p),
        "validation_metric": "Natural-frequency, valid-pixel-weighted physical rate MSE, two-member mean, fixed noise, all20frames",
        "sampling": "0.3 uniform +0.7 uniform over wet training cases; loss unweighted by sampling probability, matching pilot",
        "optimizer": "Adam betas(0.5,0.999); generator/discriminator initially same LR",
        "scheduler": {"kind": "ReduceLROnPlateau", "factor": .5, "patience": 3,
                      "threshold_relative": .005, "minimum_lr": 1e-6,
                      "monitor": "validation physical rate MSE", "both_optimizers": True},
        "early_stopping": {"patience_checks": 10, "minimum_updates": 1500,
                           "meaningful_relative_improvement": .005},
        "checkpoint_selection": "Lowest exact validation MSE, including improvements smaller than stopping threshold",
        "nowcastnet_evolution_updates": 500,
        "nowcastnet_stage_note": "Warmup fixed at500 regardless of total budget; only generator-phase validation selects checkpoints and schedules LR",
        "family_learning_rate_selection": "Lowest best validation MSE among three screening-seed runs; exact ties favor smaller initial LR",
        "family_comparison": "Mean and sample SD of best validation MSE over the screening seed plus two independent replicate seeds at selected LR",
        "representative_checkpoint": "Median validation-MSE seed among three completed seeds at selected LR; ties favor lower seed",
        "test_used_for_fitting_or_selection": False,
        "holdout_status": "Historical2023/24 test was viewed previously; any new evaluation there is a reused benchmark",
        "learning_arrays_sha256": digest(BASE / "learning_arrays.npz"),
        "training_source_sha256": digest(Path(__file__)),
        "architecture_source_sha256": digest(ROOT / "src/alta_radar_models.py"),
        "original_selected_sha256": original_hashes(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
    }
    atomic_json(path, plan)
    shutil.copy2(Path(__file__), OUT / "tune_alta_radar.executed.py")
    shutil.copy2(ROOT / "src/alta_radar_models.py", OUT / "alta_radar_models.executed.py")
    return plan


@dataclass
class Stopper:
    patience: int = 10
    min_steps: int = 1500
    relative_threshold: float = .005
    best: float = math.inf
    bad_checks: int = 0

    def update(self, score, step):
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation metric")
        if score < self.best * (1 - self.relative_threshold):
            self.best = score
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return step >= self.min_steps and self.bad_checks >= self.patience


@torch.no_grad()
def fixed_score(model, x, y, mask, *, seed=VALIDATION_NOISE_SEED, batch_size=2, evolution=False):
    """Evaluation restores RNG and module training flags exactly."""
    states = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        sq = den = absolute = 0.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for start in range(0, len(x), batch_size):
                bx, by, bm = x[start:start+batch_size], y[start:start+batch_size], mask[start:start+batch_size]
                if evolution:
                    prediction = torch.expm1(model.evolution(bx)[0]).clamp(0, 128)
                else:
                    prediction = torch.stack([torch.expm1(model(bx)).clamp(0, 128) for _ in range(2)]).mean(0)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite fixed-set forecast")
                error = prediction - torch.expm1(by)
                sq += float((error.square()*bm).sum())
                absolute += float((error.abs()*bm).sum())
                den += float(bm.sum())
        return {"mse": sq/max(den, 1), "mae": absolute/max(den, 1), "valid_pixels": den}
    finally:
        for module, training in states:
            module.training = training


def sampling_probabilities(y, mask):
    valid = mask.numpy().astype(bool)
    wet = ((np.expm1(y.numpy()) > .1) & valid).sum(axis=(1, 2, 3))/np.maximum(valid.sum(axis=(1, 2, 3)), 1)
    selected = wet >= .1
    fallback = False
    if not selected.any():
        selected = wet > 0
        fallback = True
    if not selected.any():
        raise ValueError("No wet training targets")
    p = np.full(len(y), .3/len(y), dtype=np.float64)
    p[selected] += .7/selected.sum()
    p /= p.sum()
    return p, {"wet_case_count": int(selected.sum()), "fallback_any_wet": fallback,
               "probabilities": p.tolist(), "wet_pixel_fraction": wet.tolist()}


def checkpoint(path, model, cfg, opt, dopt, discriminator, scheduler, dscheduler, rng, metadata):
    """Save inference-compatible weights and a separately named resumable state."""
    state = {**metadata, "config": asdict(cfg), "state_dict": model.state_dict(),
             "manifest": architecture_manifest(cfg, model)}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    resume = {"optimizer": opt.state_dict(),
              "discriminator_optimizer": None if dopt is None else dopt.state_dict(),
              "discriminator": None if discriminator is None else discriminator.state_dict(),
              "scheduler": scheduler.state_dict(),
              "discriminator_scheduler": None if dscheduler is None else dscheduler.state_dict(),
              "torch_rng_state": torch.get_rng_state(), "numpy_data_rng_state": rng.bit_generator.state}
    torch.save(resume, path.with_suffix(".optimizer.pt"))
    atomic_json(Path(str(path)+".json"), {k: v for k, v in state.items() if k != "state_dict"})


def train_run(name, learning_rate, seed, data, directory, *, max_steps=3000,
              eval_every=100, warmup=500, min_steps=1500, patience=10, base_channels=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"result.json").exists():
        return json.loads((directory/"result.json").read_text())
    x, y, vx, vy, ids, vids = data
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    cfg = RadarConfig(name=name, input_frames=x.shape[1], output_frames=y.shape[1],
                      base_channels=base_channels, steps=max_steps, batch_size=2, seed=seed,
                      learning_rate=learning_rate, threads=2, ensemble_training_members=2,
                      recipe="wet_mse", linear_rate_head=name=="dgmr")
    model = make_model(cfg)
    discriminator = None if name == "convlstm" else RadarDiscriminator(base_channels)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(.5, .999))
    dopt = None if discriminator is None else torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(.5, .999))
    args = dict(mode="min", factor=.5, patience=3, threshold=.005, threshold_mode="rel", min_lr=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **args)
    dscheduler = None if dopt is None else torch.optim.lr_scheduler.ReduceLROnPlateau(dopt, **args)
    xx = encode_inputs(x)
    yy, mm = encode_targets(y)
    vyy, vmm = encode_targets(vy)
    vv = (encode_inputs(vx), vyy, vmm)
    pids = probe_indices(len(xx))
    probe = (xx[pids], yy[pids], mm[pids])
    probabilities, sampling = sampling_probabilities(yy, mm)
    atomic_json(directory/"sampling.json", {**sampling, "case_ids": ids.tolist(), "probe_ids": ids[pids].tolist(), "validation_ids": vids.tolist()})
    rng = np.random.default_rng(seed+1000000)
    stop = Stopper(patience=patience, min_steps=min_steps)
    logs = []
    best = math.inf
    best_step = None
    begin = time.monotonic()
    warm = warmup if name == "nowcastnet" else 0
    interval = []
    stopped = "maximum_updates"
    counts = np.zeros(len(xx), dtype=np.int64)
    def metadata(step):
        return {"training_log": logs, "best_step": best_step, "best_validation_mse": best,
                "step": step, "initial_learning_rate": learning_rate,
                "current_learning_rate": opt.param_groups[0]["lr"],
                "training_samples": len(xx), "validation_samples": len(vv[0]),
                "elapsed_seconds": time.monotonic()-begin, "stopper_state": asdict(stop),
                "training_in_progress": True, "warmup_steps": warm,
                "sampling_seed": seed+1000000, "probe_count": len(pids)}
    for step in range(1, max_steps+1):
        ix = rng.choice(len(xx), min(cfg.batch_size, len(xx)), replace=False, p=probabilities)
        counts[ix] += 1
        bx, by, bm = xx[ix], yy[ix], mm[ix]
        model.train()
        dloss_value = dgrad = 0.
        phase = "evolution" if name == "nowcastnet" and step <= warm else "generator"
        lr_before = opt.param_groups[0]["lr"]
        if phase == "evolution":
            evo, flow = model.evolution(bx)
            smooth = (flow[..., 1:, :]-flow[..., :-1, :]).abs().mean()+(flow[..., :, 1:]-flow[..., :, :-1]).abs().mean()
            loss = physical_mean_mse([evo], by, bm)+.01*smooth
        elif name == "convlstm":
            loss = physical_mean_mse([model(bx)], by, bm)
        else:
            if name == "nowcastnet":
                for parameter in model.evolution.parameters():
                    parameter.requires_grad_(False)
            discriminator.train()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            with torch.no_grad():
                fake = model(bx)*bm
            real_score = discriminator(bx, by*bm, spatial=name=="dgmr")
            fake_score = discriminator(bx, fake, spatial=name=="dgmr")
            dl = F.relu(1-real_score).mean()+F.relu(1+fake_score).mean() if name == "dgmr" else F.softplus(-real_score).mean()+F.softplus(fake_score).mean()
            if not torch.isfinite(dl):
                raise FloatingPointError(f"Nonfinite discriminator loss at{step}")
            dopt.zero_grad()
            dl.backward()
            dgrad = float(torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5, error_if_nonfinite=True))
            dopt.step()
            dloss_value = float(dl.detach())
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            samples = [model(bx) for _ in range(cfg.ensemble_training_members)]
            score = discriminator(bx, samples[0]*bm, spatial=name=="dgmr")
            if name == "dgmr":
                loss = 20*physical_mean_mse(samples, by, bm)-.1*score.mean()
            else:
                loss = 20*physical_mean_mse(samples, by, bm)+5*ensemble_pool_loss(samples, by, bm)+.1*F.softplus(-score).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite generator loss at{step}")
        opt.zero_grad()
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True))
        opt.step()
        interval.append((float(loss.detach()), dloss_value, grad, dgrad))
        if step % eval_every == 0 or step == max_steps or step == warm:
            values = np.asarray(interval)
            validation = fixed_score(model, *vv, evolution=phase=="evolution")
            train_probe = fixed_score(model, *probe, seed=PROBE_NOISE_SEED, evolution=phase=="evolution")
            row = {"step": step, "phase": phase, "interval_updates": len(interval),
                   "interval_mean_objective": float(values[:, 0].mean()),
                   "last_minibatch_objective": float(loss.detach()),
                   "interval_objective_std": float(values[:, 0].std()),
                   "interval_mean_discriminator_objective": float(values[:, 1].mean()),
                   "interval_mean_gradient_norm_before_clipping": float(values[:, 2].mean()),
                   "interval_max_gradient_norm_before_clipping": float(values[:, 2].max()),
                   "last_gradient_norm_before_clipping": grad,
                   "interval_generator_clipping_fraction": float((values[:, 2] > 5).mean()),
                   "interval_mean_discriminator_gradient_norm": float(values[:, 3].mean()),
                   "learning_rate_used": lr_before,
                   "fixed_train_probe_mse": train_probe["mse"],
                   "fixed_train_probe_mae": train_probe["mae"],
                   "validation_mse_mm_h_squared": validation["mse"],
                   "validation_mae_mm_h": validation["mae"],
                   "validation_eligible_for_selection": phase == "generator",
                   "elapsed_seconds": time.monotonic()-begin}
            exact_improvement = False
            should_stop = False
            if phase == "generator":
                exact_improvement = validation["mse"] < best
                if exact_improvement:
                    best, best_step = validation["mse"], step
                scheduler.step(validation["mse"])
                if dscheduler is not None:
                    dscheduler.step(validation["mse"])
                should_stop = stop.update(validation["mse"], step)
            row["next_learning_rate"] = opt.param_groups[0]["lr"]
            row["stopper_bad_checks"] = stop.bad_checks
            logs.append(row)
            atomic_json(directory/"history.json", logs)
            print(json.dumps({"model": name, "initial_lr": learning_rate, "seed": seed, **row}), flush=True)
            if exact_improvement:
                checkpoint(directory/"best.pt", model, cfg, opt, dopt, discriminator, scheduler, dscheduler, rng, metadata(step))
            interval = []
            if should_stop:
                stopped = "validation_plateau_patience"
                break
    if best_step is None:
        raise RuntimeError("Training did not reach a selectable generator checkpoint")
    final_meta = {**metadata(step), "training_in_progress": False, "stop_reason": stopped}
    checkpoint(directory/"final.pt", model, cfg, opt, dopt, discriminator, scheduler, dscheduler, rng, final_meta)
    # Best file has its true optimizer/RNG state from its selected update. Its
    # sidecar remains an honest checkpoint-time snapshot; result holds final run status.
    result = {"model": name, "seed": seed, "initial_learning_rate": learning_rate,
              "final_learning_rate": opt.param_groups[0]["lr"], "best_step": best_step,
              "best_validation_mse": best, "final_validation_mse": logs[-1]["validation_mse_mm_h_squared"],
              "final_train_probe_mse": logs[-1]["fixed_train_probe_mse"],
              "completed_updates": step, "stop_reason": stopped,
              "elapsed_seconds": time.monotonic()-begin, "status": "complete",
              "best_checkpoint": str((directory/"best.pt").relative_to(ROOT)),
              "best_checkpoint_sha256": digest(directory/"best.pt"),
              "final_checkpoint": str((directory/"final.pt").relative_to(ROOT)),
              "final_checkpoint_sha256": digest(directory/"final.pt"),
              "training_draw_counts": counts.tolist(), "unique_training_cases_drawn": int((counts > 0).sum()),
              "test_used_for_fitting_or_selection": False}
    atomic_json(directory/"result.json", result)
    print(json.dumps({"run_finished": result}), flush=True)
    return result


def run_id(rate, seed):
    return f"lr{rate:.0e}_seed{seed}"


def worker(name):
    plan = freeze_plan()
    data = load_training_data()
    family_dir = OUT/name
    family_dir.mkdir(exist_ok=True)
    records = []
    def execute(rate, seed, stage):
        folder = family_dir/run_id(rate, seed)
        try:
            result = train_run(name, rate, seed, data, folder)
        except Exception as error:
            result = {"model": name, "seed": seed, "initial_learning_rate": rate,
                      "status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
            folder.mkdir(parents=True, exist_ok=True)
            atomic_json(folder/"failure.json", result)
            print(json.dumps(result), flush=True)
        result = {**result, "stage": stage}
        records.append(result)
        atomic_json(family_dir/"runs.json", records)
        return result
    for rate in RATES:
        execute(rate, SEEDS[0], "learning_rate_screen")
    valid = [r for r in records if r["status"] == "complete"]
    if not valid:
        raise RuntimeError(f"All learning rate candidates failed for{name}")
    selected = min(valid, key=lambda r: (r["best_validation_mse"], r["initial_learning_rate"]))
    atomic_json(family_dir/"learning_rate_selection.json", {"selected": selected, "candidates": records,
                "criterion": plan["family_learning_rate_selection"], "selected_before_replication": True,
                "test_used": False})
    for seed in SEEDS[1:]:
        execute(selected["initial_learning_rate"], seed, "seed_replication")
    same_rate = [r for r in records if r["status"] == "complete" and r["initial_learning_rate"] == selected["initial_learning_rate"]]
    values = np.array([r["best_validation_mse"] for r in same_rate])
    representative = sorted(same_rate, key=lambda r: (r["best_validation_mse"], r["seed"]))[len(same_rate)//2]
    result = {"model": name, "selected_initial_learning_rate": selected["initial_learning_rate"],
              "representative_checkpoint": representative["best_checkpoint"],
              "representative_checkpoint_sha256": representative["best_checkpoint_sha256"],
              "representative_seed": representative["seed"],
              "representative_best_step": representative["best_step"],
              "mean_best_validation_mse": float(values.mean()),
              "sd_best_validation_mse": float(values.std(ddof=1)) if len(values)>1 else None,
              "replicate_count": len(values), "runs": records, "test_used_for_selection": False,
              "original_checkpoints_unchanged": original_hashes() == plan["original_selected_sha256"]}
    assert result["original_checkpoints_unchanged"]
    atomic_json(family_dir/"family_result.json", result)


def launch_all():
    plan = freeze_plan()
    workers = []
    for name in NAMES:
        log = (OUT/f"{name}.log").open("a")
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--model", name],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        workers.append((name, process, log))
    atomic_json(OUT/"workers.json", {n: p.pid for n,p,_ in workers})
    results = {}
    while len(results) < len(workers):
        for name, process, log in workers:
            if name not in results and process.poll() is not None:
                results[name] = process.returncode
                log.close()
                atomic_json(OUT/"worker_status.json", results)
        time.sleep(2)
    if any(results.values()):
        raise SystemExit(1)
    families = [json.loads((OUT/n/"family_result.json").read_text()) for n in NAMES]
    # Select by replication-average validation performance, never historical test.
    selected = min(families, key=lambda f: f["mean_best_validation_mse"])
    atomic_json(OUT/"selection.json", {"families": families, "selected_family": selected["model"],
                "selected_initial_learning_rate": selected["selected_initial_learning_rate"],
                "representative_checkpoint": selected["representative_checkpoint"],
                "selection_criterion": plan["family_comparison"], "test_used_for_selection": False,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "original_checkpoints_unchanged": original_hashes() == plan["original_selected_sha256"]})
    model_map = {}
    for family in families:
        model_map[family["model"]] = {
            "selected_initial_lr": family["selected_initial_learning_rate"],
            "representative_checkpoint": family["representative_checkpoint"],
            "representative_checkpoint_sha256": family["representative_checkpoint_sha256"],
            "representative_seed": family["representative_seed"],
            "representative_best_step": family["representative_best_step"],
            "mean_best_validation_mse": family["mean_best_validation_mse"],
            "std_best_validation_mse": family["sd_best_validation_mse"],
            "replicate_count": family["replicate_count"],
            "runs": [r for r in family["runs"] if r["status"] == "complete" and
                     r["initial_learning_rate"] == family["selected_initial_learning_rate"]],
        }
    atomic_json(OUT/"final_selection.json", {
        "models": model_map, "winner_by_mean_validation": selected["model"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": digest(OUT/"plan.json"), "test_used_for_selection": False,
        "original_checkpoints_unchanged": original_hashes() == plan["original_selected_sha256"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=NAMES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        print(json.dumps(freeze_plan(), indent=2))
    elif args.all:
        launch_all()
    elif args.model:
        worker(args.model)
    else:
        parser.error("Choose --all, --model or --freeze-only")
