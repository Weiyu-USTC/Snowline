"""Orchestration-only acceleration of frozen Alta training functions.

Wait for the original DGMR worker to finish its first complete run, stop it,
preserve any just-started next run as an interrupted artifact, then distribute
the remaining exact trials into unused slots. No numerical training source,
sampling, stopping criterion, model, rate or seed changes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import tune_alta_radar as tune


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def trial(rate, seed):
    tune.freeze_plan()
    folder = tune.OUT/"dgmr"/tune.run_id(rate, seed)
    try:
        tune.train_run("dgmr", rate, seed, tune.load_training_data(), folder)
    except Exception as error:
        folder.mkdir(parents=True, exist_ok=True)
        tune.atomic_json(folder/"failure.json", {
            "status": "failed", "model": "dgmr", "initial_learning_rate": rate,
            "seed": seed, "error": repr(error), "traceback": traceback.format_exc()})
        raise


def orchestrate():
    plan = tune.freeze_plan()
    original_workers = json.loads((tune.OUT/"workers.json").read_text())
    amendment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reuse free CPU slots; no scientific-plan or numerical-training change",
        "training_source_sha256": tune.digest(tune.ROOT/"src/tune_alta_radar.py"),
        "architecture_source_sha256": tune.digest(tune.ROOT/"src/alta_radar_models.py"),
        "plan_sha256": tune.digest(tune.OUT/"plan.json"),
        "orchestration_source_sha256": tune.digest(Path(__file__)),
        "original_workers": original_workers,
        "maximum_concurrent_training_workers": 3,
        "handoff": "Stop original DGMR worker immediately after first completed result; archive any just-started next run and rerun that exact trial from its original seed",
        "rate_selection": plan["family_learning_rate_selection"],
        "replicates": "Only after all three rate screens complete; same two predeclared seeds",
        "incomplete_replicates": "Do not publish final family ranking unless all three predeclared seeds completed for every family",
        "test_used": False,
    }
    tune.atomic_json(tune.OUT/"orchestration_amendment.json", amendment)
    first = tune.OUT/"dgmr"/tune.run_id(tune.RATES[0], tune.SEEDS[0])/"result.json"
    while not first.exists():
        if not alive(original_workers["dgmr"]):
            raise RuntimeError("Original DGMR worker ended before its first complete result")
        time.sleep(.5)
    pid = original_workers["dgmr"]
    if alive(pid):
        os.kill(pid, signal.SIGSTOP)
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        for _ in range(40):
            if not alive(pid):
                break
            time.sleep(.25)
        if alive(pid):
            raise RuntimeError("Could not verify original DGMR worker exit")
    archived = []
    for directory in sorted((tune.OUT/"dgmr").glob("lr*_seed*")):
        if directory.is_dir() and not (directory/"result.json").exists():
            destination = tune.OUT/"interrupted-orchestration"/directory.name
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                raise RuntimeError("Interrupted-artifact destination already exists")
            directory.rename(destination)
            archived.append(str(destination.relative_to(tune.ROOT)))
    amendment["handoff_completed_utc"] = datetime.now(timezone.utc).isoformat()
    amendment["archived_partial_runs"] = archived
    tune.atomic_json(tune.OUT/"orchestration_amendment.json", amendment)
    print(json.dumps({"handoff_complete": True, "archived_partial_runs": archived}), flush=True)
    foreign = [original_workers["convlstm"], original_workers["nowcastnet"]]

    def execute_queue(queue):
        running = []
        while queue or running:
            for item in list(running):
                rate, seed, process, log = item
                if process.poll() is not None:
                    log.close()
                    running.remove(item)
                    if process.returncode:
                        raise RuntimeError(f"DGMR trial failed: LR{rate}, seed{seed}; preserved logs, no ranking")
            slots = 3 - sum(alive(pid) for pid in foreign) - len(running)
            while queue and slots > 0:
                rate, seed = queue.pop(0)
                result = tune.OUT/"dgmr"/tune.run_id(rate, seed)/"result.json"
                if result.exists():
                    continue
                log = (tune.OUT/f"dgmr_{tune.run_id(rate, seed)}.log").open("w")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--trial",
                                            "--rate", str(rate), "--seed", str(seed)],
                                           cwd=tune.ROOT, stdout=log, stderr=subprocess.STDOUT)
                running.append((rate, seed, process, log))
                print(json.dumps({"started_trial": {"rate": rate, "seed": seed, "pid": process.pid}}), flush=True)
                slots -= 1
            tune.atomic_json(tune.OUT/"orchestration_status.json", {
                "running": [{"rate": r, "seed": s, "pid": p.pid} for r,s,p,_ in running],
                "queued": [{"rate": r, "seed": s} for r,s in queue],
                "original_other_workers_alive": [p for p in foreign if alive(p)],
            })
            time.sleep(2)

    execute_queue([(rate, tune.SEEDS[0]) for rate in tune.RATES[1:]])
    screening = [json.loads((tune.OUT/"dgmr"/tune.run_id(rate, tune.SEEDS[0])/"result.json").read_text()) for rate in tune.RATES]
    chosen = min(screening, key=lambda r: (r["best_validation_mse"], r["initial_learning_rate"]))
    tune.atomic_json(tune.OUT/"dgmr_parallel_rate_selection.json", {
        "selected": chosen, "candidates": screening,
        "selected_before_replication": True, "test_used": False,
        "frozen_utc": datetime.now(timezone.utc).isoformat()})
    execute_queue([(chosen["initial_learning_rate"], seed) for seed in tune.SEEDS[1:]])
    # All five DGMR fits now exist; the unchanged family worker only collates
    # them and verifies original hashes, without fitting a duplicate run.
    tune.worker("dgmr")
    while any(alive(p) for p in foreign):
        time.sleep(2)
    families = [json.loads((tune.OUT/n/"family_result.json").read_text()) for n in tune.NAMES]
    if any(f["replicate_count"] != 3 for f in families):
        raise RuntimeError("A family lacks all three predeclared complete seeds; final selection withheld")
    # Replay only the unchanged collation path; every result.json is complete.
    # Preserve the intentionally stopped original controller's status first.
    old_status = tune.OUT/"worker_status.json"
    if old_status.exists():
        tune.atomic_json(tune.OUT/"original_worker_status_before_parallel_handoff.json", json.loads(old_status.read_text()))
    tune.launch_all()
    amendment["completed_utc"] = datetime.now(timezone.utc).isoformat()
    amendment["original_checkpoints_unchanged"] = tune.original_hashes() == plan["original_selected_sha256"]
    tune.atomic_json(tune.OUT/"orchestration_amendment.json", amendment)
    print(json.dumps({"status": "all15_runs_complete", "final_selection": str(tune.OUT/"final_selection.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--rate", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.trial:
        trial(args.rate, args.seed)
    else:
        orchestrate()
