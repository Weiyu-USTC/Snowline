"""Execute the frozen Alta radar training and save unscored holdout predictions.

Run --all to launch the three CPU workers concurrently. Each worker uses two
threads, the same predeclared split/configuration and its own durable logs.
No test losses, plot selection, or hyperparameter search occur in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from alta_radar_models import train_model,predict_ensemble,load_bundle,MODEL_LABELS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/alta-nowcast"
MODELS=OUT/"models"
PRED=OUT/"radar_predictions"
NAMES=("convlstm","dgmr","nowcastnet")


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(8*1024**2),b""):h.update(chunk)
    return h.hexdigest()


def worker(name,predict_only=False,recipe="uniform_l1",train_only=False):
    if recipe=="wet_mse_linear_head" and name!="dgmr":
        raise ValueError("The physical-head correction applies only to DGMR")
    model_dir=MODELS/("selected" if recipe=="selected" else f"recipe_{recipe}")
    model_dir.mkdir(parents=True,exist_ok=True);PRED.mkdir(parents=True,exist_ok=True)
    plan_path=OUT/"frozen_plan.json";array_path=OUT/"learning_arrays.npz"
    plan=json.loads(plan_path.read_text());params=plan["training"]
    with np.load(array_path) as z:
        # Membership is determined only by the frozen time split and radar
        # completeness/quality. Snowfall magnitudes are never inspected.
        ids=z["case_id"].astype(str);split=z["split"].astype(str)
        winter=z["winter"];eligible=z["eligible"].astype(bool)
        x=z["x"];y=z["y"]
    train_index=np.flatnonzero(eligible&(split=="train"))
    dev_index=np.flatnonzero(eligible&(split=="validation"))
    selection_index=dev_index[::4]
    forecast_index=np.flatnonzero(eligible&np.isin(split,["validation","test"]))
    assert len(train_index)>0 and len(selection_index)>0 and len(forecast_index)>0
    assert set(winter[train_index])<=set(plan["training_winters"])
    assert set(winter[selection_index])=={plan["development_winter"]}
    assert not set(ids[train_index])&set(ids[selection_index])
    assert not set(ids[train_index])&set(ids[forecast_index])
    audit={"model":name,"label":MODEL_LABELS[name],
        "plan_sha256":digest(plan_path),"learning_arrays_sha256":digest(array_path),
        "implementation_sha256":digest(ROOT/"src/alta_radar_models.py"),
        "orchestration_sha256":digest(Path(__file__)),
        "training_cases":ids[train_index].tolist(),
        "checkpoint_selection_cases":ids[selection_index].tolist(),
        "prediction_cases":ids[forecast_index].tolist(),
        "training_count":len(train_index),"selection_count":len(selection_index),
        "prediction_count":len(forecast_index),"test_used_for_fitting_or_selection":False,
        "hyperparameters":params,"seed":plan["seed"],"cpu_threads":2}
    audit["recipe"]=recipe
    (model_dir/f"{name}_split_audit.json").write_text(json.dumps(audit,indent=2))
    checkpoint=model_dir/f"{name}.pt"
    if predict_only:
        bundle=load_bundle(checkpoint)
        assert bundle["config"].name==name
    else:
        bundle=train_model(name,x[train_index],y[train_index],
            validation=(x[selection_index],y[selection_index]),
            steps=params["steps"],batch_size=params["batch_size"],
            base_channels=params["base_channels"],seed=plan["seed"],
            learning_rate=params["learning_rate"],threads=2,
            checkpoint_path=checkpoint,log_every=params["checkpoint_frequency_steps"],
            recipe="wet_mse" if recipe=="wet_mse_linear_head" else recipe,
            linear_rate_head=recipe=="wet_mse_linear_head")
    if train_only:
        print(json.dumps({"model":name,"recipe":recipe,"status":"training_complete",
                          "best_step":bundle["best_step"]}),flush=True)
        return
    # The large tensors stay memory-mappable for downstream metric computation.
    # Preserve float32 so small stochastic differences are not quantized away.
    members=params["ensemble_members"]
    shape=(len(forecast_index),members,y.shape[1],*y.shape[-2:])
    predictions_path=PRED/f"{name}.npy"
    arr=np.lib.format.open_memmap(predictions_path,mode="w+",dtype=np.float32,shape=shape)
    num_values=zero_values=0;rate_sum=spread_sum=0.;spread_values=0;begin=time.monotonic()
    # Predict in bounded batches with different fixed seeds; persist input order.
    for start in range(0,len(forecast_index),4):
        ix=forecast_index[start:start+4]
        forecast=predict_ensemble(bundle,x[ix],members=members,batch_size=4,
                                  seed=plan["seed"]+100000+start)
        if not np.isfinite(forecast).all() or forecast.min()<0:
            raise FloatingPointError(f"Invalid predictions from {name} at {start}")
        num_values+=forecast.size;zero_values+=int((forecast==0).sum())
        rate_sum+=float(forecast.sum(dtype=np.float64))
        spread=forecast.std(axis=1)
        spread_sum+=float(spread.sum(dtype=np.float64));spread_values+=spread.size
        arr[start:start+len(ix)]=forecast
        if start%40==0:
            arr.flush()
            print(json.dumps({"model":name,"prediction_cases_done":start+len(ix),
                              "prediction_cases_total":len(forecast_index),
                              "elapsed_seconds":time.monotonic()-begin}),flush=True)
    arr.flush();del arr
    np.savez_compressed(PRED/f"{name}_index.npz",case_id=ids[forecast_index],
                        source_index=forecast_index,split=split[forecast_index],winter=winter[forecast_index])
    summary={**audit,"checkpoint":str(checkpoint.relative_to(ROOT)),
             "checkpoint_sha256":digest(checkpoint),"best_step":bundle["best_step"],
             "prediction_file":str(predictions_path.relative_to(ROOT)),
             "prediction_sha256":digest(predictions_path),"prediction_shape":list(shape),
             "prediction_dtype":"float32 (lossless float32 inference storage)","units":"mm/hour liquid equivalent",
             "prediction_seed_scheme":"seed +100000 +batch_start; sequential batches of4 cases",
             "mean_predicted_rate_mm_h":rate_sum/max(num_values,1),
             "mean_ensemble_spread_mm_h":spread_sum/max(spread_values,1),
             "fraction_exact_zero_predictions":zero_values/max(num_values,1),
             "prediction_seconds":time.monotonic()-begin,
             "training_seconds":bundle["elapsed_seconds"],"status":"complete"}
    (PRED/f"{name}_manifest.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"model":name,"status":"complete","best_step":bundle["best_step"],
                      "prediction_shape":list(shape)}),flush=True)


def all_workers(predict_only=False,recipe="uniform_l1",train_only=False):
    model_dir=MODELS/("selected" if recipe=="selected" else f"recipe_{recipe}")
    model_dir.mkdir(parents=True,exist_ok=True)
    processes=[]
    for name in (("dgmr",) if recipe=="wet_mse_linear_head" else NAMES):
        log=(model_dir/f"{name}.log").open("a" if predict_only else "w")
        cmd=[sys.executable,str(Path(__file__).resolve()),"--model",name,"--recipe",recipe]
        if predict_only:cmd.append("--predict-only")
        if train_only:cmd.append("--train-only")
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        processes.append((name,p,log));print(f"Started {name} pid={p.pid}",flush=True)
    results={}
    while len(results)<len(processes):
        for name,p,log in processes:
            if name not in results and p.poll() is not None:
                results[name]=p.returncode;log.close()
                print(json.dumps({"model":name,"exit_code":p.returncode}),flush=True)
        time.sleep(2)
    (model_dir/"worker_status.json").write_text(json.dumps(results,indent=2))
    if any(results.values()):raise SystemExit(1)


def select_recipes():
    """Choose between two recorded recipes using development scores only."""
    selected_dir=MODELS/"selected";selected_dir.mkdir(exist_ok=True)
    records=[]
    for name in NAMES:
        candidates=[]
        recipes=[("uniform_l1",MODELS/"recipe_uniform_l1"),("wet_mse",MODELS/"recipe_wet_mse")]
        if name=="dgmr":recipes.append(("wet_mse_linear_head",MODELS/"recipe_wet_mse_linear_head"))
        for recipe,directory in recipes:
            checkpoint=directory/f"{name}.pt"
            metadata=json.loads(Path(str(checkpoint)+".json").read_text())
            rows=[r for r in metadata["training_log"] if "validation_mse_mm_h_squared" in r]
            best=min(rows,key=lambda r:r["validation_mse_mm_h_squared"])
            assert best["step"]==metadata["best_step"]
            logged=[json.loads(line) for line in (directory/f"{name}.log").read_text().splitlines()
                    if line.startswith('{')]
            logged_steps=[row["step"] for row in logged if "step" in row]
            candidates.append({"recipe":recipe,"checkpoint":str(checkpoint),
                               "development_mse":best["validation_mse_mm_h_squared"],
                               "best_step":best["step"],"checkpoint_sha256":digest(checkpoint),
                               "requested_steps":metadata["config"]["steps"],
                               "completed_logged_steps":max(logged_steps),
                               "training_finished":not metadata.get("training_in_progress",False)})
        chosen=min(candidates,key=lambda c:c["development_mse"])
        destination=selected_dir/f"{name}.pt"
        shutil.copy2(chosen["checkpoint"],destination)
        shutil.copy2(chosen["checkpoint"]+".json",str(destination)+".json")
        records.append({"model":name,"candidates":candidates,"selected":chosen,
                        "selected_checkpoint":str(destination),"test_score_used":False})
    (selected_dir/"recipe_selection.json").write_text(json.dumps(records,indent=2))
    # Keep the first-recipe prediction fields separately; subsequent public
    # comparison consumes only the newly selected predictions at the top level.
    archive=PRED/"recipe_uniform_l1";archive.mkdir(exist_ok=True)
    for name in NAMES:
        for suffix in [".npy","_index.npz","_manifest.json"]:
            source=PRED/(name+suffix)
            if source.exists():shutil.move(str(source),str(archive/source.name))
    print(json.dumps(records,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",choices=NAMES)
    p.add_argument("--all",action="store_true");p.add_argument("--predict-only",action="store_true")
    p.add_argument("--train-only",action="store_true");p.add_argument("--select-recipes",action="store_true")
    p.add_argument("--recipe",choices=["uniform_l1","wet_mse","wet_mse_linear_head","selected"],default="uniform_l1")
    a=p.parse_args()
    if a.select_recipes:select_recipes()
    elif a.all:all_workers(a.predict_only,a.recipe,a.train_only)
    elif a.model:worker(a.model,a.predict_only,a.recipe,a.train_only)
    else:p.error("Choose --all or --model")
