import argparse
import os

import numpy as np
import yaml
from ase.io.trajectory import Trajectory
from rich.progress import track

from evaluate import GenEval, get_validity_one, smact_validity


def resolve_path(path: str | None, base_dir: str) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ["main_path", "traj_train", "traj_test"]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = resolve_path(cfg[key], base_dir)
    return cfg


def collect_sorted_traj_names(folder: str, prefix: str) -> list[str]:
    names = [name for name in os.listdir(folder) if name.startswith(prefix) and name.endswith(".traj")]
    if prefix == "gen":
        return sorted(names, key=lambda x: int(x[3:-5]))
    return sorted(names, key=lambda x: int(x[6:-5]))


def mark_constructed(atoms) -> None:
    try:
        pos = atoms.get_scaled_positions()
        cell = np.array(atoms.get_cell())
        atoms.info["constructed"] = bool(
            np.all(np.isnan(pos) == False) and np.all(np.isnan(cell) == False) and atoms.get_volume() > 0.1
        )
    except Exception:
        atoms.info["constructed"] = False


def preprocess_pairwise_traj(main_path: str, gentraj: str, targettraj: str) -> tuple[str, str]:
    gen_id = gentraj[3:-5]
    pred_path = os.path.join(main_path, f"pred{gen_id}.traj")
    true_path = os.path.join(main_path, f"true{gen_id}.traj")

    if os.path.exists(pred_path):
        return pred_path, true_path

    unrelax = Trajectory(os.path.join(main_path, "gen", gentraj))
    relax = Trajectory(os.path.join(main_path, "target", targettraj))
    traj_pred = Trajectory(pred_path, mode="a")
    traj_true = Trajectory(true_path, mode="a")

    for i in track(range(len(unrelax)), description=f"Preprocessing {gentraj}"):
        ats_pred = unrelax[i]
        ats_true = relax[i]
        mark_constructed(ats_pred)
        mark_constructed(ats_true)
        smact_validity(ats_pred)
        get_validity_one(ats_pred)
        smact_validity(ats_true)
        get_validity_one(ats_true)
        traj_pred.write(ats_pred, append=True)
        traj_true.write(ats_true, append=True)

    traj_pred.close()
    traj_true.close()
    return pred_path, true_path


def append_match_log(csv_path: str, run_id: str, n_match: int, rms_dists, matchrate: float, mean_rms_dist: float) -> None:
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(f"{run_id},{n_match},{rms_dists}\n")
        f.write("matchrate,rmse\n")
        f.write(f"{matchrate * 100}%,{mean_rms_dist}\n")


def main():
    parser = argparse.ArgumentParser(description="Run paired CSP benchmark evaluation for CGTransformer outputs.")
    parser.add_argument("--config", required=True, help="Path to a CSP benchmark YAML config file.")
    args = parser.parse_args()

    eval_cfg = load_config(args.config)
    print(f"Loaded config from {args.config}")

    gen_path = os.path.join(eval_cfg["main_path"], "gen")
    target_path = os.path.join(eval_cfg["main_path"], "target")
    sorted_gentrajs = collect_sorted_traj_names(gen_path, "gen")
    sorted_targettrajs = collect_sorted_traj_names(target_path, "target")

    for gen_name, target_name in zip(sorted_gentrajs, sorted_targettrajs):
        if int(gen_name[3:-5]) != int(target_name[6:-5]):
            raise ValueError(f"Mismatched pair: {gen_name} vs {target_name}")

    traj_train = Trajectory(eval_cfg["traj_train"])
    traj_test = Trajectory(eval_cfg["traj_test"])
    n_match = 0
    rms_dists = []
    count = 0
    csv_path = os.path.join(eval_cfg["main_path"], "match_rate.csv")

    for gentraj, targettraj in zip(sorted_gentrajs, sorted_targettrajs):
        count += 1
        pred_path, true_path = preprocess_pairwise_traj(eval_cfg["main_path"], gentraj, targettraj)
        traj_pred = Trajectory(pred_path)
        traj_true = Trajectory(true_path)
        match_path = os.path.join(eval_cfg["main_path"], f"match{gentraj[3:-5]}.traj")
        geval = GenEval(traj_pred, traj_true, traj_train, traj_test, match_path, eval_cfg)
        metrics = geval.get_metrics()
        n_match += metrics["n_match"]
        rms_dist = metrics["rms_dists"]
        rms_dists.extend(rms_dist)
        matchrate_tmp = n_match / (count * eval_cfg["batchsize"])
        valid_rms = np.array(rms_dists, dtype=object)
        valid_rms = valid_rms[valid_rms != None]
        mean_rms_dist = valid_rms.astype(float).mean() if len(valid_rms) else float("nan")
        append_match_log(csv_path, gentraj[3:-5], metrics["n_match"], metrics["rms_dists"], matchrate_tmp, mean_rms_dist)

    rms_dists = np.array(rms_dists, dtype=object)
    rms_dists = rms_dists[rms_dists != None]
    matchrate = n_match / len(traj_test) if len(traj_test) else 0.0
    mean_rms_dist = rms_dists.astype(float).mean() if len(rms_dists) else float("nan")
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write("matchrate,rmse\n")
        f.write(f"{matchrate * 100}%,{mean_rms_dist}\n")

    print(f"Match rate: {matchrate}")
    print(f"Mean RMS distance: {mean_rms_dist}")


if __name__ == "__main__":
    main()
