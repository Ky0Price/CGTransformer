import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple

import yaml
from ase.io.trajectory import Trajectory
from tqdm.auto import tqdm

from evaluate import GenEval, get_fingerprints, get_validity_one, smact_validity

warnings.filterwarnings("ignore")


def resolve_path(path: str | None, base_dir: str) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.abspath(config_path))

    paths = cfg.setdefault("paths", {})
    for key in ["gen_root", "train_traj", "test_traj", "true_cache_dir", "log_dir"]:
        if key in paths and paths[key] is not None:
            paths[key] = resolve_path(paths[key], base_dir)

    return cfg


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def locate_experiment_root(gen_root: str, test_type: str | None) -> Tuple[str, List[str]]:
    candidates = []
    if test_type:
        candidates.append(os.path.join(gen_root, test_type))
    candidates.append(gen_root)

    for exp_root in candidates:
        if not os.path.exists(exp_root):
            continue

        direct_traj = os.path.join(exp_root, "gen.traj")
        if os.path.isfile(direct_traj):
            return exp_root, ["."]

        subdirs = [name for name in os.listdir(exp_root) if os.path.isdir(os.path.join(exp_root, name))]
        if subdirs:
            def sort_key(name: str):
                try:
                    return (0, int(name))
                except ValueError:
                    return (1, name)

            return exp_root, sorted(subdirs, key=sort_key)

    raise FileNotFoundError(f"Could not find experiment outputs under: {candidates}")


def should_compute_fingerprints(eval_cfg: Dict) -> bool:
    return (
        eval_cfg.get("get_fp", False)
        or eval_cfg.get("calc_coverage", False)
        or eval_cfg.get("calc_comp_div", False)
        or eval_cfg.get("calc_struct_div", False)
    )


def initialize_eval_info(atoms) -> None:
    atoms.info["constructed"] = False
    atoms.info["comp_valid"] = False
    atoms.info["struct_valid"] = False
    atoms.info["valid"] = False
    atoms.info["comp_fp"] = None
    atoms.info["struct_fp"] = None


def preprocess_atoms_for_eval(atoms, compute_fp: bool) -> None:
    initialize_eval_info(atoms)

    try:
        smact_validity(atoms)
    except Exception:
        atoms.info["comp_valid"] = False

    try:
        get_validity_one(atoms)
    except Exception:
        atoms.info["constructed"] = False
        atoms.info["struct_valid"] = False
        atoms.info["valid"] = False

    if compute_fp:
        try:
            get_fingerprints(atoms)
        except Exception:
            atoms.info["comp_fp"] = None
            atoms.info["struct_fp"] = None


def preprocess_traj(
    input_traj_path: str,
    output_traj_path: str,
    compute_fp: bool,
    description: str,
    force_reprocess: bool = False,
) -> None:
    if os.path.exists(output_traj_path) and not force_reprocess:
        return
    if not os.path.exists(input_traj_path):
        raise FileNotFoundError(f"Input trajectory not found: {input_traj_path}")

    raw_traj = Trajectory(input_traj_path)
    with Trajectory(output_traj_path, mode="w") as processed_traj:
        for atoms in tqdm(raw_traj, total=len(raw_traj), desc=description):
            atoms_copy = atoms.copy()
            preprocess_atoms_for_eval(atoms_copy, compute_fp=compute_fp)
            processed_traj.write(atoms_copy)


def evaluate_one_epoch(
    pred_traj_path: str,
    true_traj_path: str,
    train_traj_path: str,
    test_traj_path: str,
    eval_cfg: Dict,
) -> Dict:
    traj_pred = Trajectory(pred_traj_path)
    traj_true = Trajectory(true_traj_path)
    traj_train = Trajectory(train_traj_path)
    traj_test = Trajectory(test_traj_path)

    geval = GenEval(
        traj_pred=traj_pred,
        traj_true=traj_true,
        traj_train=traj_train,
        traj_test=traj_test,
        traj_match_path=None,
        Eval=eval_cfg,
    )
    return geval.get_metrics()


def write_log(log_path: str, record: Dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unconditional benchmark evaluation for CGTransformer outputs.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    eval_cfg = config["Eval"]
    compute_fp = should_compute_fingerprints(eval_cfg)
    ensure_dir(config["paths"]["true_cache_dir"])
    ensure_dir(config["paths"]["log_dir"])

    exp_root, model_dirs = locate_experiment_root(
        gen_root=config["paths"]["gen_root"],
        test_type=config.get("test_type"),
    )

    processed_true_traj = os.path.join(
        config["paths"]["true_cache_dir"],
        f"{config['dataset']}_true_processed.traj",
    )
    preprocess_traj(
        input_traj_path=config["paths"]["test_traj"],
        output_traj_path=processed_true_traj,
        compute_fp=compute_fp,
        description="Preprocessing reference structures",
        force_reprocess=config.get("force_reprocess_true", False),
    )

    run_name = config.get("test_type") or os.path.basename(exp_root.rstrip("/\\")) or "benchmark"
    log_path = os.path.join(config["paths"]["log_dir"], f"{run_name}.jsonl")

    for model_dir_name in model_dirs:
        model_dir = exp_root if model_dir_name == "." else os.path.join(exp_root, model_dir_name)
        display_name = os.path.basename(model_dir.rstrip("/\\")) if model_dir_name == "." else model_dir_name
        print(f"========== Evaluating {display_name} ==========")

        raw_pred_traj = os.path.join(model_dir, "gen.traj")
        processed_pred_traj = os.path.join(model_dir, "pred_processed.traj")
        if not os.path.exists(raw_pred_traj):
            print(f"Skip {display_name}: gen.traj not found")
            continue

        preprocess_traj(
            input_traj_path=raw_pred_traj,
            output_traj_path=processed_pred_traj,
            compute_fp=compute_fp,
            description=f"Preprocessing predictions for {display_name}",
            force_reprocess=config.get("force_reprocess_pred", False),
        )

        metrics = evaluate_one_epoch(
            pred_traj_path=processed_pred_traj,
            true_traj_path=processed_true_traj,
            train_traj_path=config["paths"]["train_traj"],
            test_traj_path=config["paths"]["test_traj"],
            eval_cfg=eval_cfg,
        )

        decoded_count = len(Trajectory(processed_pred_traj))
        expected_num_gen = config["num_gen"]
        decoded_rate = decoded_count / expected_num_gen if expected_num_gen else 0.0
        overall_struct_valid_rate = metrics.get("struct_valid", 0.0) * decoded_rate

        record = {
            "run": display_name,
            "decoded_count": decoded_count,
            "expected_num_gen": expected_num_gen,
            "decoded_rate": decoded_rate,
            "struct_valid_among_decoded": metrics.get("struct_valid", None),
            "overall_struct_valid_rate": overall_struct_valid_rate,
            "metrics": metrics,
        }
        write_log(log_path, record)

        print(f"Decoded structures: {decoded_count}/{expected_num_gen}")
        print(f"Structural validity among decoded: {metrics.get('struct_valid', None)}")
        print(f"Overall structural validity rate: {overall_struct_valid_rate}")
        print("========== Evaluation finished ==========")


if __name__ == "__main__":
    main()
