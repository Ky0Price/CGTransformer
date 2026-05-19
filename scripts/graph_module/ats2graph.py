import argparse
import functools
import json
import os
import random
from typing import List

import networkx as nx
import numpy as np
import yaml
from ase.io.trajectory import Trajectory
from pymatgen.analysis.local_env import EconNN, MinimumDistanceNN, VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor
from spglib import get_symmetry_dataset

from ats2txt import write_json
from crystalgraph import bfs_sorting

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))


def resolve_path(path: str | None, base_dir: str) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ["traj_dir", "out_json", "fail_log"]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = resolve_path(cfg[key], base_dir)
    return cfg


METHOD_TO_MODE = {
    "3": "gs",
    "4": "random",
    "5": "esg",
    "6": "xyz",
    "7": "ps",
    "8": "esp",
}


def sort_pair(sorted_paired):
    result = []
    current_key = None
    current_group = []

    for key, value in sorted_paired:
        if key != current_key:
            if current_group:
                result.append(current_group)
            current_key = key
            current_group = [value]
        else:
            current_group.append(value)
    if current_group:
        result.append(current_group)
    return result


def sort_xyz(pos, i, direct=True):
    intercept = -pos[i]
    pos_rei = (pos + intercept) % 1 if direct else pos + intercept

    def cmp_vec(m, n):
        pos_m = pos_rei[m]
        pos_n = pos_rei[n]

        if pos_m[0] < pos_n[0]:
            return -1
        elif pos_m[0] > pos_n[0]:
            return 1
        elif pos_m[1] < pos_n[1]:
            return -1
        elif pos_m[1] > pos_n[1]:
            return 1
        elif pos_m[2] < pos_n[2]:
            return -1
        elif pos_m[2] > pos_n[2]:
            return 1
        return 0

    temp = list(range(len(pos)))
    temp.remove(i)
    perm = sorted(temp, key=functools.cmp_to_key(cmp_vec))
    return [i] + perm


def get_neighbors_of_site_with_index(struct, n, approach, delta, cutoff):
    if approach == "min_dist":
        return MinimumDistanceNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)
    if approach == "voronoi":
        return VoronoiNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)
    if approach == "econ":
        return EconNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)
    raise ValueError(f"Unsupported approach: {approach}")


def quot_gen(ats, approach, delta):
    struct = AseAtomsAdaptor.get_structure(ats)
    cutoff = ats.cell.cellpar()[:3].mean()
    graph = nx.MultiGraph()
    for i in range(len(struct)):
        graph.add_node(i)

    for i in range(len(struct)):
        neighs_list = get_neighbors_of_site_with_index(struct, i, approach, delta, cutoff)
        for nn in neighs_list:
            j = nn["site_index"]
            if i <= j:
                graph.add_edge(i, j, vector=np.array(nn["image"]), direction=(i, j))

    if nx.number_connected_components(graph) != 1:
        return False
    return graph


def append_fail_index(fail_log, index):
    with open(fail_log, "a", encoding="utf-8") as failobj:
        failobj.write(f"{index}\n")


def build_permutations(ats, method, fail_log):
    sym_num = None

    if len(ats) == 1:
        return [[0]], sym_num

    if method == "1":
        perm = []
        for n in range(len(ats)):
            total = list(range(len(ats)))
            perm_single = total[n:]
            perm_single.extend(list(reversed(total[:n])))
            perm.append(perm_single)
        return perm, sym_num

    if method == "2":
        graph = quot_gen(ats, "voronoi", 0)
        if not graph:
            return None, None
        return bfs_sorting(ats, graph, baryPos=False), sym_num

    if method == "3":
        graph = quot_gen(ats, "voronoi", 0.5)
        delta = 0.4
        while (not graph) and (delta >= 0):
            graph = quot_gen(ats, "voronoi", delta)
            delta -= 0.1
        if not graph:
            return None, None
        perm_old = bfs_sorting(ats, graph, baryPos=False)
        ats_cell = (np.array(ats.cell), ats.positions, ats.get_atomic_numbers())
        data = get_symmetry_dataset(ats_cell, symprec=0.1)
        wyckoffs = data["wyckoffs"]
        sym_num = data["number"]
        sym_list = sorted(list(set(wyckoffs)))
        sym_index = []
        perm = []

        for sym_letter in sym_list:
            sym_index.append([i for i, num in enumerate(wyckoffs) if num == sym_letter])
        for head in sym_index[0]:
            perm_single = perm_old[head]
            sym_index_new = []
            for sym_index_row in sym_index:
                sym_perm = []
                for indexi in sym_index_row:
                    sym_perm.append(perm_single.index(indexi))
                combined = sorted(zip(sym_perm, sym_index_row), key=lambda x: x[0])
                sym_index_new.extend([x[1] for x in combined])
            perm.append(sym_index_new)
        return perm, sym_num

    if method == "4":
        perm = []
        for n in range(len(ats)):
            perm_single = [n]
            temp = list(range(len(ats)))
            temp.remove(n)
            random.shuffle(temp)
            perm_single.extend(temp)
            perm.append(perm_single)
        return perm, sym_num

    if method == "5":
        graph = quot_gen(ats, "voronoi", 0.5)
        delta = 0.4
        while (not graph) and (delta >= 0):
            graph = quot_gen(ats, "voronoi", delta)
            delta -= 0.1
        if not graph:
            return None, None
        atomic_numbers = ats.get_atomic_numbers()
        ats_number = sorted(list(set(atomic_numbers)))
        num_index = []
        for number in ats_number:
            num_index.append([i for i, num in enumerate(atomic_numbers) if num == number])

        num_sym = []
        ats_cell = (np.array(ats.cell), ats.positions, ats.get_atomic_numbers())
        data = get_symmetry_dataset(ats_cell, symprec=0.1)
        try:
            wyckoffs = data["wyckoffs"]
            sym_num = data["number"]
        except Exception:
            return None, None
        for sub_list in num_index:
            paired = [(wyckoffs[i], i) for i in sub_list]
            paired.sort(key=lambda x: x[0])
            temp = sort_pair(paired)
            for sub_temp in temp:
                num_sym.append(sub_temp)

        perm = []
        perm_old = bfs_sorting(ats, graph, baryPos=False)
        for head in num_sym[0]:
            perm_single = perm_old[head]
            num_sym_new = []
            for num_sym_row in num_sym:
                sym_perm = []
                for indexi in num_sym_row:
                    sym_perm.append(perm_single.index(indexi))
                combined = sorted(zip(sym_perm, num_sym_row), key=lambda x: x[0])
                num_sym_new.extend([x[1] for x in combined])
            perm.append(num_sym_new)
        return perm, sym_num

    if method == "6":
        perm = []
        spos = ats.get_scaled_positions()
        for n in range(len(ats)):
            perm.append(sort_xyz(spos, n, direct=True))
        return perm, sym_num

    if method == "7":
        perm_old = []
        spos = ats.get_positions()
        for n in range(len(ats)):
            perm_old.append(sort_xyz(spos, n, direct=False))
        ats_cell = (np.array(ats.cell), ats.positions, ats.get_atomic_numbers())
        data = get_symmetry_dataset(ats_cell, symprec=0.1)
        try:
            wyckoffs = data["wyckoffs"]
            sym_num = data["number"]
        except Exception:
            return None, None
        sym_list = sorted(list(set(wyckoffs)))
        sym_index = []
        perm = []

        for sym_letter in sym_list:
            sym_index.append([i for i, num in enumerate(wyckoffs) if num == sym_letter])
        for head in sym_index[0]:
            perm_single = perm_old[head]
            sym_index_new = []
            for sym_index_row in sym_index:
                sym_perm = []
                for indexi in sym_index_row:
                    sym_perm.append(perm_single.index(indexi))
                combined = sorted(zip(sym_perm, sym_index_row), key=lambda x: x[0])
                sym_index_new.extend([x[1] for x in combined])
            perm.append(sym_index_new)
        return perm, sym_num

    if method == "8":
        perm_old = []
        spos = ats.get_scaled_positions()
        for n in range(len(ats)):
            perm_old.append(sort_xyz(spos, n, direct=True))

        atomic_numbers = ats.get_atomic_numbers()
        ats_number = sorted(list(set(atomic_numbers)))
        num_index = []
        for number in ats_number:
            num_index.append([i for i, num in enumerate(atomic_numbers) if num == number])
        num_sym = []
        ats_cell = (np.array(ats.cell), ats.positions, ats.get_atomic_numbers())
        data = get_symmetry_dataset(ats_cell, symprec=0.1)
        try:
            wyckoffs = data["wyckoffs"]
            sym_num = data["number"]
        except Exception:
            return None, None
        for sub_list in num_index:
            paired = [(wyckoffs[i], i) for i in sub_list]
            paired.sort(key=lambda x: x[0])
            temp = sort_pair(paired)
            for sub_temp in temp:
                num_sym.append(sub_temp)
        perm = []
        for head in num_sym[0]:
            perm_single = perm_old[head]
            num_sym_new = []
            for num_sym_row in num_sym:
                sym_perm = []
                for indexi in num_sym_row:
                    sym_perm.append(perm_single.index(indexi))
                combined = sorted(zip(sym_perm, num_sym_row), key=lambda x: x[0])
                num_sym_new.extend([x[1] for x in combined])
            perm.append(num_sym_new)
        return perm, sym_num

    raise ValueError(f"Unsupported method: {method}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert trajectory files into CGTransformer JSON datasets.")
    parser.add_argument("--config", required=True, help="Path to a dataset conversion YAML config file.")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    config = load_config(cli_args.config)
    print(f"Loaded config from {cli_args.config}")

    required = ["traj_dir", "target", "data_name", "method"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    args = argparse.Namespace(**config)
    mode = METHOD_TO_MODE.get(str(args.method), str(args.method))
    traj_path = os.path.join(args.traj_dir, f"{args.target}.traj")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    dataset_dir = os.path.join(REPO_ROOT, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    out_json = args.out_json or os.path.join(dataset_dir, f"{args.target}_{args.data_name}_{mode}.json")
    fail_log = args.fail_log or os.path.join(dataset_dir, f"{args.target}_log_{args.data_name}.txt")

    traj = Trajectory(traj_path)
    large_data = []
    for i, ats in enumerate(traj):
        perm, sym_num = build_permutations(ats, args.method, fail_log)
        if perm is None:
            append_fail_index(fail_log, i)
            continue
        large_data.append(write_json(ats, perm, direct=True, sym=sym_num))

    with open(out_json, "w", encoding="utf-8") as fileobj:
        json.dump(large_data, fileobj, ensure_ascii=False, indent=4)

    print(f"Saved {len(large_data)} entries to {out_json}")
    print(f"Failed sample indices are logged to {fail_log}")


if __name__ == "__main__":
    main()
