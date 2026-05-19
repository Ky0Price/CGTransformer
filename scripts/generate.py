import argparse
import os
import random
import warnings

import numpy as np
import torch
import yaml
from ase.io import read, write
from ase.io.trajectory import Trajectory
from tqdm.auto import tqdm

import utility

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))


def resolve_path(path, base_dir):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def resolve_checkpoint_path(model_root, test_type, ckpt_name, base_dir):
    if ckpt_name is None:
        return None
    if os.path.isabs(ckpt_name):
        return ckpt_name

    direct_path = os.path.abspath(os.path.join(base_dir, ckpt_name))
    if os.path.exists(direct_path):
        return direct_path

    if model_root and test_type:
        return os.path.join(model_root, test_type, ckpt_name)
    if model_root:
        return os.path.join(model_root, ckpt_name)
    return direct_path


def discretize_bandgap(value):
    if value is None or (isinstance(value, (int, float)) and value <= 0):
        return 0
    return int(value // 0.5)


def discretize_energy(value):
    if value is None or (isinstance(value, (int, float)) and value <= 0):
        return 0
    return int(value // 0.02)


def process_properties(
    prop_dict,
    prop_base=[
        "dft_band_gap",
        "energy_above_hull",
        "spacegroup",
        "formation_energy",
        "density",
        "volume",
        "crystal_system",
        "point_group",
        "has_magnetism",
        "is_metal",
    ],
):
    prop_tokens = ["<bos>"]
    target_prop = prop_dict.keys()
    for prop_name in prop_base:
        val = prop_dict.get(prop_name, None)
        should_reveal = (val is not None and val != "") and (prop_name in target_prop)

        if should_reveal:
            if prop_name == "dft_band_gap":
                token = str(discretize_bandgap(prop_dict[prop_name]))
            elif prop_name == "energy_above_hull":
                token = str(discretize_energy(prop_dict[prop_name]))
            elif prop_name == "spacegroup":
                token = str(prop_dict[prop_name])
            else:
                token = str(prop_dict[prop_name])
            prop_tokens.append(token)
        else:
            prop_tokens.append("unk_prop")

    return prop_tokens


def build_cryst_prompt(prop_dict, species=None, formula=None, min_atom=None, max_atom=None):
    if (prop_dict is None) or (len(prop_dict) == 0):
        prop_tokens = ["<bos>"] + ["unk_prop"] * 10
    else:
        prop_tokens = process_properties(prop_dict)

    if not species:
        return prop_tokens + ["species"]

    if formula is None:
        if min_atom is None or max_atom is None:
            raise ValueError("Please provide both min_atom and max_atom.")
        total_atoms = random.randint(int(min_atom), int(max_atom))
        return prop_tokens + ["species"] + species + ["total_atoms", str(total_atoms)]

    if len(species) != len(formula):
        raise ValueError("chemical_species and formula must have the same length.")
    if min_atom is None or max_atom is None:
        raise ValueError("Please provide both min_atom and max_atom.")

    base_sum = int(sum(formula))
    if base_sum <= 0:
        raise ValueError("The sum of formula must be positive.")

    max_mult = max(1, int(max_atom) // base_sum)
    min_mult = max(1, int(min_atom) // base_sum) if int(min_atom) >= base_sum else 1
    mult = random.randint(min_mult, max_mult)
    num_atoms = [str(int(f) * mult) for f in formula]
    total_atoms = sum(int(x) for x in num_atoms)

    return (
        prop_tokens
        + ["species"]
        + species
        + ["total_atoms", str(total_atoms), "num_atoms"]
        + num_atoms
        + ["<sep>"]
    )


def load_model(base_model_dir, vocab, baseckpt=None, peftckpt=None, device=None):
    from transformers import AutoConfig, GPT2LMHeadModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pad_id, eos_id = vocab["<pad>"], vocab["<eos>"]

    if not baseckpt or not os.path.exists(baseckpt):
        raise FileNotFoundError(f"Base checkpoint path does not exist: {baseckpt}")

    if os.path.isfile(baseckpt):
        cfg = AutoConfig.from_pretrained(
            base_model_dir,
            vocab_size=len(vocab),
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            local_files_only=True,
        )
        base = GPT2LMHeadModel(cfg)
        base.load_state_dict(torch.load(baseckpt, map_location="cpu")["model"])
    else:
        base = GPT2LMHeadModel.from_pretrained(baseckpt, local_files_only=True)

    if base.get_input_embeddings().weight.size(0) != len(vocab):
        base.resize_token_embeddings(len(vocab))

    base.config.pad_token_id = pad_id
    base.config.eos_token_id = eos_id

    if peftckpt is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, peftckpt, is_trainable=False)
    else:
        model = base

    model.eval().to(device)
    return model


def encode_batch(prompt_batch, vocab):
    pad_id = vocab["<pad>"]
    ids_list = [torch.tensor([vocab[t] for t in seq], dtype=torch.long) for seq in prompt_batch]
    max_len = max(x.size(0) for x in ids_list)
    out_ids = torch.stack(
        [
            torch.cat([x, torch.full((max_len - x.size(0),), pad_id, dtype=torch.long)])
            for x in ids_list
        ],
        dim=0,
    )
    attn = (out_ids != pad_id).long()
    return out_ids, attn


def traj2strucs(trajpath, outputformat, outputpath):
    supported_formats = ["cif", "xyz", "vasp"]
    if outputformat not in supported_formats:
        raise ValueError(f"Unsupported output format: {outputformat}. Supported formats: {supported_formats}")

    structures = read(trajpath, index=":")
    os.makedirs(outputpath, exist_ok=True)

    for i, struct in enumerate(structures):
        output_file = os.path.join(outputpath, f"structure_{i}.{outputformat}")
        write(output_file, struct)

    print(f"Converted {len(structures)} structures from {trajpath} to {outputformat} format in {outputpath}.")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate crystal structures from a YAML configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to a generation config YAML file.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load or parse config file {args.config}: {exc}") from exc

    config_dir = os.path.dirname(os.path.abspath(args.config))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    vocab = utility.custom_vocab_dict
    id2tok = utility.id2vocab_dict
    eos_id = vocab["<eos>"]
    pad_id = vocab["<pad>"]

    pretrained_model_path = resolve_path(config["pretrained_model_path"], config_dir)
    model_path = resolve_path(config.get("model_path"), config_dir) if config.get("model_path") else None
    test_type = config.get("test_type")

    baseckpt = resolve_checkpoint_path(model_path, test_type, config.get("baseckpt"), config_dir)
    peftckpt = resolve_checkpoint_path(model_path, test_type, config.get("peftckpt"), config_dir) if config.get("peftckpt") else None

    model = load_model(
        base_model_dir=pretrained_model_path,
        vocab=vocab,
        baseckpt=baseckpt,
        peftckpt=peftckpt,
        device=device,
    )

    tag = config.get("prefix", "sample")
    default_output_path = os.path.join(config.get("test_path", os.path.join(REPO_ROOT, "outputs")), tag)
    outputs_dir = resolve_path(config.get("output_path", default_output_path), config_dir)
    os.makedirs(outputs_dir, exist_ok=True)

    gen_cfg = config["gen_para"]
    num_gen = int(gen_cfg["num_gen"])
    bs = int(gen_cfg.get("bs", 8))
    do_sample = bool(gen_cfg.get("do_sample", True))
    top_k = int(gen_cfg.get("top_k", 50))
    temperature = float(gen_cfg.get("temperature", 0.8))
    prop_dict = gen_cfg.get("prop_dict", {})
    species = gen_cfg.get("chemical_species", None) or None
    formula = gen_cfg.get("formula", None) or None
    min_atom = gen_cfg.get("min_atom", None)
    max_atom = gen_cfg.get("max_atom", None)

    sample_prompt = build_cryst_prompt(prop_dict, species, formula, min_atom, max_atom)
    input_len_est = len(sample_prompt)
    per_atom_budget = 17
    budget = int(max_atom) * per_atom_budget + 10 if max_atom else 512
    max_new_tokens = min(1024 - input_len_est, budget)
    if max_new_tokens <= 0:
        max_new_tokens = 32

    generated_ids = []
    with torch.inference_mode():
        pbar = tqdm(range(num_gen), desc=f"[{tag}]")
        rounds = (num_gen + bs - 1) // bs
        for round_idx in range(rounds):
            cur_bs = bs if (round_idx + 1) * bs <= num_gen else (num_gen - round_idx * bs)
            if cur_bs <= 0:
                break

            prompt_batch = [build_cryst_prompt(prop_dict, species, formula, min_atom, max_atom) for _ in range(cur_bs)]
            in_ids, attn = encode_batch(prompt_batch, vocab)
            in_ids = in_ids.to(device)
            attn = attn.to(device)

            out = model.generate(
                input_ids=in_ids,
                attention_mask=attn,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                repetition_penalty=1.1,
            )
            for i in range(out.size(0)):
                generated_ids.append(out[i].detach().cpu())

            pbar.update(cur_bs)
        pbar.close()

    maxlen = min(1024, max(seq.size(0) for seq in generated_ids))
    gen_padded = torch.stack(
        [
            torch.cat([seq, torch.full((maxlen - seq.size(0),), pad_id, dtype=torch.long)])
            if seq.size(0) < maxlen
            else seq[:maxlen]
            for seq in generated_ids
        ],
        dim=0,
    )

    decoded_lines = utility.crystalBatchDecoder(gen_padded, id2vocab_dict=id2tok)
    with open(os.path.join(outputs_dir, f"gentext_{tag}.log"), "w", encoding="utf-8") as f:
        f.write(f"[{tag}]\n")
        f.write("input_settings\n")
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        f.write("\n# ===== Generated Structures =====\n")
        for line in decoded_lines:
            f.write(line + "\n")

    gen_atoms_list = utility.crystText2Atoms(decoded_lines)
    valid_atoms, bad_idx = [], []
    for i, atoms in enumerate(gen_atoms_list):
        if atoms is not None:
            valid_atoms.append(atoms)
        else:
            bad_idx.append(i)

    print(f"[{tag}] Total={len(gen_atoms_list)}  Successfully decoded={len(valid_atoms)}  Errors={len(bad_idx)}")
    traj_path = os.path.join(outputs_dir, "gen.traj")
    with Trajectory(traj_path, "w") as traj:
        for atoms in valid_atoms:
            traj.write(atoms)

    outputformat = config.get("outputformat", None)
    if outputformat is not None:
        traj2strucs(traj_path, outputformat, os.path.join(outputs_dir, f"{outputformat}s"))
    else:
        print("No outputformat specified. Skipping structure file conversion.")


if __name__ == "__main__":
    main()
