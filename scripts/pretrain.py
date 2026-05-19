import os, json, random, math, argparse, contextlib

import numpy as np
import torch, torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, AutoConfig, get_scheduler
from tqdm.auto import tqdm
import yaml
import utility


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "checkpoints", "basegpt"))


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
    for key in ["train_json", "val_json", "model_dir", "save_dir"]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = resolve_path(cfg[key], base_dir)
    return cfg


def build_runtime_args(config: dict) -> argparse.Namespace:
    defaults = {
        "model_dir": DEFAULT_MODEL_DIR,
        "epochs": 500,
        "batch_size": 64,
        "acc_steps": 1,
        "seed": 42,
        "lr": 5e-4,
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "stage": "pretrain",
        "finetune_target": None,
        "quick_test": False,
        "quick_test_max_train": 20,
        "quick_test_max_val": 20,
    }
    merged = {**defaults, **config}
    merged["stage"] = str(merged.get("stage", "pretrain")).lower()
    merged["local_rank"] = int(os.environ.get("LOCAL_RANK", merged.get("local_rank", -1)))

    finetune_target = merged.get("finetune_target")
    if isinstance(finetune_target, str):
        merged["finetune_target"] = [finetune_target]

    merged["quick_test"] = bool(merged.get("quick_test", False))
    if merged["quick_test"]:
        merged["epochs"] = 2
        merged["batch_size"] = 1
        merged["acc_steps"] = 1

    missing = [key for key in ["train_json", "val_json", "save_dir"] if not merged.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    if merged["stage"] == "finetune" and not merged.get("finetune_target"):
        raise ValueError("finetune_target must be provided when stage is 'finetune'.")

    return argparse.Namespace(**merged)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(local_rank: int):
    """Initialise NCCL process group if launched with torchrun."""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

custom_vocab_dict = utility.custom_vocab_dict


def collate_fn(batch_samples):
    batch_seq = []
    batch_prompt = []
    max_len = 0
    for sample in batch_samples:
        seq = sample["crystPrompt"] + sample["cryst_seq_ele"]
        batch_seq.append(seq)
        batch_prompt.append(sample["propPrompt"])
        max_len = max(max_len, len(seq))
    batch_seq = [seq + ["<pad>"] * (max_len - len(seq)) for seq in batch_seq]
    tokenized_data = utility.crystalTokenizer(batch_prompt, batch_seq, custom_vocab_dict)
    input_ids = torch.stack(tokenized_data["input_ids"], 0)
    attention_mask = torch.stack(tokenized_data["attention_mask"], 0)
    labels = torch.stack(tokenized_data["labels"], 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# dataset defined for CrystalStructureDataset
class CrystalStructureDataset(Dataset):
    def __init__(self, json_file, randomchose=True, dim=2,mode='pretrain',finetune_targetprop_list=None):
        self.data = json_file
        self.random = randomchose
        self.dim = dim
        if mode.lower() not in ['pretrain','finetune']:
            raise ValueError(f"mode must be 'pretrain' or 'finetune', but got {mode}")
        self.mode = mode.lower()
        self.targetprop_list = []
        if self.mode == 'finetune' and finetune_targetprop_list is not None:
            self.targetprop_list = [prop.lower() for prop in finetune_targetprop_list]
        if self.mode == 'finetune' and finetune_targetprop_list is None:
            raise ValueError("finetune_targetprop_list must be provided when mode is 'finetune'")
        self.property_order = [
            'dft_band_gap',
            'energy_above_hull',
            'spacegroup',
            'formation_energy',
            'density',
            'volume',
            'crystal_system',
            'point_group',
            'has_magnetism',
            'is_metal'
        ]
        
    def __len__(self):
        return len(self.data)
    
    def _discretize_bandgap(self, value):
        if value is None or (isinstance(value, (int, float)) and value <= 0):
            return 0
        return int(value // 0.5)

    def _discretize_energy(self, value):
        if value is None or (isinstance(value, (int, float)) and value <= 0):
            return 0
        return int(value // 0.02)


    def _process_properties(self, raw_props):
        """Convert property dict to fixed-length list of tokens."""
        prop_tokens = []
        
        target_set = set(self.targetprop_list) if self.targetprop_list else set()

        for prop_name in self.property_order:
            val = raw_props.get(prop_name, None)
            
            should_reveal = (
                (val is not None and val != '')
                and 
                (not target_set or prop_name in target_set)
            )

            if should_reveal:
                if prop_name == 'dft_band_gap':
                    token = str(self._discretize_bandgap(val))
                elif prop_name == 'energy_above_hull':
                    token = str(self._discretize_energy(val))
                elif prop_name == 'spacegroup':
                    token = str(int(val)) if val is not None else 'unk_prop'
                else:
                    token = str(val)
                prop_tokens.append(token)
            else:
                prop_tokens.append('unk_prop')
            
        return prop_tokens

    def __getitem__(self, idx):
        entry=self.data[idx]
        structures = entry["variants"]
        if self.mode == 'finetune':
            propertys = {k: v for k, v in entry.items() if k != "variants"}
        else:
            propertys = None
        if self.random and self.dim == 2:
            sampled_structure = random.choice(structures)
        elif self.random == False and self.dim == 2:
            sampled_structure =structures[0]
        else:
            sampled_structure = structures
        #return sampled_structure
        def preprocess_object(structure,propertys):
            #species = []
            #coordinates = []
            elements = []
            seq_Ele = []
            seq_noEle = []
            crystPrompt = []
            propPrompt = []
            #lattice_params = []
            atom_idx = 0
            #numclass = 119
            lattice_params=[structure['a'], structure['b'], structure['c'], structure['alpha'], structure['beta'], structure['gamma']]
            while f'x{atom_idx}' in structure:
                seq_Ele.extend([f'x{atom_idx}']+list(str(structure[f'x{atom_idx}']))+[f'y{atom_idx}']+list(str(structure[f'y{atom_idx}']))+[f'z{atom_idx}']+list(str(structure[f'z{atom_idx}'])))
                element = structure[f'Ele{atom_idx}']
                seq_Ele.extend([element])
                elements.append(element)
                #if not element in species:
                    #species.extend(element) 
                atom_idx += 1
            #num_atoms = len(elements)
            species, num_atoms = np.unique(elements, return_counts=True)
            species = species.tolist()
            total_atoms = sum(num_atoms.tolist())
            num_atoms = list(map(str, num_atoms))
            seq_Ele.extend(['a']+list(str(structure['a']))+['b']+list(str(structure['b']))+['c']+list(str(structure['c']))+['alpha']+list(str(structure['alpha']))+['beta']+list(str(structure['beta']))+['gamma']+list(str(structure['gamma'])))
            seq_Ele.append('<eos>')
            #seq_noEle.append('<eos>')
            if self.mode == 'finetune':
                prop_seq = self._process_properties(propertys)
                propPrompt.extend(['<bos>']+prop_seq)
            else:
                propPrompt.extend(['<bos>']+['unk_prop']*10)
            crystPrompt=propPrompt.copy()
            crystPrompt.extend(['species']+species+['total_atoms']+[str(total_atoms)]+['num_atoms']+ num_atoms+['<sep>'])
            #crystPrompt.extend(species)
            return {'num_atom': num_atoms,
                    'species': species,
                    'propPrompt': propPrompt,
                    'crystPrompt': crystPrompt,
                    #'cryst_seq_noele': seq_noEle,
                    'cryst_seq_ele': seq_Ele,
                    #'crystal_json': obj
                    }
        return preprocess_object(sampled_structure,propertys)


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            outs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            pred = outs.logits.argmax(-1)
            labels = batch["labels"].to(device)
            mask = labels != -100
            correct += ((pred == labels) & mask).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)

def parse_args():
    p = argparse.ArgumentParser(description="Pretrain CGTransformer on serialized crystal JSON data.")
    p.add_argument("--config", type=str, required=True, help="Path to a pretraining YAML config file.")
    return p.parse_args()

def main():
    cli_args = parse_args()
    config = load_config(cli_args.config)
    print(f"Loaded config from {cli_args.config}")
    args = build_runtime_args(config)
    init_distributed(args.local_rank)
    set_global_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    device = torch.device("cuda", args.local_rank if args.local_rank >= 0 else 0)

    # ---------- datasets ----------
    train_data = json.load(open(args.train_json))
    val_data = json.load(open(args.val_json))
    if args.quick_test:
        train_data = train_data[: int(args.quick_test_max_train)]
        val_data = val_data[: int(args.quick_test_max_val)]
        print(
            f"Quick test mode enabled: epochs=2, batch_size=1, train_samples={len(train_data)}, val_samples={len(val_data)}"
        )
    train_ds = CrystalStructureDataset(train_data, randomchose=True,dim=2, mode=args.stage)
    val_ds   = CrystalStructureDataset(val_data,   randomchose=False,dim=2, mode=args.stage)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if dist.is_initialized() else None

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    config = AutoConfig.from_pretrained(
        args.model_dir,
        vocab_size=len(custom_vocab_dict),
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        #n_inner=6400,
        local_files_only=True
    )
    model = GPT2LMHeadModel(config).to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
    model.gradient_checkpointing_enable()
    if dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # ---- LR scaling util ------------------------------------
    BASE_EFF_BS = 32
    BASE_LR     = args.lr

    eff_bs = args.batch_size * args.acc_steps
    if dist.is_initialized():
        eff_bs *= dist.get_world_size()
    scaled_lr = BASE_LR * (eff_bs / BASE_EFF_BS) * 1

    if is_main_process():
        print(f"Effective batch = {eff_bs},  learning rate = {scaled_lr:.3e}")

    optimizer = AdamW(model.parameters(), lr=scaled_lr, betas=(0.9,0.99),fused=True)
    # --- scheduler need real opt steps ---
    steps_per_epoch = math.ceil(len(train_dataloader) / args.acc_steps)
    if dist.is_initialized():
        steps_per_epoch = math.ceil(steps_per_epoch / dist.get_world_size())
    sched_steps = steps_per_epoch * args.epochs
    #scaled_lr *= 0.9 
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=int(0.03 * sched_steps), num_training_steps=sched_steps
    )

    scaler = torch.cuda.amp.GradScaler()
    best_val = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss, step_in_epoch = 0.0, 0
        prog = tqdm(train_dataloader, disable=(dist.is_initialized() and not is_main_process()), desc=f"Epoch {epoch+1}")
  
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(prog, 1):
            accumulate = (step % args.acc_steps) != 0

            if dist.is_initialized() and accumulate:
                cm = model.no_sync()
            else:
                cm = contextlib.nullcontext()

            with cm:
                with torch.cuda.amp.autocast():
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    loss = out.loss / args.acc_steps
                scaler.scale(loss).backward()
                total_loss += out.loss.item()
                step_in_epoch += 1
            if not accumulate:
                scaler.step(optimizer) 
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                del loss, out
                torch.cuda.empty_cache()
                # -----------------------------

            if is_main_process() and step % (args.acc_steps * 10) == 0:
                prog.set_postfix(loss=f"{total_loss/step_in_epoch:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.3e}")

        # ---- validation ----
        val_acc = evaluate(model.module if isinstance(model, DDP) else model, val_dataloader, device)
        if is_main_process():
            print(f"Epoch {epoch+1}: val token acc = {val_acc*100:.2f}%")

            # save best model
            if val_acc > best_val:
                best_val = val_acc
                save_path = os.path.join(args.save_dir, f"best_epoch{epoch+1}_acc{best_val*100:.2f}.pt")
                torch.save({
                    "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": best_val,
                }, save_path)
                print(f"[saved] {save_path}")
            if ((epoch+1)<10) or (10 <= (epoch+1) <=100 and (epoch+1) % 10 == 0) or ((epoch+1)>100 and (epoch+1)<=1000 and (epoch+1) % 50 == 0) or ((epoch+1)>1000 and (epoch+1) % 100 == 0):
                print('saving new weights...\n')
                save_path = os.path.join(args.save_dir, f"best_epoch{epoch+1}_acc{best_val*100:.2f}.pt")
                torch.save({
                    "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": best_val,
                }, save_path)
        print("Done!")
    
    cleanup_distributed()

if __name__ == '__main__':
    main()
