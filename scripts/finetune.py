import os, json, random, math, argparse, contextlib

import numpy as np
import torch, torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, AutoConfig, get_scheduler
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from typing import Optional, Tuple
import yaml
import utility

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

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
    for key in [
        "train_json",
        "val_json",
        "model_dir",
        "baseckpt",
        "peftckpt",
        "save_dir",
        "export_merged_dir",
    ]:
        if key in cfg and cfg[key] is not None:
            cfg[key] = resolve_path(cfg[key], base_dir)
    return cfg


def build_runtime_args(config: dict) -> argparse.Namespace:
    defaults = {
        "model_dir": DEFAULT_MODEL_DIR,
        "baseckpt": None,
        "peftckpt": None,
        "epochs": 500,
        "batch_size": 64,
        "acc_steps": 1,
        "seed": 42,
        "lr": 5e-4,
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "stage": "pretrain",
        "finetune_target": None,
        "use_lora": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_targets": ["c_attn", "c_proj", "c_fc"],
        "valid_interval": 20,
        "export_merged_dir": None,
        "quick_test": False,
        "quick_test_max_train": 20,
        "quick_test_max_val": 20,
    }
    merged = {**defaults, **config}
    merged["stage"] = str(merged.get("stage", "pretrain")).lower()
    merged["local_rank"] = int(os.environ.get("LOCAL_RANK", merged.get("local_rank", -1)))
    merged["use_lora"] = bool(merged.get("use_lora", False))
    merged["quick_test"] = bool(merged.get("quick_test", False))

    for key in ["finetune_target", "lora_targets"]:
        value = merged.get(key)
        if isinstance(value, str):
            merged[key] = [value]

    if merged["quick_test"]:
        merged["epochs"] = 2
        merged["batch_size"] = 1
        merged["acc_steps"] = 1
        merged["valid_interval"] = 1

    required = ["model_dir"]
    if merged.get("export_merged_dir"):
        required.extend(["baseckpt", "peftckpt"])
    else:
        required.extend(["train_json", "val_json", "save_dir"])

    missing = [key for key in required if not merged.get(key)]
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

def _is_ddp(m):
    import torch.nn.parallel as p
    return isinstance(m, (p.DistributedDataParallel, p.DataParallel))

def _unwrap(m):
    return m.module if _is_ddp(m) else m

def save_checkpoint(
    model: torch.nn.Module,
    save_dir: str,
    epoch: int,
    val_acc: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_lora: bool = True,
    tag: str = "best",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    m = _unwrap(model)

    if use_lora:
        m.save_pretrained(save_dir)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_dir, "scaler.pt"))
        with open(os.path.join(save_dir, "train_state.json"), "w", encoding="utf-8") as f:
            json.dump({"epoch": int(epoch), "val_acc": float(val_acc), "tag": tag}, f)
    else:
        torch.save(
            {
                "model": m.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": int(epoch),
                "val_acc": float(val_acc),
                "tag": tag,
            },
            os.path.join(save_dir, f"full_{tag}.pt"),
        )

def load_checkpoint(
    model: torch.nn.Module,
    load_path_or_dir: str,
    device: str = "cuda",
    use_lora: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True,
) -> Tuple[torch.nn.Module, int, float]:
    if use_lora:
        base = _unwrap(model)
        if isinstance(base, PeftModel):
            base.load_adapter(load_path_or_dir, adapter_name="default", is_trainable=True, overwrite=True)
            base.set_adapter("default")
            new_model = model
        #else:
            #from peft import get_peft_model
            #new_model = _PeftModel.from_pretrained(base, load_path_or_dir, is_trainable=True)

        ep, va = 0, 0.0
        ts = os.path.join(load_path_or_dir, "train_state.json")
        if os.path.isfile(ts):
            with open(ts, "r", encoding="utf-8") as f:
                st = json.load(f)
                ep, va = int(st.get("epoch", 0)), float(st.get("val_acc", 0.0))

        p_opt = os.path.join(load_path_or_dir, "optimizer.pt")
        if optimizer is not None and os.path.isfile(p_opt):
            optimizer.load_state_dict(torch.load(p_opt, map_location=device))
        p_sch = os.path.join(load_path_or_dir, "scheduler.pt")
        if lr_scheduler is not None and os.path.isfile(p_sch):
            lr_scheduler.load_state_dict(torch.load(p_sch, map_location=device))
        p_sc = os.path.join(load_path_or_dir, "scaler.pt")
        if scaler is not None and os.path.isfile(p_sc):
            scaler.load_state_dict(torch.load(p_sc, map_location=device))

        return new_model, ep, va
    else:
        ckpt = torch.load(load_path_or_dir, map_location=device)
        _unwrap(model).load_state_dict(ckpt["model"], strict=strict)
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None and ckpt.get("scheduler") is not None:
            lr_scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        return model, int(ckpt.get("epoch", 0)), float(ckpt.get("val_acc", 0.0))

def merge_lora_and_export(
    base_or_peft_model: torch.nn.Module,
    lora_dir: str,
    export_dir: str,
) -> str:
    os.makedirs(export_dir, exist_ok=True)

    from peft import PeftModel as _PeftModel
    m = _unwrap(base_or_peft_model)

    if isinstance(m, _PeftModel):
        merged = m.merge_and_unload()
    else:
        lora = _PeftModel.from_pretrained(m, lora_dir, is_trainable=False)
        merged = lora.merge_and_unload()

    merged.save_pretrained(export_dir)
    return export_dir

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
            elements = []
            seq_Ele = []
            seq_noEle = []
            crystPrompt = []
            propPrompt = []
            atom_idx = 0
            lattice_params=[structure['a'], structure['b'], structure['c'], structure['alpha'], structure['beta'], structure['gamma']]
            while f'x{atom_idx}' in structure:
                seq_Ele.extend([f'x{atom_idx}']+list(str(structure[f'x{atom_idx}']))+[f'y{atom_idx}']+list(str(structure[f'y{atom_idx}']))+[f'z{atom_idx}']+list(str(structure[f'z{atom_idx}'])))
                element = structure[f'Ele{atom_idx}']
                seq_Ele.extend([element])
                elements.append(element)
                atom_idx += 1
            species, num_atoms = np.unique(elements, return_counts=True)
            species = species.tolist()
            total_atoms = sum(num_atoms.tolist())
            num_atoms = list(map(str, num_atoms))
            seq_Ele.extend(['a']+list(str(structure['a']))+['b']+list(str(structure['b']))+['c']+list(str(structure['c']))+['alpha']+list(str(structure['alpha']))+['beta']+list(str(structure['beta']))+['gamma']+list(str(structure['gamma'])))
            seq_Ele.append('<eos>')
            if self.mode == 'finetune':
                prop_seq = self._process_properties(propertys)
                propPrompt.extend(['<bos>']+prop_seq)
            else:
                propPrompt.extend(['<bos>']+['unk_prop']*10)
            crystPrompt=propPrompt.copy()
            crystPrompt.extend(['species']+species+['total_atoms']+[str(total_atoms)]+['num_atoms']+ num_atoms+['<sep>'])
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
    p = argparse.ArgumentParser(description="Fine-tune CGTransformer checkpoints with full tuning or LoRA.")
    p.add_argument("--config", type=str, required=True, help="Path to a finetuning YAML config file.")
    return p.parse_args()

def main():
    cli_args = parse_args()
    config = load_config(cli_args.config)
    print(f"Loaded config from {cli_args.config}")
    args = build_runtime_args(config)
    init_distributed(args.local_rank)
    set_global_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    device = torch.device("cuda", args.local_rank if args.local_rank >= 0 else 0)

    if args.export_merged_dir:
        if not args.baseckpt or not args.peftckpt:
            raise ValueError("--export_merged_dir requires both --baseckpt and --peftckpt.")
        config = AutoConfig.from_pretrained(
            args.model_dir,
            vocab_size=len(custom_vocab_dict),
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            local_files_only=True,
        )
        model = GPT2LMHeadModel(config)
        base = torch.load(args.baseckpt, weights_only=True, map_location="cpu")
        model.load_state_dict(base["model"])
        export_dir = merge_lora_and_export(model, args.peftckpt, args.export_merged_dir)
        if is_main_process():
            print(f"Merged model exported to {export_dir}")
        cleanup_distributed()
        return

    # ---------- datasets ----------
    train_data = json.load(open(args.train_json))
    val_data = json.load(open(args.val_json))
    if args.quick_test:
        train_data = train_data[: int(args.quick_test_max_train)]
        val_data = val_data[: int(args.quick_test_max_val)]
        print(
            f"Quick test mode enabled: epochs=2, batch_size=1, train_samples={len(train_data)}, val_samples={len(val_data)}"
        )
    train_ds = CrystalStructureDataset(train_data, randomchose=True,dim=2, mode=args.stage, finetune_targetprop_list=args.finetune_target)
    val_ds   = CrystalStructureDataset(val_data,  randomchose=False,dim=2, mode=args.stage, finetune_targetprop_list=args.finetune_target)

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
    model.config.pad_token_id = custom_vocab_dict["<pad>"]
    model.config.eos_token_id = custom_vocab_dict["<eos>"]
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.baseckpt:
        base = torch.load(args.baseckpt, weights_only=True, map_location="cpu")
        model.load_state_dict(base["model"])

    if args.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets,   # ["c_attn","c_proj","c_fc"]
            bias="none"
        )
        model = get_peft_model(model, lora_cfg)
        for n, p in model.named_parameters():
            if "ln_" in n:
                p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params=: {num_trainable_params/1000:.1f}K')
    # ---- LR scaling util ------------------------------------
    BASE_EFF_BS = 32
    BASE_LR     = args.lr

    eff_bs = args.batch_size * args.acc_steps
    if dist.is_initialized():
        eff_bs *= dist.get_world_size()
    scaled_lr = BASE_LR * (eff_bs / BASE_EFF_BS) * 1

    if is_main_process():
        print(f"Effective batch = {eff_bs},  learning rate = {scaled_lr:.3e}")

    optimizer = AdamW(trainable_params if args.use_lora else model.parameters(), lr=scaled_lr, betas=(0.9,0.99),fused=True)
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
    best_val, start_epoch = 0.0, 0
    os.makedirs(args.save_dir, exist_ok=True)

    if args.peftckpt != None:
        model, start_epoch, best_val = load_checkpoint(
            model,
            args.peftckpt,
            device=str(device),
            use_lora=args.use_lora,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )
    elif args.use_lora == False:
        model, start_epoch, best_val = load_checkpoint(
            model,
            args.baseckpt,
            device=str(device),
            use_lora=args.use_lora,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )
    else:
        pass
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    
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
                # -----------------------------

            if is_main_process() and step % (args.acc_steps * 10) == 0:
                torch.cuda.empty_cache()
                prog.set_postfix(loss=f"{total_loss/step_in_epoch:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.3e}")

        # ---- validation ----
        run_eval = (epoch == 0) or ((epoch + 1) % args.valid_interval == 0)
        if run_eval:
            val_acc = evaluate(model.module if isinstance(model, DDP) else model, val_dataloader, device)
        if is_main_process():
            #print(f"Epoch {epoch+1}: val token acc = {val_acc*100:.2f}%")
            # save best model
            if run_eval and val_acc > best_val:
                best_val = val_acc
                save_path = os.path.join(args.save_dir, f"best")
                save_checkpoint(
                model,
                save_dir=save_path,
                epoch=epoch + 1,
                val_acc=best_val,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                use_lora=args.use_lora,
                tag="best",
                )
                print(f"[saved] {save_path}")
            if (10 <= (epoch+1) <=100 and (epoch+1) % 10 == 0) or ((epoch+1)>100 and (epoch+1)<=1000 and (epoch+1) % 50 == 0) or ((epoch+1)>1000 and (epoch+1) % 100 == 0):
                print('saving new weights...\n')
                save_path = os.path.join(args.save_dir, f"epoch{epoch+1}")
                save_checkpoint(
                model,
                save_dir=save_path,
                epoch=epoch + 1,
                val_acc=best_val,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                use_lora=args.use_lora,
                tag=f"e{epoch+1}",
                )
                print(f"[saved] {save_path}")
        print("Done!")
    
    cleanup_distributed()

if __name__ == '__main__':
    main()
