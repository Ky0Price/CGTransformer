# CGTransformer: A Topology-Aware Autoregressive Framework for Crystal Generation and Inverse Design

Official code release for CGTransformer.

This repository contains the main workflow used in the paper, including dataset construction from trajectory files, autoregressive pretraining, property-conditioned finetuning with optional LoRA adapters, conditional crystal generation, and benchmark evaluation.

## Table of Contents

- [Getting Started](#getting-started)
- [Repository Layout](#repository-layout)
- [Data and Checkpoints](#data-and-checkpoints)
- [Basic CLI Usage](#basic-cli-usage)
- [Quick Local Test](#quick-local-test)
- [Configuration Notes](#configuration-notes)
- [Citation](#citation)
- [License](#license)

## Getting Started

### Prerequisites

- Python 3.10 is recommended.
- PyTorch 2.5.1 is the recommended version for this project.
- CUDA 11.8 is the recommended GPU runtime.

### Create a Local Environment

We recommend using a clean virtual environment or Conda environment before installing the dependencies for this project.

### Install PyTorch

Install PyTorch separately first. For the recommended CUDA 11.8 setup:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

If you need a CPU-only build or a different CUDA version, follow the official instructions from the PyTorch installation guide.

### Install the Remaining Dependencies

After PyTorch is available in your environment, install the remaining Python packages:

```bash
pip install -r requirements.txt
```

## Repository Layout

- `scripts/` - entry scripts for training, generation, and evaluation
- `configs/` - example YAML configs for every public entry script
- `dataset/` - processed JSON datasets and related local resources
- `checkpoints/` - base models, finetuned checkpoints, and adapters
- `env/` - exported environment snapshots kept only as reference

## Data and Checkpoints

This repository is organized as a script-based paper codebase. Large datasets and trained weights may be distributed separately from the Git repository.

Before running any workflow, make sure the paths referenced in your config file exist locally:

- processed training and validation JSON files for training
- base model files under `checkpoints/basegpt`
- full-model checkpoints and optional PEFT adapter directories for finetuning or generation

The public scripts and config files already assume these directory names. If your local layout differs, update the corresponding config paths before running.

## Basic CLI Usage

All public entry scripts use a YAML config file passed with `--config`. Start from the matching `*.example.yaml` file in `configs/`. The example configs contain the detailed parameter comments; the commands below show the standard entry points only.

### General Notes Before Running

- Paths in the example configs are written relative to the config file location.
- It is usually better to copy an example config and edit the copy for your own run.
- Keep dataset paths, checkpoint paths, and output paths explicit to avoid mixing experiments.
- For quick local debugging, prefer the built-in smoke-test settings before launching a long run.

### 1. Dataset Construction

Convert trajectory files into the JSON format expected by the training scripts:

```bash
python scripts/graph_module/ats2graph.py --config configs/ats2graph.example.yaml
```

Before running:

- point `traj_dir` to the directory containing the source `.traj` file
- set `target` to the trajectory stem, for example `train_mp` for `train_mp.traj`
- common ordering methods are `"5"` for ESG and `"8"` for ESP

After running:

- the serialized dataset JSON is written to `dataset/` by default unless `out_json` is set explicitly
- failed cases can be recorded to a separate log file through `fail_log`

### 2. Pretraining

Launch autoregressive pretraining on serialized crystal text:

```bash
python scripts/pretrain.py --config configs/pretrain.example.yaml
```

Before running:

- confirm that `train_json`, `val_json`, and `model_dir` point to valid local files
- keep `stage: pretrain` for standard pretraining
- use `quick_test: true` for a short local sanity check before long training jobs

After running:

- checkpoints are written to `save_dir`
- the quick-test mode automatically uses a very small setup for fast bug checking

### 3. Finetuning / LoRA Finetuning

Run property-conditioned finetuning or LoRA adapter training:

```bash
python scripts/finetune.py --config configs/finetune.example.yaml
```

Before running:

- `baseckpt` should point to the full-model checkpoint used as the starting point
- set `use_lora: true` for adapter finetuning or `false` for full finetuning
- `peftckpt` is optional and is mainly used to resume LoRA training or to export a merged standalone model
- use `quick_test: true` for a short local verification run

After running:

- full finetuning writes standard checkpoints to `save_dir`
- LoRA finetuning writes adapter artifacts such as `adapter_model.safetensors` and `adapter_config.json`
- if `export_merged_dir` is set, the script merges the base checkpoint and LoRA adapter into a standalone Hugging Face model directory and exits

### 4. Generation

Generate structures from a finetuned checkpoint:

```bash
python scripts/generate.py --config configs/generate.example.yaml
```

Before running:

- `pretrained_model_path` should point to the base GPT directory, typically `checkpoints/basegpt`
- `baseckpt` selects the main trained checkpoint
- leave `peftckpt` empty if you want to generate from a full-model checkpoint without a LoRA adapter
- property constraints, species constraints, and sampling settings are configured under `gen_para`

After running:

- the main trajectory output is written as `gen.traj`
- additional exported structures may be written to `cifs/`, `xyzs/`, or `vasps/` depending on the selected format
- all generation outputs are collected under `output_path`

### 5. Unconditional Benchmark

Evaluate generated structures with the unconditional benchmark workflow:

```bash
python scripts/benchmark_unconditional.py --config configs/benchmark_unconditional.example.yaml
```

Before running:

- make sure the config points to a valid generated trajectory file
- check the reference dataset paths used for the benchmark
- use the example config as the starting template for output locations and evaluation settings

After running:

- the script writes benchmark outputs to the paths defined in the config file

### 6. CSP Benchmark

Evaluate paired conditional generation or crystal structure prediction outputs:

```bash
python scripts/benchmark_csp.py --config configs/benchmark_csp.example.yaml
```

Before running:

- prepare the generated and target trajectory files expected by the config
- keep the pairing between predictions and references consistent

After running:

- the script writes paired evaluation results to the paths defined in the config file

## Quick Local Test

Before uploading changes or launching long training jobs, we recommend running a short local smoke test.

For `pretrain.py` and `finetune.py`, set `quick_test: true` in the corresponding example config and run the standard command. In quick-test mode, the scripts automatically switch to a very small setup intended for fast debugging:

- `epochs = 2`
- `batch_size = 1`
- `acc_steps = 1`
- at most 20 training samples and 20 validation samples
- `finetune.py` also validates every epoch in quick-test mode

This mode is designed for sanity checking the end-to-end pipeline rather than reproducing the full paper results.

## Configuration Notes

- Detailed parameter explanations are intentionally kept in the example config files under `configs/`.
- When adapting the repository to a new dataset or checkpoint layout, update the config paths first rather than editing the scripts.
- Relative paths in configs are resolved from the directory containing the config file.
- If you rename checkpoint folders, update all related config values to keep the workflow consistent.

## Citation

If you use this repository, please cite the corresponding paper.

Title: CGTransformer: A Topology-Aware Autoregressive Framework for Crystal Generation and Inverse Design

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
