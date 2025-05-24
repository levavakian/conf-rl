# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is a research project for Reinforcement Learning from Human Feedback (RLHF) using Qwen language models with Sparse Autoencoder (SAE) analysis. The project has two main components:

### 1. RLHF Training Pipeline (`main.py`)
- Uses Unsloth for efficient model loading and LoRA fine-tuning
- Implements Group-Relative Policy Optimization (GRPO) via TRL library
- Trains Qwen3-14B model on GSM8K math problems with structured XML output format
- Multiple reward functions enforce correctness, formatting, and numerical validity
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### 2. SAE Training Pipeline (`activations.py` → `sae.py`)
- **Step 1**: `activations.py` extracts residual stream activations from layer 20 during model inference
- **Step 2**: `sae.py` trains a Sparse Autoencoder on extracted activations using PyTorch Lightning
- Activations saved to `qwen_acts_idx/` directory as sharded `.pt` files
- SAE checkpoints saved to `sae_checkpoints/` directory

## Common Development Commands

### Training RLHF Model
```bash
python main.py
```

### Generating Activations
```bash
python activations.py
```

### Training SAE
```bash
python sae.py
```

### Monitoring Training
```bash
tensorboard --logdir lightning_logs/
```

## Key Configuration Constants

### RLHF Training (`main.py`)
- `max_seq_length = 1024`
- `lora_rank = 128`
- Model: `Qwen/Qwen3-14B`
- Learning rate: `5e-6`
- Batch size: 1 with 4 gradient accumulation steps

### Activation Extraction (`activations.py`)
- `LAYER = 20` (target transformer layer)
- `SEQ_LEN = 8192`
- `BATCH_SIZE = 4`
- `OUTPUT_DIR = "qwen_acts_idx"`

### SAE Training (`sae.py`)
- `D_MODEL = 5120` (Qwen3-14B hidden size)
- `EXPANSION_FACTOR = 8`
- `L1_COEFF = 3e-4`
- `BATCH_SIZE = 4096`

## Data Flow

1. GSM8K questions → RLHF training with XML-structured responses
2. Model inference → activation extraction at layer 20
3. Extracted activations → SAE training for interpretability analysis
4. Both pipelines use checkpointing for resumable training

## Dependencies

Project uses `uv` package manager with PyTorch, Unsloth, TRL, Transformer Lens, and PyTorch Lightning. Flash Attention requires special build configuration (see `pyproject.toml`).