#!/bin/bash
#SBATCH --job-name=gpt2-shakespeare-pp
#SBATCH --partition=gpunodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# --------------------
# Environment
# --------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_ASYNC_ERROR_HANDLING=1

export HF_HOME=$PWD/huggingface/home
export HF_DATASETS_CACHE=$PWD/huggingface/datasets
export TRANSFORMERS_CACHE=$PWD/huggingface/models

export TRITON_CACHE_DIR=$PWD/triton_cache/$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR"

# --------------------
# torch.distributed rendezvous (Slurm provides ranks)
# --------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "Nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST")"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$SLURM_JOB_NUM_NODES"
echo "NTASKS=$SLURM_NTASKS"

# --------------------
# Launch (IMPORTANT: no torchrun, no deepspeed launcher)
# --------------------
srun --cpu-bind=none \
  python scripts/run_clm_gpt2_pipeline_parallel.py \
  --model_name_or_path gpt2-large \
  --train_file shakespeare.txt \
  --do_train \
  --block_size 512 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --fp16 \
  --logging_steps 10 \
  --deepspeed configs/pipeline_parallel.json \
  --output_dir out-gpt2-pp \
  --overwrite_output_dir
