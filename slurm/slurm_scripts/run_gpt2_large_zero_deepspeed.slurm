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

# --------------------
# Environment
# --------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker

export HF_HOME=$PWD/huggingface/home
export HF_DATASETS_CACHE=$PWD/huggingface/datasets
export TRANSFORMERS_CACHE=$PWD/huggingface/models

export TRITON_CACHE_DIR=$PWD/triton_cache/$SLURM_JOB_ID
mkdir -p $TRITON_CACHE_DIR

# --------------------
# torchrun rendezvous
# --------------------
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "Nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST")"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$SLURM_JOB_NUM_NODES"

# --------------------
# Launch
# --------------------
srun torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/run_clm_zero_deepspeed.py \
  --model_name_or_path gpt2-large \
  --train_file shakespeare.txt \
  --do_train \
  --block_size 512 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --fp16 \
  --deepspeed configs/zero_opt_deepspeed.json \
  --output_dir out-gpt2-pp \
  --overwrite_output_dir
