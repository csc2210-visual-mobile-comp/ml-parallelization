#!/bin/bash
#SBATCH --job-name=test-deepspeed
#SBATCH --partition=gpunodes
#SBATCH --nodes=2
#SBATCH --nodelist=gpunode32,gpunode33
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# NCCL sanity (strongly recommended for multi-node)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_NET_GDR_LEVEL=2

export HF_HOME=$PWD/huggingface/home
export HF_DATASETS_CACHE=$PWD/huggingface/datasets
export TRANSFORMERS_CACHE=$PWD/huggingface/models

export TRITON_CACHE_DIR=$PWD/triton_cache/$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "Nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST")"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$SLURM_JOB_NUM_NODES"
echo "NTASKS=$SLURM_NTASKS"

# -------- run --------
srun python scripts/run_basic_deepspeed.py
