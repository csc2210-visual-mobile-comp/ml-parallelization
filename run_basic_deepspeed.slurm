#!/bin/bash
#SBATCH --job-name=ds-pipeline-2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 demonstrates
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --output=ds_%j.out
#SBATCH --error=ds_%j.err

# -------- environment --------
module load cuda        # if your cluster uses modules
source .venv/bin/activate

# NCCL sanity (strongly recommended for multi-node)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# -------- run --------
srun deepspeed pipeline_lr.py
