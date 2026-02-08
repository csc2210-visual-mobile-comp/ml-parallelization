#!/bin/bash
#SBATCH --job-name=gpt2-shakespeare-dp
#SBATCH --partition=gpunodes
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpunode32,gpunode33
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "Nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST")"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "SLURM_NTASKS=$SLURM_NTASKS SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"

srun --label torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/run_clm.py \
  --model_name_or_path gpt2-large \
  --train_file shakespeare.txt \
  --do_train \
  --block_size 512 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --weight_decay 0.01 \
  --fp16 \
  --logging_steps 50 \
  --output_dir out-gpt2-dp \
  --overwrite_output_dir
