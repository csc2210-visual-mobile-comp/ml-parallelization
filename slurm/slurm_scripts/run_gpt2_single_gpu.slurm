#!/bin/bash
#SBATCH --job-name=gpt2-shakespeare
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpunode32
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# (helps debugging if something hangs)
export NCCL_DEBUG=INFO

export DEEPSPEED_COMM_BACKEND=nccl
export DEEPSPEED_DISABLE_MPI=1

srun --label torchrun \
  --nproc_per_node=1 \
  scripts/run_clm.py \
  --model_name_or_path gpt2 \
  --train_file shakespeare.txt \
  --do_train \
  --block_size 512 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --weight_decay 0.01 \
  --fp16 \
  --logging_steps 50 \
  --output_dir out-gpt2-shakespeare \
  --overwrite_output_dir
