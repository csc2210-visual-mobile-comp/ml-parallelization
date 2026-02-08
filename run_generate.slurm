#!/bin/bash
#SBATCH --job-name=gpt2-infer
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/infer-%j.out

python scripts/generate.py