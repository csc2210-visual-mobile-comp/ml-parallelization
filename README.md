# Setup
```bash
chmod +x data/download_shakespeare.sh
./data/download_shakespeare.sh deepspeed
pip install -r requirements.txt
```

# Slurm
## Configuration
`zero_opt_torchrun.json` Run with ZeRO configuration in Slurm

## Run script
```bash
cd slurm
sbatch slurm_scripts/run_gpt2_large_multi_node.slurm
```

# Run directly in Nodes
## Training
```bash
cd node_direct
sh runs/multi_gpu_deepspeed.sh
```

# View profiler result
```bash
pip install tensorboard torch_tb_profiler
torchboard --logdir=profiler
```
