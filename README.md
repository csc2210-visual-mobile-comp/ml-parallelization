## Setup
```bash
chmod +x data/download_shakespeare.sh
./data/download_shakespeare.sh
pip install -r requirements.txt
```

## Training
```bash
sbatch run_gpt2.slurm
```

## Inference
```bash
sbatch run_generate.slurm
```

## Configuration
`multi_gpu.json` DeepSpeed config for having multi GPU in a node setup
`single_gpu.json` DeepSpeed config for just a single GPU 

## Slurm file
`run_generate.slurm` For inference in slurm cluster once model is trained, specify the model directory
`run_gpt2_multi_node.slurm` To run data parallel in 2 nodes with 1 GPU each
`run_gpt2_single_gpu.slurm` Single GPU single node

## View profiler result
```bash
pip install tensorboard torch_tb_profiler
torchboard --logdir=profiler
```
