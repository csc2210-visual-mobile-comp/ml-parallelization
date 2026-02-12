export NCCL_P2P_DISABLE=1s
export NCCL_IB_DISABLE=1

deepspeed --num_gpus=2 scripts/run_gpt2_deepspeed_cli.py \
    --model_name_or_path gpt2 \
  --train_file shakespeare.txt \
  --do_train \
  --block_size 512 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --fp16 \
  --logging_steps 10 \
  --deepspeed runs/multi_gpu.json \
  --output_dir out-gpt2-pp \
  --profiler_path ./profiler_multi_gpu \
  --overwrite_output_dir
