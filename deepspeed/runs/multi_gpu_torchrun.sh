CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/run_clm.py \
  --model_name_or_path gpt2 \
  --train_file shakespeare.txt \
  --do_train \
  --profiler_path profiler/single_node_2gpu \
  --block_size 512 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --weight_decay 0.01 \
  --fp16 \
  --logging_steps 50 \
  --output_dir out-gpt2-dp \
  --overwrite_output_dir
