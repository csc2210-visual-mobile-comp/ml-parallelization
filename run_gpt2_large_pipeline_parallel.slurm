{
  "train_batch_size": 4,
  "train_micro_batch_size_per_gpu": 2,

  "pipeline": {
    "enabled": true,
    "stages": 2,
    "activation_checkpoint_interval": 1
  },

  "fp16": {
    "enabled": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "zero_optimization": {
    "stage": 0
  }
}
