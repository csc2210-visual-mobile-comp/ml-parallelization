import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import deepspeed
from deepspeed.pipe import PipelineModule


# -------------------------
# Dataset
# -------------------------
class LinearRegressionDataset(Dataset):
    def __init__(self, n_samples=1000, a=2.0, b=1.0, noise_std=0.1):
        self.x = torch.randn(n_samples, 1)
        noise = noise_std * torch.randn(n_samples, 1)
        self.y = a * self.x + b + noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_dataloader(batch_size=32, n_samples=1000):
    dataset = LinearRegressionDataset(n_samples=n_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -------------------------
# Model dimensions
# -------------------------
in_features = 1
hidden_dim = 10
out_features = 1


# -------------------------
# Main
# -------------------------
def main():
    # Pipeline layers must be a LIST
    layers = [
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_features),
    ]

    pipe = PipelineModule(
        layers=layers,
        loss_fn=nn.MSELoss(),
        num_stages=2,
        partition_method="parameters",
        activation_checkpoint_interval=0,
    )

    # Minimal DeepSpeed config
    ds_config = {
        "train_batch_size": 32,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-2
            }
        },
        "pipeline": {
            "seed_layers": True
        }
    }

    engine, _, _, _ = deepspeed.initialize(
        model=pipe,
        model_parameters=pipe.parameters(),
        config=ds_config,
    )

    train_loader = make_dataloader(batch_size=32)
    train_iter = iter(train_loader)

    for step in range(20):
        loss = engine.train_batch(data_iter=train_iter)
        if engine.global_rank == 0:
            print(f"step {step}, loss = {loss:.6f}")


if __name__ == "__main__":
    main()
