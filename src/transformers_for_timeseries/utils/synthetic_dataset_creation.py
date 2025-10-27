import torch
import numpy as np
from torch.utils.data import Dataset, random_split

from transformers_for_timeseries.config_and_scripts.n1_settings import config


class forced_pendulum_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        observations = torch.tensor(sample['angle'], dtype=torch.bfloat16)
        questions = torch.tensor(sample['forcing'], dtype=torch.bfloat16)
        params = torch.tensor([sample['kapa'], sample['beta']], dtype=torch.bfloat16)
        return observations, questions, params
    


def build_synthetic_damped_forced_pendulum_dataset() -> None:
    path = config.DIR_RAW_DATA / "synthetic_damped_forced_pendulum_dataset.npy"
    dataset = np.load(path, allow_pickle=True)

    split_ratio = config.SPLIT_RATIO  # Train, Validation, Test

    train_size = int(split_ratio[0] * len(dataset))
    val_size = int(split_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    Dataset_custom = forced_pendulum_dataset(dataset)

    train_dataset, valid_dataset, test_dataset = random_split(Dataset_custom, [train_size, val_size, test_size])

    path = config.DIR_PREPROCESSED_DATA
    torch.save(train_dataset, path / "train_dataset.pt")
    torch.save(valid_dataset, path / "valid_dataset.pt")
    torch.save(test_dataset, path / "test_dataset.pt")

    return None



if __name__ == "__main__":

    build_synthetic_damped_forced_pendulum_dataset()