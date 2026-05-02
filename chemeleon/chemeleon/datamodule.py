from typing import Any, Dict

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

from chemeleon.datasets import MPDataset


class DataModule(LightningDataModule):
    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # configs for dataset
        self.dataset_name = _config["dataset_name"]
        self.data_dir = _config["data_dir"]
        print(f"dataset_name: {self.dataset_name}")
        print(f"data_dir: {self.data_dir}")

        # configs for dataloader
        self.batch_size = _config["batch_size"]
        self.num_workers = _config["num_workers"]
        self.pin_memory = _config["pin_memory"]

        # configs for chemeleon
        self.text_guide = _config["text_guide"]
        self.text_targets = _config["text_targets"]

    @property
    def dataset_cls(self) -> Dataset:
        if self.dataset_name == "mp-40":
            return MPDataset
        else:
            raise NotImplementedError(f"{self.dataset_name} should be one of mp-40")

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                split="train",
                text_guide=self.text_guide,
                text_targets=self.text_targets,
            )
            self.val_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                split="val",
                text_guide=self.text_guide,
                text_targets=self.text_targets,
            )
            self.test_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                split="test",
                text_guide=self.text_guide,
                text_targets=self.text_targets,
            )
        elif stage == "test" or stage is None:
            self.test_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                split="test",
                text_guide=self.text_guide,
                text_targets=self.text_targets,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
