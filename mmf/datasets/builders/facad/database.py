import json

import pandas as pd
import torch
from mmf.utils.file_io import PathManager


class FACADDatabase(torch.utils.data.Dataset):
    SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self._load_annotation_db(splits_path)

    def _load_annotation_db(self, splits_path):
        data = []

        with PathManager.open(splits_path, "r") as f:
            annotations_json = json.load(f)

        if self.config.use_patch_labels and self.dataset_type == "train":
            with PathManager.open(splits_path.replace("_info", "_codes"), "r") as f:
                patch_label_json = json.load(f)
            for item in annotations_json:
                data.append(
                    {
                        "image_path": item["image"],
                        "sentences": item["color"] + " " + item["title"],
                        "patch_labels": patch_label_json[item["image"].split(".")[0]],
                    }
                )
        else:
            for item in annotations_json:
                data.append(
                    {
                        "image_path": item["image"],
                        "sentences": item["color"] + " " + item["title"],
                    }
                )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]