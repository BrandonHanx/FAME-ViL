import os
import json

import torch
from mmf.utils.file_io import PathManager


class UIGRDatabase(torch.utils.data.Dataset):
    SPLITS = {"train": "train", "val": "val", "test": "test"}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.task_type = config.get("task_type", "comp")
        assert self.task_type in ["comp", "outfit"]
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self._load_annotation_db(splits_path)

    def _load_annotation_db(self, splits_path):
        data = []
        json_name = "{}_triplets_{}.json".format(self.task_type, self.splits)
        with PathManager.open(os.path.join(splits_path, json_name), "r") as f:
            annotations_json = json.load(f)

        if self.splits == "train":
            for item in annotations_json:
                data.append(
                    {
                        "ref_path": item["candidate"],
                        "tar_path": item["target"],
                        "sentences": ", ".join(item["captions"]),
                    }
                )
        else:
            for item in annotations_json:
                data.append(
                    {
                        "ref_path": item["candidate"],
                        "tar_path": item["target"],
                        "sentences": ", ".join(item["captions"]),
                        "target_id": item["target_id"],
                        "fake_data": item["fake_data"],
                    }
                )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]