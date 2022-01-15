import json

import torch
from mmf.utils.file_io import PathManager


class FashionGenDatabase(torch.utils.data.Dataset):
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

        pairs = annotations_json["pairs"]
        descriptions = annotations_json["descriptions"]

        for item in pairs:
            data.append(
                {
                    "image_path": item["image"],
                    "id": item["id"],
                    "sentences": descriptions[str(item["id"])],
                }
            )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]