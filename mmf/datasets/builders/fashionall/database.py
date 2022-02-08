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
        self.data = self._load_annotation_db(splits_path)

    @staticmethod
    def _load_annotation_db(splits_path):
        data = []

        with PathManager.open(splits_path, "r") as f:
            annotations_json = json.load(f)

        for item in annotations_json:
            data.append(
                {
                    "image_path": item["images"],
                    "id": item["id"],
                    "sentences": item["title"] + "." + item["description"],
                    "attributes_id": item["attributes_id"],
                    "category_id": item["category_id"],
                    "subcategory_id": item["subcategory_id"],
                }
            )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BigFACADDatabase(torch.utils.data.Dataset):
    SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self.data = self._load_annotation_db(splits_path)

    @staticmethod
    def _load_annotation_db(splits_path):
        data = []

        with PathManager.open(splits_path, "r") as f:
            annotations_json = json.load(f)

        for item in annotations_json:
            data.append(
                {
                    "image_path": item["images"],
                    "id": item["id"],
                    "sentences": item["color"] + "." + item["title"] + "." + item["description"],
                    "attributes_id": item["attributes_id"],
                }
            )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Fashion200kDatabase(torch.utils.data.Dataset):
    SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self.data = self._load_annotation_db(splits_path)

    @staticmethod
    def _load_annotation_db(splits_path):
        data = []

        with PathManager.open(splits_path, "r") as f:
            annotations_json = json.load(f)

        for item in annotations_json:
            data.append(
                {
                    "image_path": item["images"],
                    "id": item["id"],
                    "sentences": item["title"],
                    "attributes_id": item["attributes_id"],
                }
            )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PolyvoreOutfitsDatabase(torch.utils.data.Dataset):
    SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self.data = self._load_annotation_db(splits_path)

    @staticmethod
    def _load_annotation_db(splits_path):
        data = []

        with PathManager.open(splits_path, "r") as f:
            annotations_json = json.load(f)

        for item in annotations_json:
            data.append(
                {
                    "image_path": [item["images"]],
                    "id": item["id"],
                    "sentences": item["title"] + "." + item["description"],
                    "attributes_id": item["attributes_id"],
                }
            )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FashionAllDatabase(torch.utils.data.Dataset):
    SPLITS = {"train": ["train"], "val": ["val"], "test": ["test"]}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self.data = self._load_annotation_db(splits_path)

    @staticmethod
    def _load_annotation_db(splits_path):
        if "BigFACAD" in splits_path:
            return BigFACADDatabase._load_annotation_db(splits_path)
        elif "Fashion200k" in splits_path:
            return Fashion200kDatabase._load_annotation_db(splits_path)
        elif "FashionGen" in splits_path:
            return FashionGenDatabase._load_annotation_db(splits_path)
        elif "PolyvoreOutfits" in splits_path:
            return PolyvoreOutfitsDatabase._load_annotation_db(splits_path)
        else:
            raise FileNotFoundError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
