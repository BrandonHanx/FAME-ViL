import copy
import json

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset
from .database import PolyvoreOCIRDatabase


class PolyvoreOCIRDataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "polyvore_ocir",
            config,
            dataset_type,
            index,
            PolyvoreOCIRDatabase,
            *args,
            **kwargs,
        )

    def init_processors(self):
        super().init_processors()
        if self._use_images:
            # Assign transforms to the image_db
            if self._dataset_type == "train":
                self.image_db.transform = self.train_image_processor
            else:
                self.image_db.transform = self.eval_image_processor

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]

        current_sample = Sample()
        current_sample.blank_image = self.image_db.from_path(sample_info["blank_path"])["images"][0]
        current_sample.blank_id = torch.tensor(sample_info["blank_id"], dtype=torch.long)
        current_sample.blank_cat_id = torch.tensor(sample_info["blank_cat_id"], dtype=torch.long)

        current_sample.question_id = torch.tensor(sample_info["question_id"], dtype=torch.long)
        current_sample.question_cat_id = torch.tensor(sample_info["question_cat_id"], dtype=torch.long)
        if self._dataset_type == "train":
            current_sample.question_image = self.image_db.from_path(sample_info["question_path"])["images"][0]
            current_sample.negative_image = self.image_db.from_path(sample_info["negative_path"])["images"][0]
            current_sample.combine_cat_id = torch.tensor(
                [sample_info["question_cat_id"], sample_info["blank_cat_id"]],
                dtype=torch.long
            )
            current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)
        else:
            q_len = len(current_sample.question_id)
            question_image = self.image_db.from_path(sample_info["question_path"])["images"]
            current_sample.question_image = torch.stack(question_image)
            combine_cat_id = []
            for id in sample_info["question_cat_id"]:
                combine_cat_id.append([id, sample_info["blank_cat_id"]])
            current_sample.combine_cat_id = torch.tensor(combine_cat_id, dtype=torch.long)
            current_sample.fake_data = torch.tensor(sample_info["fake_data"], dtype=torch.bool)
            current_sample.ann_idx = torch.tensor([idx] * q_len, dtype=torch.long)

        current_sample.targets = None  # Dummy for Loss

        return current_sample
