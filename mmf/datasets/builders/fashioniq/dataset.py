import copy
import json

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset
from .database import FashionIQDatabase


class FashionIQDataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "fashioniq",
            config,
            dataset_type,
            index,
            FashionIQDatabase,
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

    def _get_valid_text_attribute(self, sample_info):
        if "captions" in sample_info:
            return "captions"

        if "sentences" in sample_info:
            return "sentences"

        raise AttributeError("No valid text attribution was found")

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        text_attr = self._get_valid_text_attribute(sample_info)

        current_sample = Sample()
        sentence = sample_info[text_attr]
        processed_sentence = self.text_processor({"text": sentence})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)
        if self._use_images:
            current_sample.ref_image = self.image_db.from_path(sample_info["ref_path"] + ".jpg")["images"][0]
            current_sample.tar_image = self.image_db.from_path(sample_info["tar_path"] + ".jpg")["images"][0]
        else:
            current_sample.ref_image = self.features_db.from_path(sample_info["ref_path"] + ".npy")["image_feature_0"]
            current_sample.tar_image = self.features_db.from_path(sample_info["tar_path"] + ".npy")["image_feature_0"]
        current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)
        current_sample.targets = None  # Dummy for Loss

        if "fake_data" in sample_info:
            current_sample.target_id = torch.tensor(sample_info['target_id'], dtype=torch.long)
            current_sample.fake_data = torch.tensor(sample_info["fake_data"], dtype=torch.bool)
            current_sample.garment_class = torch.tensor(sample_info["garment_class"], dtype=torch.long)

        return current_sample
