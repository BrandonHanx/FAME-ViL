import copy
import json
import random

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset
from .database import FACADDatabase


class FACADDataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "facad",
            config,
            dataset_type,
            index,
            FACADDatabase,
            *args,
            **kwargs,
        )
        self._false_caption = config.get("false_caption", True)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)

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

    def _get_mismatching_caption(self, idx):
        random_idx = random.randint(0, len(self.annotation_db) - 1)
        while random_idx == idx:
            random_idx = random.randint(0, len(self.annotation_db) - 1)

        other_item = self.annotation_db[random_idx]
        return other_item

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        text_attr = self._get_valid_text_attribute(sample_info)

        current_sample = Sample()
        is_correct = 1
        if self._false_caption and self._dataset_type == "train" and random.random() < self._false_caption_probability:
            sentence = self._get_mismatching_caption(idx)[text_attr]
            is_correct = 0
        else:
            sentence = sample_info[text_attr]
        current_sample.text = sentence

        if hasattr(self, "masked_token_processor") and self._dataset_type == "train":
            processed_sentence = self.masked_token_processor({"text": sentence})
            current_sample.update(processed_sentence)
        else:
            processed_sentence = self.text_processor({"text": sentence})
            current_sample.update(processed_sentence)

        if self._use_images:
            current_sample.image = self.image_db[idx]["images"][0]
        else:
            current_sample.image = self.features_db[idx]["image_feature_0"]

        if hasattr(self, "masked_image_processor"):
            current_sample.image_masks = self.masked_image_processor(current_sample.image)
        current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)
        current_sample.targets = torch.tensor(is_correct, dtype=torch.long)

        if "patch_labels" in sample_info:
            current_sample.patch_labels = torch.tensor(sample_info["patch_labels"], dtype=torch.long)

        return current_sample
