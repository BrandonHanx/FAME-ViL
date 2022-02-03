import copy
import json
import random

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset
from .database import FashionAllDatabase, FashionGenDatabase, Fashion200kDatabase, BigFACADDatabase, PolyvoreOutfitsDatabase


class FashionDataset(MMFDataset):
    def __init__(
        self,
        name: str,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            name,
            config,
            dataset_type,
            index,
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
        current_sample.text = sentence

        if hasattr(self, "masked_token_processor") and self._dataset_type == "train":
            processed_sentence = self.masked_token_processor({"text": sentence})
            current_sample.update(processed_sentence)
        else:
            processed_sentence = self.text_processor({"text": sentence})
            current_sample.update(processed_sentence)

        image_path = sample_info["image_path"]
        if self._dataset_type == "train":
            image_path = random.choices(image_path)[0]
            if self._use_images:
                current_sample.image = self.image_db.from_path(image_path)["images"][0]
            elif self._use_features:
                feature_path = ".".join(image_path.split(".")[:-1]) + ".npy"
                current_sample.image = self.features_db.from_path(feature_path)["image_feature_0"]
        else:
            if self._use_images:
                images = self.image_db.from_path(image_path)["images"]
                images = torch.stack(images)
                current_sample.image = images
            elif self._use_features:
                features = []
                for p in image_path:
                    feature_path = ".".join(p.split(".")[:-1]) + ".npy"
                    features.append(self.features_db.from_path(feature_path)["image_feature_0"])
                features = torch.stack(features)
                current_sample.image = features

            current_sample.text_id = torch.tensor(sample_info["id"], dtype=torch.long)
            current_sample.image_id = current_sample.text_id.repeat(len(image_path), 1)

        current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)
        current_sample.targets = torch.tensor(1, dtype=torch.long)

        if hasattr(self, "masked_image_processor"):
            current_sample.image_masks = self.masked_image_processor(current_sample.image)

        return current_sample


class FashionGenDataset(FashionDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "fashiongen",
            config,
            dataset_type,
            index,
            FashionGenDatabase,
            *args,
            **kwargs,
        )


class Fashion200kDataset(FashionDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "fashion200k",
            config,
            dataset_type,
            index,
            Fashion200kDatabase,
            *args,
            **kwargs,
        )


class BigFACADDataset(FashionDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "big_facad",
            config,
            dataset_type,
            index,
            BigFACADDatabase,
            *args,
            **kwargs,
        )


class PolyvoreOutfitsDataset(FashionDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "polyvore_outfits",
            config,
            dataset_type,
            index,
            PolyvoreOutfitsDatabase,
            *args,
            **kwargs,
        )


class FashionAllDataset(FashionDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "fashionall",
            config,
            dataset_type,
            index,
            FashionAllDatabase,
            *args,
            **kwargs,
        )
