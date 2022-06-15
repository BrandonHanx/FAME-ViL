from mmf.common.registry import registry
from mmf.datasets.builders.uigr.dataset import UIGRDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("uigr_tgr")
class UIGRBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="uigr_tgr", dataset_class=UIGRDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/uigr/tgr.yaml"


@registry.register_builder("uigr_vcr")
class UIGRBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="uigr_vcr", dataset_class=UIGRDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/uigr/vcr.yaml"
