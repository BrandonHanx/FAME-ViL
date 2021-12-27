from mmf.common.registry import registry
from mmf.datasets.builders.fashioniq.dataset import FashionIQDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("fashioniq")
class FashionIQBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="fashioniq", dataset_class=FashionIQDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/fashioniq/defaults.yaml"
