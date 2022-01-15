from mmf.common.registry import registry
from mmf.datasets.builders.fashiongen.dataset import FashionGenDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("fashiongen")
class FashionGenBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="fashiongen", dataset_class=FashionGenDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/fashiongen/defaults.yaml"
