from mmf.common.registry import registry
from mmf.datasets.builders.facad.dataset import FACADDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("facad")
class FACADBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="facad", dataset_class=FACADDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/facad/defaults.yaml"
