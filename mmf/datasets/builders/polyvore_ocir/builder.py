from mmf.common.registry import registry
from mmf.datasets.builders.polyvore_ocir.dataset import PolyvoreOCIRDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("polyvore_ocir")
class PolyvoreOCIRBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="polyvore_ocir", dataset_class=PolyvoreOCIRDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/polyvore_ocir/defaults.yaml"
