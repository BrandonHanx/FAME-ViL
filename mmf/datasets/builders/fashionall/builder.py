from mmf.common.registry import registry
from .dataset import FashionAllDataset, FashionGenDataset, FashionGenCAPDataset, FashionGenCLSDataset, Fashion200kDataset, BigFACADDataset, PolyvoreOutfitsDataset
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


@registry.register_builder("fashiongen_cls")
class FashionGenCLSBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="fashiongen_cls", dataset_class=FashionGenCLSDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/fashiongen/classification.yaml"


@registry.register_builder("fashiongen_cap")
class FashionGenCAPBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="fashiongen_cls", dataset_class=FashionGenCAPDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/fashiongen/captioning.yaml"


@registry.register_builder("fashion200k")
class Fashion200kBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="fashion200k", dataset_class=Fashion200kDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/fashion200k/defaults.yaml"


@registry.register_builder("big_facad")
class BigFACADBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="big_facad", dataset_class=BigFACADDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/big_facad/defaults.yaml"


@registry.register_builder("polyvore_outfits")
class PolyvoreOutfitsBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="polyvore_outfits", dataset_class=PolyvoreOutfitsDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/polyvore_outfits/defaults.yaml"


@registry.register_builder("fashionall")
class FashionAllBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="fashionall", dataset_class=FashionAllDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/fashionall/defaults.yaml"
