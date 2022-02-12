import os
import json
import random

import torch
from mmf.utils.file_io import PathManager


CAT_DICT = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    15: 9,
    17: 10,
    18: 11,
    19: 12,
    21: 13,
    23: 14,
    24: 15,
    25: 16,
    26: 17,
    27: 18,
    28: 19,
    29: 20,
    30: 21,
    31: 22,
    35: 23,
    36: 24,
    37: 25,
    38: 26,
    39: 27,
    40: 28,
    41: 29,
    42: 30,
    43: 31,
    46: 32,
    47: 33,
    48: 34,
    49: 35,
    50: 36,
    51: 37,
    52: 38,
    53: 39,
    55: 40,
    56: 41,
    57: 42,
    58: 43,
    59: 44,
    61: 45,
    62: 46,
    64: 47,
    65: 48,
    68: 49,
    69: 50,
    1606: 51,
    1607: 52,
    1605: 53,
    104: 54,
    105: 55,
    106: 56,
    107: 57,
    231: 58,
    236: 59,
    237: 60,
    238: 61,
    239: 62,
    240: 63,
    241: 64,
    243: 65,
    244: 66,
    248: 67,
    249: 68,
    250: 69,
    251: 70,
    252: 71,
    253: 72,
    254: 73,
    255: 74,
    256: 75,
    257: 76,
    258: 77,
    259: 78,
    260: 79,
    261: 80,
    262: 81,
    263: 82,
    264: 83,
    265: 84,
    266: 85,
    267: 86,
    268: 87,
    269: 88,
    270: 89,
    272: 90,
    273: 91,
    275: 92,
    277: 93,
    278: 94,
    279: 95,
    280: 96,
    281: 97,
    282: 98,
    284: 99,
    286: 100,
    287: 101,
    288: 102,
    289: 103,
    290: 104,
    291: 105,
    292: 106,
    293: 107,
    294: 108,
    295: 109,
    296: 110,
    297: 111,
    298: 112,
    299: 113,
    300: 114,
    301: 115,
    302: 116,
    303: 117,
    304: 118,
    305: 119,
    306: 120,
    307: 121,
    309: 122,
    310: 123,
    315: 124,
    318: 125,
    332: 126,
    341: 127,
    342: 128,
    343: 129,
    4452: 130,
    4454: 131,
    4455: 132,
    4456: 133,
    4457: 134,
    4458: 135,
    4459: 136,
    4460: 137,
    4461: 138,
    4462: 139,
    4463: 140,
    4464: 141,
    4465: 142,
    4466: 143,
    4467: 144,
    4468: 145,
    4470: 146,
    4472: 147,
    4473: 148,
    4495: 149,
    4496: 150,
    4517: 151,
    4518: 152
}


class PolyvoreOCIRDatabase(torch.utils.data.Dataset):
    SPLITS = {"train": "train", "val": "test", "test": "test"}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self._load_annotation_db(splits_path)

    def _load_annotation_db(self, splits_path):
        data = []

        with PathManager.open(splits_path, "r") as f:
            annotations_json = json.load(f)

        head_path, _ = os.path.split(splits_path)
        meta_path = os.path.join(head_path, "polyvore_item_metadata.json")

        with PathManager.open(meta_path, "r") as f:
            meta = json.load(f)

        if self.splits == "train":
            negative_pool = {}
            for item in annotations_json:
                for x in item["question"] + [item["blank"]]:
                    cat_id = meta[x]["category_id"]
                    if cat_id in negative_pool.keys():
                        negative_pool[cat_id].append(x)
                    else:
                        negative_pool[cat_id] = [x]
            negative_pool = {k: list(set(v)) for k, v in negative_pool.items()}

            for item in annotations_json:
                question_id, blank_id = random.choices(item["question"] + [item["blank"]], k=2)
                # question_id = random.choices(item["question"])[0]
                # blank_id = item["blank"]
                question_cat_id = meta[question_id]["category_id"]
                blank_cat_id = meta[blank_id]["category_id"]
                data.append(
                    {
                        "question_path": question_id + ".jpg",
                        "blank_path": blank_id + ".jpg",
                        "question_id": int(question_id),
                        "blank_id": int(blank_id),
                        "question_cat_id": CAT_DICT[int(question_cat_id)],
                        "blank_cat_id": CAT_DICT[int(blank_cat_id)],
                        "negative_path": random.choices(negative_pool[blank_cat_id])[0] + ".jpg",
                    }
                )
        else:
            for item in annotations_json:
                question_id = item["question"]
                blank_id = item["blank"]
                question_cat_id = [CAT_DICT[int(meta[x]["category_id"])] for x in question_id]
                blank_cat_id = CAT_DICT[int(meta[blank_id]["category_id"])]
                data.append(
                    {
                        "question_path": [x + ".jpg" for x in question_id],
                        "blank_path": blank_id + ".jpg",
                        "question_id": [int(x) for x in question_id],
                        "blank_id": int(blank_id),
                        "question_cat_id": question_cat_id,
                        "blank_cat_id": int(blank_cat_id),
                        "fake_data": item["fake_data"],
                    }
                )

        if len(data) == 0:
            raise RuntimeError("Dataset is empty")

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
