# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from mmf.modules.encoders import VQVAEEncoder
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.total_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_imgs[idx].split(".")[0]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_json(content, fname):
    # print("writing features to {}".format(fname))
    with open(fname, "wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False, cls=NpEncoder)


def extract_dataset(data_dir, out_json):
    use_cuda = torch.cuda.is_available()

    config = VQVAEEncoder.Config(
        pretrained_path="data/pretrained_models/vqvae_ema_pp_224_7x7_encoder.pth",
        resolution=224,
        num_tokens=1024,
        codebook_dim=256,
        attn_resolutions=[7],
        hidden_dim=128,
        in_channels=3,
        ch_mult=[1, 1, 2, 2, 4, 4],
        num_res_blocks=2,
        dropout=0,
        z_channels=256,
        double_z=False,
    )

    vqvae_encoder = VQVAEEncoder(config)
    vqvae_encoder = vqvae_encoder.eval()

    if use_cuda:
        vqvae_encoder = vqvae_encoder.cuda()

    my_dataset = MyDataset(data_dir)
    data_loader = DataLoader(
        my_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False
    )

    name_list = []
    indices = []
    for item in tqdm(data_loader):
        image, name = item
        if use_cuda:
            image = image.cuda()

        with torch.no_grad():
            _, _, idx = vqvae_encoder(image)

        indices.append(idx)
        name_list.extend(list(name))

    indices = torch.cat(indices, dim=0)
    indices = indices.long().cpu().numpy()
    write_json(dict(zip(name_list, indices)), out_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)

    args = parser.parse_args()

    extract_dataset(args.data_dir, args.out_json)
