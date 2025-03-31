import json
import os
import pickle
import random
import re

import cv2
import numpy as np
import torch
from PIL import Image
from easydict import EasyDict
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import _resize_pil_image, load_16big_png_depth, _resize_np_image, _seq_name_to_seed, process_prompt


class Real10kDataset(Dataset):
    def __init__(self, dataset_root, data_list, nframe, height, width,
                 rank=0, mode="train", min_interval=1, max_interval=8, **kwargs):
        super(Real10kDataset, self).__init__()
        self.dataset_root = dataset_root
        self.height = height
        self.width = width
        self.global_data_list = []  # 全局index，每个epoch不会改变
        self.nframe = nframe
        self.mode = mode
        self.global_tag = "real10k"
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.n_samples_per_subset = kwargs.get("n_samples_per_subset", -1)
        self.random_frame_sample = kwargs.get("random_frame_sample", True)
        self.sort_frames = kwargs.get("sort_frames", False)
        self.camera_longest_side = kwargs.get("camera_longest_side", None)
        self.sample_whole_scene = kwargs.get("sample_whole_scene", False)
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.CenterCrop(size=(height, width)),
        #                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.prefix = "train" if mode == "train" else "test"

        self.global_seq_dict = pickle.load(open("./data/real10k/global_seq_dict.pkl", "rb"))
        self.total_num = 0
        with open(data_list, "r") as f:
            for line in tqdm(f, desc=f"loading global real10k {mode} sequences...", disable=rank != 0):
                class_name = line.strip()
                self.global_data_list.append(class_name)
                self.total_num += len(self.global_seq_dict[class_name])
        self.caption = f"./data/mv_data_captions/{self.global_tag}"
        self.enable_depth = kwargs.get("enable_depth", False)
        self.depth_root = f"./data/multi_view_depthanythingv2/{self.global_tag}"
        self.align_depth = kwargs.get("align_depth", False)
        self.align_scale_root = f"./data/multi_view_depthanythingv2_scaleshift/{self.global_tag}"
        self.scale_map = None

        self.data_list_per_epoch, self.seq_list_per_epoch = [], []

        # for training, it should be set in the sampler for each epoch
        # for testing, directly loading from global_data_list
        if mode == 'val':
            self.reset_dataset(mode=mode, rank=rank)

    def __len__(self):
        return len(self.data_list_per_epoch)

    def reset_dataset(self, mode="train", rank=0, **kwargs):
        file_list = []
        sequence_index = []
        random_seed = kwargs.get("random_seed", 123)

        class_names = self.global_data_list.copy()
        if self.n_samples_per_subset < 1:
            # real10k数据占比太多，训练的时候对场景也要进行采样
            random.shuffle(class_names)
            class_names = class_names[::int(1 / self.n_samples_per_subset)]
            n_samples_per_subset = 1
        else:
            n_samples_per_subset = self.n_samples_per_subset

        # n_samples_per_subset用于限制在scannet中总共取多少组frames
        # 每组由相同的interval组成，不同组之间interval不一样
        min_range = self.nframe * self.min_interval
        max_range = self.nframe * self.max_interval
        for i, class_name in enumerate(class_names):
            seq_indices = self.global_seq_dict[class_name].copy()  # 首先获取某个场景图片数量
            seq_indices = [f"{class_name}/{s}" for s in seq_indices]
            if not self.sample_whole_scene:
                splits = []
                st = 0
                while len(splits) == 0 or splits[-1][1] < len(seq_indices) - min_range:
                    if mode == "train":
                        # splits.append([st, st + random.randint(min_range, max_range)])
                        splits.append([st, st + random.Random(random_seed + i * st).randint(min_range, max_range)])
                    else:
                        splits.append([st, st + int((min_range + max_range) / 2)])
                    st = splits[-1][1] + 1

                splits[-1][1] = min(splits[-1][1], len(seq_indices))

                if mode == "train":
                    # random.shuffle(splits)
                    random.Random(random_seed + i).shuffle(splits)
                splits = splits[:n_samples_per_subset]
            else:
                splits = [[0, len(seq_indices)]]

            seq_num = 0
            for split in splits:
                seq_indices_ = seq_indices[split[0]:split[1]]
                if len(seq_indices_) < self.nframe:
                    continue
                if self.nframe > 0:
                    if self.random_frame_sample:
                        if mode == "train":
                            seq_idx_shuffled = random.sample(sorted(seq_indices_), len(seq_indices_))
                        else:
                            seed = _seq_name_to_seed(f"{class_name}") + random_seed
                            seq_idx_shuffled = random.Random(seed).sample(sorted(seq_indices_), len(seq_indices_))
                        new_idx = seq_idx_shuffled[:self.nframe]
                    else:  # 等间隔采样
                        new_idx = seq_indices_[::len(seq_indices_) // self.nframe][:self.nframe]
                else:
                    new_idx = seq_indices_

                if self.sort_frames:
                    new_idx.sort(key=lambda x: x.split("/")[-1])
                file_list.extend(new_idx)
                sequence_index.extend([seq_num] * len(new_idx))
                seq_num += 1

        self.data_list_per_epoch = file_list
        self.seq_list_per_epoch = sequence_index

    def load_camera(self, fname, origin_h, origin_w):
        camera_path = os.path.join("/".join(fname.split("/")[:-1]), "opencv_camera.json")
        with open(camera_path, "r") as f:
            camera_file = json.load(f)
            intrinsic = np.array(camera_file[fname.split("/")[-1]]["intrinsic"])
            extrinsic = np.array(camera_file[fname.split("/")[-1]]["extrinsic"])
            if camera_file[fname.split("/")[-1]]['w'] == 1 and camera_file[fname.split("/")[-1]]['h'] == 1:
                intrinsic[0, :] *= origin_w
                intrinsic[1, :] *= origin_h

        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
        extrinsic = torch.tensor(extrinsic, dtype=torch.float32)

        # 归一化外参平移最长边
        if self.camera_longest_side is not None:
            camera_scale_path = os.path.join("/".join(fname.split("/")[:-1]), "camera_scale.json")
            with open(camera_scale_path, "r") as f:
                scale = json.load(f)
                cam_size = np.array(scale["max_scale"]) - np.array(scale["min_scale"])
                max_size = np.max(cam_size)
                # max_size might be 0!
                rescale = self.camera_longest_side / max_size if max_size > self.camera_longest_side else 1.0
            extrinsic[:3, 3:4] *= rescale
        else:
            rescale = 1.0

        return intrinsic, extrinsic, rescale

    def center_crop_intrinsic(self, intrinsic, h0, w0, h1, w1):
        assert h0 >= h1
        assert w0 >= w1
        crop_size = (w0 - w1) / 2
        intrinsic[0, 2] -= crop_size
        crop_size = (h0 - h1) / 2
        intrinsic[1, 2] -= crop_size
        return intrinsic

    def __getitem__(self, index):
        image = Image.open(f"{self.dataset_root}/{self.data_list_per_epoch[index]}").convert("RGB")
        origin_w, origin_h = image.size
        intrinsic, extrinsic, rescale = self.load_camera(f"{self.dataset_root}/{self.data_list_per_epoch[index]}", origin_h, origin_w)
        seq_index = self.seq_list_per_epoch[index]
        seq_name = self.data_list_per_epoch[index]

        # load monocular depth
        if self.enable_depth:
            depth_path = f"{self.depth_root}/{self.data_list_per_epoch[index]}"
            depth_path = re.sub(r'\.[^.]+$', '.png', depth_path)
            if not os.path.exists(depth_path):
                print(f"Not found depth in {depth_path}.")
                depth = np.zeros((origin_h, origin_w), dtype=np.float32)  # let it be the cfg training
            else:
                disp = load_16big_png_depth(depth_path)
                disp = cv2.resize(disp, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
                if self.align_depth:
                    scale_path = depth_path.replace(self.depth_root, self.align_scale_root).replace(".png", ".json")
                    if os.path.exists(scale_path):
                        align_scale = json.load(open(scale_path))
                    else:
                        align_scale = {'scale': 0.0, 'shift': 0.0}
                        print("ERROR! No scale file of", scale_path)
                    if align_scale["scale"] == 0:
                        depth = np.zeros_like(disp)  # no valid scale, used for cfg
                    else:
                        disp = np.clip(disp * align_scale["scale"] + align_scale["shift"], 1e-4, 1e4)
                        depth = np.clip(1 / disp, 0, 1e4) * rescale
                else:
                    # default: we normalize the disparity to 0~1 as the "depth"
                    depth = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
        else:
            depth = None

        if self.scale_map is not None:  # multi-scale training
            height, width = self.scale_map[index]
        else:
            height, width = self.height, self.width

        # specific_wh = (width, height) if self.scale_map is not None else None
        specific_wh = (width, height) if width != height else None
        image = _resize_pil_image(image, edge_size=min(height, width), longest=False, specific_wh=specific_wh, force_resize=True)
        if depth is not None:
            depth = _resize_np_image(depth, edge_size=min(height, width), longest=False, interpolation="nearest", specific_wh=specific_wh, force_resize=True)
        new_w, new_h = image.size

        # rescale intrinsic
        intrinsic[0, :] *= (new_w / origin_w)
        intrinsic[1, :] *= (new_h / origin_h)

        image = transforms.Compose([transforms.ToTensor(),
                                    transforms.CenterCrop(size=(height, width)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(image)
        # image = self.transform(image)  # value:[-1~1] shape:[3,h,w]
        if depth is not None:
            depth = transforms.Compose([transforms.ToTensor(),
                                        transforms.CenterCrop(size=(height, width))])(depth)

        # we always use center crop for real10k to make intrinsic process easy
        intrinsic = self.center_crop_intrinsic(intrinsic, h0=new_h, w0=new_w, h1=height, w1=width)

        # load caption
        caption_path = f"{self.caption}/{'/'.join(seq_name.split('/')[:2])}.txt"
        if os.path.exists(caption_path):
            with open(caption_path) as f:
                prompt = ""
                for line in f.readlines():
                    prompt += line.strip()
            prompt = process_prompt(prompt)
        else:
            prompt = ""
            # print(f"No caption for {self.global_tag}", caption_path)

        meta = {
            "image": image,
            "sequence_index": seq_index,
            "sequence_name": seq_name,
            "tag": "real10k",
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "prompt": prompt
        }

        if depth is not None:
            meta['depth'] = depth

        return EasyDict(meta)
