import collections
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

from utils import _resize_pil_image, _seq_name_to_seed, process_prompt, load_16big_png_depth, _resize_np_image


class MVImageNetDataset(Dataset):
    def __init__(self, dataset_root, data_list, nframe, height, width,
                 rank=0, mode="train", sampling_interval=1.0, **kwargs):
        super(MVImageNetDataset, self).__init__()
        self.dataset_root = dataset_root
        self.height = height
        self.width = width
        self.global_data_list = collections.defaultdict(list)  # 全局index，每个epoch不会改变
        self.nframe = nframe
        self.mode = mode
        self.global_tag = "mvimagenet"
        self.sampling_interval = sampling_interval
        self.mask_path = kwargs.get("mask_path", "./data/mvimagenet/masks")
        self.n_samples_per_subset = kwargs.get("n_samples_per_subset", -1)
        self.random_frame_sample = kwargs.get("random_frame_sample", True)
        self.sort_frames = kwargs.get("sort_frames", False)
        self.camera_longest_side = kwargs.get("camera_longest_side", None)
        # we omit centercrop here because mvi could be cropped by mask!
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.global_seq_dict = pickle.load(open("./data/mvimagenet/global_seq_dict.pkl", "rb"))
        self.total_num = 0
        with open(data_list, "r") as f:
            for line in tqdm(f, desc=f"loading global mvimagenet {mode} sequences...", disable=rank != 0):
                line = line.strip()
                class_name, seq_name = line.split("/")
                self.global_data_list[class_name].append(seq_name)
                self.total_num += len(self.global_seq_dict[f"{class_name}/{seq_name}"])
        self.caption = f"./data/mv_data_captions/{self.global_tag}"
        self.enable_depth = kwargs.get("enable_depth", False)
        self.depth_root = f"./data/multi_view_depthanythingv2/{self.global_tag}"
        self.align_depth = kwargs.get("align_depth", False)
        self.align_scale_root = f"./data/multi_view_depthanythingv2_scaleshift/{self.global_tag}"
        self.scale_map = None

        self.data_list_per_epoch, self.seq_list_per_epoch = [], []  # used for consistent random cropping

        if mode == "val":
            self.reset_dataset(mode=mode, rank=rank)  # for testing, directly loading from global_data_list

    def __len__(self):
        return len(self.data_list_per_epoch)

    def reset_dataset(self, mode="train", rank=0, **kwargs):
        file_list = []
        sequence_index = []
        random_seed = kwargs.get("random_seed", 123)

        class_names = list(self.global_data_list.keys())
        if 0 < self.n_samples_per_subset < 1:
            class_names = class_names[::int(1 / self.n_samples_per_subset)]
            n_samples_per_subset = 1
        else:
            n_samples_per_subset = self.n_samples_per_subset

        for class_name in class_names:
            seq_names = self.global_data_list[class_name]
            seq_num = 0
            if mode == "train":
                random.Random(random_seed).shuffle(seq_names)

            for seq_name in seq_names:

                if n_samples_per_subset > 0 and seq_num >= n_samples_per_subset:
                    break

                seq_indices = self.global_seq_dict[f"{class_name}/{seq_name}"].copy()
                seq_indices = [f"{class_name}/{seq_name}/images/{l}" for l in seq_indices]

                if len(seq_indices) < self.nframe:
                    continue

                if type(self.sampling_interval) == str and self.sampling_interval.startswith("random"):
                    if mode == "val":  # 验证的时候，不随机，固定1.0
                        sampling_interval_ = 1.0
                    else:
                        lower = float(self.sampling_interval.split("_")[1])
                        sampling_interval_ = lower + random.random() * (1.0 - lower)
                else:
                    sampling_interval_ = self.sampling_interval

                if sampling_interval_ < 1.0:
                    split_num = max(int(len(seq_indices) * sampling_interval_), self.nframe)
                    seq_indices = seq_indices[:split_num]
                    if mode == "train":
                        rst = min(random.randint(0, int((1.0 - sampling_interval_) * len(seq_indices))), len(seq_indices) - split_num)
                        seq_indices = seq_indices[rst:rst + split_num]
                    else:
                        seq_indices = seq_indices[:split_num]

                if self.nframe > 0:
                    if self.random_frame_sample:
                        # infer the seed from the sequence name, this is reproducible
                        # and makes the selection differ for different sequences
                        seed = _seq_name_to_seed(f"{class_name}/{seq_name}") + random_seed
                        seq_idx_shuffled = random.Random(seed).sample(sorted(seq_indices), len(seq_indices))
                        new_idx = seq_idx_shuffled[:self.nframe]
                    else:  # 等间隔采样
                        new_idx = seq_indices[::len(seq_indices) // self.nframe][:self.nframe]
                else:
                    new_idx = seq_indices

                if self.sort_frames:
                    new_idx.sort(key=lambda x: x.split("/")[-1])
                file_list.extend(new_idx)
                sequence_index.extend([seq_num] * len(new_idx))

                seq_num += 1

        self.seq_list_per_epoch = sequence_index
        self.data_list_per_epoch = file_list

    def load_camera(self, fname):
        camera_path = os.path.join("/".join(fname.split("/")[:-2]), "opencv_camera.json")
        with open(camera_path, "r") as f:
            camera_file = json.load(f)
            intrinsic = np.array(camera_file["intrinsic"])
            extrinsic = np.array(camera_file["extrinsic"][fname.split("/")[-1]])

        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
        extrinsic = torch.tensor(extrinsic, dtype=torch.float32)

        # 归一化外参平移最长边
        if self.camera_longest_side is not None:
            camera_scale_path = os.path.join("/".join(fname.split("/")[:-2]), "camera_scale.json")
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

    def crop_with_mask(self, image, mask, intrinsic, is_random=False, crop_h=None, crop_w=None):
        image = np.array(image)
        h, w = image.shape[0], image.shape[1]
        y_pos, x_pos = np.where(mask == 1)
        bbox_xyxy = [x_pos.min(), y_pos.min(), x_pos.max(), y_pos.max()]

        if h > w:
            if crop_h is None:
                crop_h = w
            bbox_center = ((bbox_xyxy[1] + bbox_xyxy[3]) // 2)
            if is_random:
                bbox_center += random.randint(-int(w * 0.1), int(w * 0.1))
            y0 = bbox_center - crop_h // 2
            y1 = bbox_center + crop_h // 2
            if y0 < 0:
                y1 -= y0
                y0 = 0
            elif y1 > h:
                y0 -= (y1 - h)
                y1 = h
            clamp_bbox_xyxy = [0, y0, w, y1]
        else:
            if crop_w is None:
                crop_w = w
            bbox_center = ((bbox_xyxy[0] + bbox_xyxy[2]) // 2).item()
            if is_random:
                bbox_center += random.randint(-int(h * 0.1), int(h * 0.1))
            x0 = bbox_center - crop_w // 2
            x1 = bbox_center + crop_w // 2
            if x0 < 0:
                x1 -= x0
                x0 = 0
            elif x1 > w:
                x0 -= (x1 - w)
                x1 = w
            clamp_bbox_xyxy = [x0, 0, x1, h]

        image = image[clamp_bbox_xyxy[1]:clamp_bbox_xyxy[3], clamp_bbox_xyxy[0]:clamp_bbox_xyxy[2]]
        mask = mask[clamp_bbox_xyxy[1]:clamp_bbox_xyxy[3], clamp_bbox_xyxy[0]:clamp_bbox_xyxy[2]]

        intrinsic[0, 2] -= clamp_bbox_xyxy[0]
        intrinsic[1, 2] -= clamp_bbox_xyxy[1]

        image = Image.fromarray(image)

        return image, mask, clamp_bbox_xyxy, intrinsic

    def __getitem__(self, index):
        image = Image.open(f"{self.dataset_root}/{self.data_list_per_epoch[index]}").convert("RGB")
        seq_index = self.seq_list_per_epoch[index]
        seq_name = self.data_list_per_epoch[index]
        class_seq_name = "/".join(self.data_list_per_epoch[index].split("/")[-4:-2])
        intrinsic, extrinsic, rescale = self.load_camera(f"{self.dataset_root}/{self.data_list_per_epoch[index]}")

        # load object mask
        filename = self.data_list_per_epoch[index].split("/")[-1]
        origin_w, origin_h = image.size
        if os.path.exists(f"{self.mask_path}/{class_seq_name}/{filename.replace('.jpg', '.png')}"):
            mask = Image.open(f"{self.mask_path}/{class_seq_name}/{filename.replace('.jpg', '.png')}").convert("L")
        else:
            mask = Image.fromarray(np.zeros((origin_h, origin_w), dtype=np.uint8)).convert("L")

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

        specific_wh = (width, height) if self.scale_map is not None else None
        image = _resize_pil_image(image, edge_size=min(height, width), longest=False, specific_wh=specific_wh)
        mask = _resize_pil_image(mask, edge_size=min(height, width), longest=False, specific_wh=specific_wh)
        if depth is not None:
            depth = _resize_np_image(depth, edge_size=min(height, width), longest=False, interpolation="nearest", specific_wh=specific_wh)
        mask = np.array(mask) / 255
        mask[mask > 0] = 1

        if mask.sum() == 0:
            side_y = mask.shape[0] // 4
            side_x = mask.shape[1] // 4
            mask[side_y:-side_y, side_x:-side_x] = 1

        new_w, new_h = image.size

        # rescale intrinsic
        intrinsic[0, :] *= (new_w / origin_w)
        intrinsic[1, :] *= (new_h / origin_h)

        # crop images with mask
        image, _, clamp_bbox_xyxy, intrinsic = self.crop_with_mask(image, mask, intrinsic, is_random=True if self.mode == "train" else False,
                                                                   crop_h=height, crop_w=width)
        if depth is not None:
            depth = depth[clamp_bbox_xyxy[1]:clamp_bbox_xyxy[3], clamp_bbox_xyxy[0]:clamp_bbox_xyxy[2]]
            depth = transforms.ToTensor()(depth)

        image = self.transform(image)  # value:[-1~1] shape:[3,h,w]

        # load caption
        caption_path = f"{self.caption}/{class_seq_name}.txt"
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
            "tag": "mvimagenet",
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "prompt": prompt,
        }

        if depth is not None:
            meta['depth'] = depth

        return EasyDict(meta)
