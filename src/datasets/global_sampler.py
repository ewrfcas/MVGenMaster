import collections
import json
import math
import random

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from src.datasets.multi_scale import assign_multi_scale_info, get_multiview_multiscale_indices


# co3d is extremely unbalanced data, so we need to sample subset for each epoch during the training
# Moreover, we need to ensure the shuffle should be performed among different scenes (N frames should not be influenced).
class GlobalConcatSampler(Sampler):
    def __init__(self,
                 data_source,
                 dynamic_sampling: bool = False,
                 n_frames_per_sample: int = 16,
                 batch_per_gpu: int = None,
                 multi_scale: bool = False,
                 shuffle: bool = True,
                 seed: int = 123,  # 这个seed不同进程必须一样，否则DDP会有重复样本
                 rank=0,
                 num_replicas=1,
                 data_config=None,
                 mode="train"):

        self.shuffle = shuffle
        self.generator = torch.manual_seed(seed)
        self.seed = seed
        self.epoch = 0
        self.mode = mode

        self.data_config = data_config
        self.dynamic_sampling = dynamic_sampling

        self.data_source_origin = data_source
        self.index_map = []
        self.sample_num_per_category = []
        self.cumulative_sizes = []

        self.n_subset = len(self.data_source_origin.datasets)
        # self.n_samples_per_subset = n_samples_per_subset * n_frames_per_sample
        self.n_frames_per_sample = n_frames_per_sample
        self.rank = rank
        self.num_replicas = num_replicas

        self.multi_scale = multi_scale
        self.batch_per_gpu = batch_per_gpu  # 每张卡的batchsize，应该为total_frames / nframe
        if self.multi_scale:
            ratio_set = json.load(open("./configs/datasets/ratio_set.json", "r"))
            self.ratio_dict = dict()
            for h, w in ratio_set:
                self.ratio_dict[h / w] = [h, w]
            self.total_bucket = None
            self.scene_ratio_dict = dict()
            for i in range(len(self.data_source_origin.datasets)):  # load scene_ratio_dict
                tag = self.data_source_origin.datasets[i].global_tag
                if tag not in ("objaverse", "front3d"):
                    if tag == "eth3d":
                        self.scene_ratio_dict[tag] = json.load(open(f"./data/{tag.upper()}/scene_ratio_dict.json", "r"))
                    else:
                        self.scene_ratio_dict[tag] = json.load(open(f"./data/{tag}/scene_ratio_dict.json", "r"))

        # 为了获取样本数，我们在一开始需要先做一个sampling
        if self.dynamic_sampling:
            self.dynamic_sampling_per_epoch()
        else:
            self.index_map = list(np.arange(len(self.data_source_origin)))
            for d in self.data_source_origin.datasets:
                self.sample_num_per_category.append(len(d))

            if self.multi_scale:
                global_cumsum_size = 0
                total_bucket = collections.defaultdict(list)
                for i in tqdm(range(len(self.data_source_origin.datasets)), desc="reset dataset", disable=self.rank != 0):
                    tag = self.data_source_origin.datasets[i].global_tag
                    if self.multi_scale:
                        total_bucket = assign_multi_scale_info(ratio_dict=self.ratio_dict,
                                                               nframe=self.n_frames_per_sample,
                                                               scene_ratio_dict=self.scene_ratio_dict[tag] if tag in self.scene_ratio_dict else None,
                                                               total_bucket=total_bucket,
                                                               dataname=tag,
                                                               global_cumsum_size=global_cumsum_size,
                                                               file_list=self.data_source_origin.datasets[i].data_list_per_epoch,
                                                               data_source_origin=self.data_source_origin,
                                                               dataset_idx=i)
                    global_cumsum_size += len(self.data_source_origin.datasets[i])
                self.total_bucket = total_bucket

            self.cumulative_sizes = np.cumsum(self.sample_num_per_category)
            self.get_sample_group_number(n_frames_per_sample)

    def __len__(self):
        return self.n_samples

    def dynamic_sampling_per_epoch(self):
        self.index_map = []
        self.sample_num_per_category = []
        global_cumsum_size = 0
        datanum_dict_before = collections.defaultdict(int)
        datanum_dict_after = collections.defaultdict(int)
        total_bucket = collections.defaultdict(list)
        for i in tqdm(range(len(self.data_source_origin.datasets)), desc="reset dataset", disable=self.rank != 0):
            tag = self.data_source_origin.datasets[i].global_tag
            self.data_source_origin.datasets[i].reset_dataset(mode="train", rank=self.rank,
                                                              random_seed=self.seed + self.epoch)
            if self.multi_scale:
                total_bucket = assign_multi_scale_info(ratio_dict=self.ratio_dict,
                                                       nframe=self.n_frames_per_sample,
                                                       scene_ratio_dict=self.scene_ratio_dict[tag] if tag in self.scene_ratio_dict else None,
                                                       total_bucket=total_bucket,
                                                       dataname=tag,
                                                       global_cumsum_size=global_cumsum_size,
                                                       file_list=self.data_source_origin.datasets[i].data_list_per_epoch,
                                                       data_source_origin=self.data_source_origin,
                                                       dataset_idx=i)
            keep_idx = list(np.arange(len(self.data_source_origin.datasets[i])))
            keep_idx = [k + global_cumsum_size for k in keep_idx]
            global_cumsum_size += len(self.data_source_origin.datasets[i])
            self.index_map.extend(keep_idx)
            self.sample_num_per_category.append(len(keep_idx))

            datanum_dict_before[tag] += self.data_source_origin.datasets[i].total_num
            datanum_dict_after[tag] += len(keep_idx)

        self.cumulative_sizes = np.cumsum(self.sample_num_per_category)

        new_cumulative_sizes = self.data_source_origin.cumsum(self.data_source_origin.datasets)

        if self.rank == 0:
            for tag in datanum_dict_before:
                print(f"{tag}: {datanum_dict_before[tag]} -> {datanum_dict_after[tag]} images")
            print(f"cumulative_sizes: {self.data_source_origin.cumulative_sizes} -> {new_cumulative_sizes}")

        if self.multi_scale:
            self.total_bucket = total_bucket

        self.data_source_origin.cumulative_sizes = new_cumulative_sizes

        self.get_sample_group_number(self.n_frames_per_sample)

    def get_sample_group_number(self, n_frames_per_sample):
        self.n_samples = 0
        for n_sample in self.sample_num_per_category:
            self.n_samples += n_sample
        assert self.n_samples % n_frames_per_sample == 0
        self.n_group = self.n_samples // n_frames_per_sample

        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"
                             " [0, {}]".format(self.rank, self.num_replicas - 1))

        if self.n_group % self.num_replicas != 0:
            self.n_group = math.ceil((self.n_group - self.num_replicas) / self.num_replicas)
        else:
            self.n_group = math.ceil(self.n_group / self.num_replicas)
        self.n_group_total = self.n_group * self.num_replicas
        self.n_samples = self.n_group * self.n_frames_per_sample
        self.n_samples_total = self.n_samples * self.num_replicas

        assert self.n_samples * self.num_replicas <= sum(self.sample_num_per_category)
        assert self.index_map[-1] < len(self.data_source_origin)

    def __iter__(self):
        indices = []
        random.seed(self.seed + self.epoch)

        if self.multi_scale:
            indices = get_multiview_multiscale_indices(total_bucket=self.total_bucket,
                                                       nframe=self.n_frames_per_sample,
                                                       batch_per_gpu=self.batch_per_gpu,
                                                       generator=self.generator,
                                                       shuffle=self.shuffle,
                                                       num_replicas=self.num_replicas,
                                                       rank=self.rank)
        else:
            # sample from each sub-dataset
            for d_idx in range(self.n_subset):
                low = 0 if d_idx == 0 else self.cumulative_sizes[d_idx - 1]
                len_subset = self.sample_num_per_category[d_idx]
                len_subgroup = len_subset // self.n_frames_per_sample
                rand_tensor = torch.arange(len_subgroup) * self.n_frames_per_sample + low
                indices.append(rand_tensor)
            indices = torch.cat(indices)  # [N//g]
            group_add = torch.arange(self.n_frames_per_sample).reshape(1, self.n_frames_per_sample).repeat(indices.shape[0], 1)
            indices = indices.unsqueeze(-1) + group_add  # [N//g, g]
            if self.shuffle:  # shuffle the sampled dataset (from multiple subsets)
                rand_tensor = torch.randperm(indices.shape[0], generator=self.generator)
                indices = indices[rand_tensor]

        # subsample for DDP
        if self.multi_scale:
            rank_size = indices.shape[0] // self.num_replicas
            assert rank_size % self.batch_per_gpu == 0
            indices = indices[self.rank * rank_size: (self.rank + 1) * rank_size]
            indices = indices.reshape(-1)  # [N]
        else:
            indices = indices[:self.n_group_total]
            assert indices.shape[0] == self.n_group_total
            indices = indices[self.rank:self.n_group_total:self.num_replicas]
            indices = indices.reshape(-1)  # [N]
            assert indices.shape[0] == self.n_samples


        # map to the origin index
        indices = indices.tolist()
        indices = [self.index_map[i] for i in indices]
        self.indices = indices

        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        # we must reset dataset here, before the __iter__, because the concated dataset's
        # cumulative_sizes must be reset before __iter__
        # print(f'Rank{self.rank} set epoch{epoch} success!')
        self.epoch = epoch
        random.seed(self.epoch + self.seed)
        self.generator = torch.manual_seed(self.epoch + self.seed)
        if self.dynamic_sampling and self.epoch > 0:  # epoch0我们在init的时候已经sampling过了
            self.dynamic_sampling_per_epoch()
