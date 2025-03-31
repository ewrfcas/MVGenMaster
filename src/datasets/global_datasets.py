from torch.utils.data import ConcatDataset

from src.datasets.acid_dataset import ACIDDataset
from src.datasets.co3d_dataset import CO3Dv2Dataset
from src.datasets.dl3dv_dataset import DL3DVDataset
from src.datasets.gl3d_dataset import GL3DDataset
from src.datasets.mvimagenet_dataset import MVImageNetDataset
from src.datasets.real10k_dataset import Real10kDataset
from src.datasets.scannet_dataset import ScanNetDataset

dataset_module_map = {
    "dl3dv": DL3DVDataset,
    "co3dv2": CO3Dv2Dataset,
    "mvimagenet": MVImageNetDataset,
    "scannet++": ScanNetDataset,
    "real10k": Real10kDataset,
    "gl3d": GL3DDataset,
    "acid": ACIDDataset,
}


def load_global_dataset(config, cfg, rank=0, eval_unordered=False):
    train_datasets = []
    val_datasets = []
    for dataset_name in cfg:
        # totally camera setting
        cfg[dataset_name].nframe = config.nframe
        cfg[dataset_name].camera_longest_side = config.get("camera_longest_side", None)
        cfg[dataset_name].enable_depth = config.model_cfg.get("enable_depth", False)
        cfg[dataset_name].align_depth = config.model_cfg.get("align_depth", False)
        assert dataset_name in dataset_module_map
        if hasattr(cfg[dataset_name], "train_list"):  # some data is only used for test (mipnerf360, T&T, ETH3D...)
            train_dataset_ = dataset_module_map[dataset_name](data_list=cfg[dataset_name].train_list, mode="train",
                                                              height=cfg[dataset_name].image_height,
                                                              width=cfg[dataset_name].image_width,
                                                              n_samples_per_subset=cfg[dataset_name].train_n_samples_per_subset,
                                                              random_frame_sample=True, rank=rank, **cfg[dataset_name])
            train_datasets.append(train_dataset_)
        val_dataset_ = dataset_module_map[dataset_name](data_list=cfg[dataset_name].val_list, mode="val",
                                                        height=cfg[dataset_name].image_height,
                                                        width=cfg[dataset_name].image_width,
                                                        n_samples_per_subset=cfg[dataset_name].val_n_samples_per_subset,
                                                        random_frame_sample=False if not eval_unordered else True,
                                                        rank=rank, **cfg[dataset_name])
        val_datasets.append(val_dataset_)

    if len(train_datasets) > 0:
        train_dataset = ConcatDataset(train_datasets)
    else:
        train_dataset = None
    val_datasets = ConcatDataset(val_datasets)

    return train_dataset, val_datasets
