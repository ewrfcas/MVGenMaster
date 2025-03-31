import numpy as np
import torch


def assign_multi_scale_info(ratio_dict, nframe, scene_ratio_dict, total_bucket, global_cumsum_size,
                            dataname, file_list, data_source_origin, dataset_idx):
    ratio_list = list(ratio_dict.keys())
    scale_map = dict()
    for i, fname in enumerate(file_list[::nframe]):  # only calculate the hw ratio for the first frame
        if dataname in ("co3dv2", "mvimagenet", "dl3dv", "real10k", "acid"):
            scene_name = "/".join(fname.split("/")[:2])
            ratio = scene_ratio_dict[scene_name]
        elif dataname in ("megascenes"):
            scene_name = "/".join(fname.split("/")[:3])
            ratio = scene_ratio_dict[scene_name]
        elif dataname in ("objaverse"):
            ratio = 1.0
        elif dataname in ("front3d"):
            ratio = 0.75
        else:
            scene_name = fname.split("/")[0]
            ratio = scene_ratio_dict[scene_name]

        sub = [abs(ratio - r) for r in ratio_list]
        [h, w] = ratio_dict[ratio_list[np.argmin(sub)]]
        total_bucket[f"{h}x{w}"].append(global_cumsum_size + i * nframe)

        # 这里的scale_map只存原始dataset里的index，而不是global index
        indices_ = np.arange(i * nframe, (i + 1) * nframe).tolist()
        for sample_idx in indices_:
            scale_map[sample_idx] = [int(h), int(w)]

    data_source_origin.datasets[dataset_idx].scale_map = scale_map

    return total_bucket


def get_multiview_multiscale_indices(total_bucket, nframe, batch_per_gpu, generator, shuffle=False, num_replicas=1,
                                     rank=0):
    indices = []
    # rearrange according resolution
    if rank == 0:
        print("#" * 20, "multi-scale resolution distribution (padding)", "#" * 20)

    for hw in total_bucket:
        new_indices = total_bucket[hw].copy()
        old_length = len(new_indices)
        if len(new_indices) % batch_per_gpu != 0:  # we random pad for different resolutions for batch-wise splitting
            pad_indices = np.random.choice(total_bucket[hw], batch_per_gpu - (len(new_indices) % batch_per_gpu), replace=True)
            new_indices.extend(pad_indices)
        new_length = len(new_indices)

        if shuffle:  # shuffle for seperate resolutions
            rand_tensor = torch.randperm(len(new_indices), generator=generator)
            new_indices = np.array(new_indices)[rand_tensor.numpy()].tolist()

        indices.extend(new_indices)

        if rank == 0:
            print(f"{hw}: {old_length * nframe} ({new_length * nframe})")

    indices = torch.tensor(indices)

    if shuffle:  # 跨分辨率shuffle
        indices = indices.reshape(len(indices) // batch_per_gpu, batch_per_gpu)
        rand_tensor = torch.randperm(indices.shape[0], generator=generator)
        indices = indices[rand_tensor]
        indices = indices.reshape(-1)
    else:  # 测试的时候，固定seed shuffle
        indices = indices.reshape(len(indices) // batch_per_gpu, batch_per_gpu)
        rand_tensor = torch.randperm(indices.shape[0], generator=torch.manual_seed(123))
        indices = indices[rand_tensor]
        indices = indices.reshape(-1)

    if len(indices) % (batch_per_gpu * num_replicas) != 0:
        indices = indices.reshape(len(indices) // batch_per_gpu, batch_per_gpu)
        pad_index = np.random.choice(np.arange(len(indices)), num_replicas - (len(indices) % num_replicas), replace=True)
        pad_indices = indices[pad_index].clone()
        indices = torch.cat([indices, pad_indices], dim=0)
        indices = indices.reshape(-1)

    group_add = torch.arange(nframe).reshape(1, nframe).repeat(indices.shape[0], 1)
    indices = indices.unsqueeze(-1) + group_add  # [N//g, g]
    indices = indices.reshape(-1)

    indices = indices.reshape(len(indices) // nframe, nframe)

    return indices
