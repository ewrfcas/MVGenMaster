import collections

import einops
import numpy as np
import torch
# from softmax_splatting import softsplat
from PIL import Image

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference


# from utils.util import visualize_flow


def tensor_to_pil(img, to_np=False):
    if img.min() < 0:  # -1~1 to 0~1
        img = (img + 1) / 2
    img = img[0].permute(1, 2, 0) * 255
    if to_np:
        return img.cpu().numpy().astype(np.uint8)
    else:
        return Image.fromarray(img.cpu().numpy().astype(np.uint8))


def get_paired_info(dust3r, images, nframe, h, w):
    # return: [nframe-1, h, w]
    dust3r_images = [dict(img=images[j:j + 1], true_shape=np.int32([[h, w]]), idx=j, instance=str(j)) for j in range(nframe)]
    pairs = make_pairs(dust3r_images, scene_graph='oneref', prefilter=None, symmetrize=True)
    output = inference(pairs, dust3r, "cuda", batch_size=len(pairs), verbose=False)

    split_idx = [[v1, len(pairs) // 2 + v1] for v1 in range(len(pairs) // 2)]
    depth_maps = []
    masks = []
    Ks = []
    c2ws = []

    for idx0, idx1 in split_idx:
        output_temp = collections.defaultdict(dict)
        for k in output.keys():
            if k == "loss":
                continue
            for k1 in output[k]:
                if type(output[k][k1]) == list:
                    mv0 = output[k][k1][idx0] if int(output[k][k1][idx0]) < 1 else 1
                    mv1 = output[k][k1][idx1] if int(output[k][k1][idx1]) < 1 else 1
                    if type(output[k][k1][idx0]) == str:
                        mv0, mv1 = str(mv0), str(mv1)
                    output_temp[k][k1] = [mv0, mv1]
                elif type(output[k][k1]) == torch.Tensor:
                    output_temp[k][k1] = torch.stack([output[k][k1][idx0], output[k][k1][idx1]], dim=0)
                else:
                    raise NotImplementedError
        scene = global_aligner(output_temp, device="cuda", mode=GlobalAlignerMode.PairViewer, verbose=False)
        depth_maps.append(scene.get_depthmaps()[0])
        Ks.append(scene.get_intrinsics())
        c2ws.append(scene.get_im_poses())
        masks.append(scene.get_masks()[0])

    depth_maps = torch.stack(depth_maps, dim=0)
    masks = torch.stack(masks, dim=0)
    Ks = torch.stack(Ks, dim=0)
    c2ws = torch.stack(c2ws, dim=0)

    return {
        "depths": depth_maps,
        "masks": masks,
        "Ks": Ks,
        "c2ws": c2ws
    }


def batch_fast_paired_info(dust3r, images, nframe, h, w, device):
    # images: [bf,3,h,w]
    # return: [nframe-1, h, w]
    if len(images.shape) == 4:
        images = einops.rearrange(images, "(b f) c h w -> b f c h w", f=nframe)

    pcds = []
    for bi in range(images.shape[0]):
        dust3r_images = [dict(img=images[bi, j:j + 1], true_shape=np.int32([[h, w]]), idx=j, instance=str(j)) for j in range(nframe)]
        pairs = make_pairs(dust3r_images, scene_graph='complete', prefilter=None, symmetrize=True)

        output = inference(pairs, dust3r, "cuda", batch_size=1, verbose=False)
        scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PairViewer, verbose=False)

        color = scene.imgs
        pcd = scene.get_pts3d()
        mask = scene.get_masks()

        new_pcd = []

        for i in range(nframe):
            color_ = color[i] * 2 - 1
            pcd_ = pcd[i].detach().cpu().numpy()
            mask_ = mask[i].detach().cpu().numpy()
            pcd_ = np.concatenate([pcd_[mask_], color_[mask_]], axis=1)
            if pcd_.shape[0] == 0:  # no valid points
                pcd_ = np.zeros((1000, 6), dtype=np.float32)
            new_pcd.append(pcd_)

        new_pcd = np.concatenate(new_pcd, axis=0)
        pcds.append(new_pcd)

    offset = [0]
    feat = []
    coord = []
    for pcd in pcds:
        feat.append(torch.tensor(pcd, dtype=torch.float32, device=device))
        coord.append(torch.tensor(pcd[:, :3], dtype=torch.float32, device=device))
        offset.append(offset[-1] + pcd.shape[0])
    feat = torch.cat(feat, dim=0)
    coord = torch.cat(coord, dim=0)
    offset = offset[1:]

    pcd_dict = dict(
        feat=feat,
        coord=coord,
        grid_size=0.02,
        offset=torch.tensor(offset, dtype=torch.long, device=device)
    )

    return pcd_dict


def batch_fast_paired_info2(dust3r, images, nframe, h, w):
    # images: [bf,3,h,w]
    # return: [nframe-1, h, w]
    if len(images.shape) == 4:
        images = einops.rearrange(images, "(b f) c h w -> b f c h w", f=nframe)

    pcds = []
    for bi in range(images.shape[0]):
        dust3r_images = [dict(img=images[bi, j:j + 1], true_shape=np.int32([[h, w]]), idx=j, instance=str(j)) for j in range(nframe)]
        pairs = make_pairs(dust3r_images, scene_graph='complete', prefilter=None, symmetrize=True)

        output = inference(pairs, dust3r, "cuda", batch_size=1, verbose=False)
        scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PairViewer, verbose=False)

        pcd = scene.get_pts3d()
        mask = scene.get_masks()

        new_pcd = []

        for i in range(nframe):
            pcd_ = pcd[i].detach()
            mask_ = mask[i].detach().float().unsqueeze(-1)
            pcd_ = torch.cat([pcd_, mask_], dim=-1)
            new_pcd.append(pcd_.permute(2,0,1))

        new_pcd = torch.stack(new_pcd, dim=0)
        pcds.append(new_pcd)
    pcds = torch.stack(pcds, dim=0)

    return pcds