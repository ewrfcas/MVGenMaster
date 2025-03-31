# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dummy optimizer for visualizing pairs
# --------------------------------------------------------
import cv2
import numpy as np
import torch
import torch.nn as nn

from dust3r.cloud_opt.commons import (edge_str, ALL_DISTS, NoGradParamDict, get_imshapes, get_conf_trf)
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import inv, geotrf, depthmap_to_absolute_camera_coordinates


def solve_pnp(args):
    pts3d, msk, pixels, K = args
    try:
        res = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                 iterationsCount=100, reprojectionError=5,
                                 flags=cv2.SOLVEPNP_SQPNP)
        success, R, T, inliers = res
        assert success
        R = cv2.Rodrigues(R)[0]  # world to cam
        pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
    except:
        pose = np.eye(4)
    return pose


from multiprocessing import Pool


def parallel_pnp(batch_size, pts3d, msk, pixels, Ks):
    with Pool(processes=8) as pool:
        args = [(pts3d[i], msk[i], pixels, Ks[i]) for i in range(batch_size)]
        poses = pool.map(solve_pnp, args)

    poses = np.stack(poses, axis=0)
    return torch.from_numpy(poses.astype(np.float32))  # [b,4,4]

class FastViewer(nn.Module):
    """
    This a Dummy Optimizer.
    To use only when the goal is to visualize the results for a pair of images (batchwise)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._init_from_views(*args, **kwargs)

        assert not self.is_symmetrized and self.n_edges == 1
        self.has_im_poses = True

        # compute all parameters directly from raw input
        self.focals = []
        self.pp = []

        H, W = self.imshapes[0]
        pts3d = self.pred_i[edge_str(0, 1)]  # [b,h,w,3]
        self.batch_size = pts3d.shape[0]

        pp = torch.tensor((W / 2, H / 2))[None].repeat(self.batch_size, 1)  # [b,2]
        focals = estimate_focal_knowing_depth(pts3d, pp, focal_mode='weiszfeld')
        self.focals.extend([focals] * 2)  # 因为2个view都固定在view1下，所以共享一套focal
        self.pp.extend([pp] * 2)
        Ks = np.zeros((self.batch_size, 3, 3), dtype=np.float32)
        Ks[:, 0, 0] = focals
        Ks[:, 1, 1] = focals
        Ks[:, 0, 2] = pp[:, 0]
        Ks[:, 1, 2] = pp[:, 1]
        Ks[:, -1, -1] = 1

        # estimate the pose of pts2 in image1
        pixels = np.mgrid[:W, :H].T.astype(np.float32)
        pts3d = self.pred_j[edge_str(0, 1)].numpy()  # [b,h,w,3]
        assert pts3d.shape[1:3] == (H, W)

        msk = self.get_masks()[1].numpy()  # view1下pts2的conf [b,h,w]

        ### HARD Code for CPU PnP-RANSAC ###
        poses = []
        for i in range(self.batch_size):
            try:
                res = cv2.solvePnPRansac(pts3d[i, msk[i]], pixels[msk[i]], Ks[i], None,
                                         iterationsCount=100, reprojectionError=5,
                                         flags=cv2.SOLVEPNP_SQPNP)
                success, R, T, inliers = res
                assert success

                R = cv2.Rodrigues(R)[0]  # world to cam
                pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
            except:
                pose = np.eye(4)
            poses.append(pose)
        poses = np.stack(poses, axis=0)
        rel_poses = torch.from_numpy(poses.astype(np.float32))  # [b,4,4]

        # rel_poses = parallel_pnp(self.batch_size, pts3d, msk, pixels, Ks)

        # ptcloud is expressed in camera1
        self.im_poses = [torch.eye(4)[None].repeat(self.batch_size, 1, 1), rel_poses]  # I, cam2-to-cam1
        self.depth = [self.pred_i['0_1'][..., 2], geotrf(inv(rel_poses), self.pred_j['0_1'])[..., 2]]

        self.im_poses = nn.Parameter(torch.stack(self.im_poses, dim=0), requires_grad=False)
        self.focals = nn.Parameter(torch.stack(self.focals), requires_grad=False)
        self.pp = nn.Parameter(torch.stack(self.pp, dim=0), requires_grad=False)
        self.depth = nn.ParameterList(self.depth)
        Ks = torch.tensor(Ks, device=self.device)
        self.Ks = nn.ParameterList([Ks, Ks])

        for p in self.parameters():
            p.requires_grad = False

    def _init_from_views(self, view1, view2, pred1, pred2,
                         dist='l1',
                         conf='log',
                         min_conf_thr=3,
                         base_scale=0.5,
                         allow_pw_adaptors=False,
                         pw_break=20,
                         rand_pose=torch.randn,
                         iterationsCount=None,
                         verbose=True):
        super().__init__()
        if not isinstance(view1['idx'], list):
            view1['idx'] = view1['idx'].tolist()
        if not isinstance(view2['idx'], list):
            view2['idx'] = view2['idx'].tolist()
        # self.edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
        self.edges = [(0, 1)]  # we fix edges to [0,1] here
        self.is_symmetrized = False
        self.dist = ALL_DISTS[dist]
        self.verbose = verbose

        self.n_imgs = 1  # self._check_edges()

        # input data
        pred1_pts = pred1['pts3d']
        pred2_pts = pred2['pts3d_in_other_view']
        # self.pred_i = NoGradParamDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)})
        # self.pred_j = NoGradParamDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)})
        self.pred_i = NoGradParamDict({"0_1": pred1_pts})
        self.pred_j = NoGradParamDict({"0_1": pred2_pts})
        self.imshapes = get_imshapes(self.edges, pred1_pts, pred2_pts)

        # work in log-scale with conf
        pred1_conf = pred1['conf']
        pred2_conf = pred2['conf']
        self.min_conf_thr = min_conf_thr
        self.conf_trf = get_conf_trf(conf)

        # self.conf_i = NoGradParamDict({ij: pred1_conf[n] for n, ij in enumerate(self.str_edges)})
        # self.conf_j = NoGradParamDict({ij: pred2_conf[n] for n, ij in enumerate(self.str_edges)})
        self.conf_i = NoGradParamDict({"0_1": pred1_conf})
        self.conf_j = NoGradParamDict({"0_1": pred2_conf})
        self.im_conf = self._compute_img_conf(pred1_conf, pred2_conf)

        # pairwise pose parameters
        self.base_scale = base_scale
        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.POSE_DIM = 7
        self.has_im_poses = False
        self.rand_pose = rand_pose

        # possibly store images for show_pointcloud
        self.imgs = None

    def _set_depthmap(self, idx, depth, force=False):
        if self.verbose:
            print('_set_depthmap is ignored in PairViewer')
        return

    def get_depthmaps(self, raw=False):
        depth = [d.to(self.device) for d in self.depth]
        return depth

    def _set_focal(self, idx, focal, force=False):
        self.focals[idx] = focal

    def get_focals(self):
        return self.focals

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.focals])

    def get_principal_points(self):
        return self.pp

    def get_intrinsics(self):
        return self.Ks

    def get_im_poses(self):
        return self.im_poses

    def depth_to_pts3d(self):
        pts3d = []
        for d, intrinsics, im_pose in zip(self.depth, self.get_intrinsics(), self.get_im_poses()):
            pts, _ = depthmap_to_absolute_camera_coordinates(d.cpu().numpy(),
                                                             intrinsics.cpu().numpy(),
                                                             im_pose.cpu().numpy())
            pts3d.append(torch.from_numpy(pts).to(device=self.device))
        return pts3d

    def forward(self):
        return float('nan')

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]

    @property
    def imsizes(self):
        return [(w, h) for h, w in self.imshapes]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def state_dict(self, trainable=True):
        all_params = super().state_dict()
        return {k: v for k, v in all_params.items() if k.startswith(('_', 'pred_i.', 'pred_j.', 'conf_i.', 'conf_j.')) != trainable}

    def load_state_dict(self, data):
        return super().load_state_dict(self.state_dict(trainable=False) | data)

    def _check_edges(self):
        indices = sorted({i for edge in self.edges for i in edge})
        assert indices == list(range(len(indices))), 'bad pair indices: missing values '
        return len(indices)

    @torch.no_grad()
    def _compute_img_conf(self, pred1_conf, pred2_conf):
        im_conf = nn.ParameterList([torch.zeros_like(pred1_conf), torch.zeros_like(pred2_conf)])
        for e, (i, j) in enumerate(self.edges):
            im_conf[i] = torch.maximum(im_conf[i], pred1_conf)
            im_conf[j] = torch.maximum(im_conf[j], pred2_conf)
        return im_conf

    def get_adaptors(self):
        adapt = self.pw_adaptors
        adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
        if self.norm_pw_scale:  # normalize so that the product == 1
            adapt = adapt - adapt.mean(dim=1, keepdim=True)
        return (adapt / self.pw_break).exp()

    def get_pw_norm_scale_factor(self):
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses

    def get_pw_scale(self):
        scale = self.pw_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor()
        return scale

    def get_pw_poses(self):  # cam to world
        RT = self._get_poses(self.pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT

    def get_masks(self):
        return [(conf > self.min_conf_thr) for conf in self.im_conf]

    def get_pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h * w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def get_conf(self, mode=None):
        trf = self.conf_trf if mode is None else get_conf_trf(mode)
        return [trf(c) for c in self.im_conf]
