import collections
import hashlib
import importlib
import struct
from typing import Optional, Union

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image


def _resize_pil_image(img, edge_size, longest=True, specific_wh=None, force_resize=False):
    if longest:
        S = max(img.size)
    else:
        S = min(img.size)
    if S > edge_size:
        interp = Image.LANCZOS
    else:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * edge_size / S)) // 16 * 16 for x in img.size)

    # 极小情况resize尺度不够，或者高宽比不统一，强制resize
    if specific_wh is not None and max(new_size) < max(specific_wh):
        return img.resize(specific_wh, interp)

    if force_resize and specific_wh is not None and ((specific_wh[0] / specific_wh[1]) > 1.0) != ((img.size[0] / img.size[1]) > 1.0):
        return img.resize(specific_wh, interp)

    return img.resize(new_size, interp)


def _resize_np_image(img, edge_size, longest=True, interpolation="nearest", specific_wh=None, force_resize=False):
    if longest:
        S = max(img.shape[:2])
    else:
        S = min(img.shape[:2])

    if interpolation == "nearest":
        interp = cv2.INTER_NEAREST
    elif interpolation == "area":
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    new_size = tuple(int(round(x * edge_size / S)) // 16 * 16 for x in img.shape[:2][::-1])

    if specific_wh is not None and max(new_size) < max(specific_wh):
        return cv2.resize(img, specific_wh, interpolation=interp)

    # important! 高宽比numpy array应该注意是w和h对比而不是h和w
    if force_resize and specific_wh is not None and ((specific_wh[0] / specific_wh[1]) > 1.0) != ((img.shape[1] / img.shape[0]) > 1.0):
        return cv2.resize(img, specific_wh, interpolation=interp)

    return cv2.resize(img, new_size, interpolation=interp)


def _seq_name_to_seed(seq_name) -> int:
    return int(hashlib.sha1(seq_name.encode("utf-8")).hexdigest(), 16)


def make_color_wheel(bins: Optional[Union[list, tuple]] = None) -> np.ndarray:
    """Build a color wheel.

    Args:
        bins(list or tuple, optional): Specify the number of bins for each
            color range, corresponding to six ranges: red -> yellow,
            yellow -> green, green -> cyan, cyan -> blue, blue -> magenta,
            magenta -> red. [15, 6, 4, 11, 13, 6] is used for default
            (see Middlebury).

    Returns:
        ndarray: Color wheel of shape (total_bins, 3).
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)

    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]

    num_bins = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((3, num_bins), dtype=np.float32)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T


def flow2rgb(flow: np.ndarray,
             color_wheel: Optional[np.ndarray] = None,
             unknown_thr: float = 1e6) -> np.ndarray:
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (float): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
            np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
            (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    rad = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(-dy, -dx) / np.pi

    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (1 -
                w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img


def visualize_flow(flow: np.ndarray) -> np.ndarray:
    """Flow visualization function.

    Args:
        flow (ndarray): The flow will be render
        save_dir ([type], optional): save dir. Defaults to None.
    Returns:
        ndarray: flow map image with RGB order.
    """

    # return value from mmcv.flow2rgb is [0, 1.] with type np.float32
    flow_map = np.uint8(flow2rgb(flow) * 255.)
    return flow_map


def get_lpips_score(loss_fn_alex, gt, pred, device):  # gt, pred: [h,w,3]
    gt_img_lpips = lpips.im2tensor(gt * 255).to(device)
    pred_img_lpips = lpips.im2tensor(pred * 255).to(device)
    lpips_ = loss_fn_alex(gt_img_lpips, pred_img_lpips).item()

    return lpips_


def get_clip_score(clip_model, gt, pred, device):  # gt, pred: [h,w,3]
    clip_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                              (0.26862954, 0.26130258, 0.27577711))])

    pred_img_clip = clip_transform(pred)[None].to(device)
    pred_img_clip = F.interpolate(pred_img_clip.to(dtype=torch.float32), [224, 224], mode='bicubic')
    pred_clip_feat = clip_model.visual(pred_img_clip.to(dtype=clip_model.ln_final.weight.dtype))
    pred_clip_feat = pred_clip_feat / pred_clip_feat.norm(dim=-1, keepdim=True)

    ref_img_clip = clip_transform(gt)[None].to(device).to(device, dtype=clip_model.ln_final.weight.dtype)
    ref_img_clip = F.interpolate(ref_img_clip.to(dtype=torch.float32), [224, 224], mode='bicubic')
    ref_clip_feat = clip_model.visual(ref_img_clip.to(dtype=clip_model.ln_final.weight.dtype))
    ref_clip_feat = ref_clip_feat / ref_clip_feat.norm(dim=-1, keepdim=True)

    clip_ = (pred_clip_feat * ref_clip_feat).sum().item()

    return clip_


# ============================ read_model.py ============================#
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

param_type = {
    'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
    'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
    'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
    'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
    'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
    'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
    'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
    'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
    'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
    'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
    'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
}


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8", errors="ignore")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image_(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image_(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


class Image_(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def process_prompt(prompt):
    prompt = prompt.replace("The video features", "").strip()
    prompt = prompt.replace("The video showcases", "").strip()

    return prompt


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


_EPSILON = np.spacing(1)
from sklearn.linear_model._ransac import *
from sklearn.utils.validation import _check_sample_weight
def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.

    n_samples : int
        Total number of samples in the data.

    min_samples : int
        Minimum number of samples chosen randomly from original data.

    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.

    """
    inlier_ratio = n_inliers / float(n_samples)
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    if nom == 1:
        return 0
    if denom == 1:
        return float("inf")
    return abs(float(np.ceil(np.log(nom) / np.log(denom))))


# Wrapping the model in a custom class
class PositiveLinearRegression(LinearRegression):
    def fit(self, X, y, coef_clip=1e-6):
        # 使用线性回归进行拟合
        super().fit(X, y)
        # 将负系数设为0
        self.coef_ = np.maximum(coef_clip, self.coef_)


class RANSACRegressorW(RANSACRegressor):
    def fit(self, X, y, sample_weight=None, coef_clip=1e-6):
        """Fit estimator using RANSAC algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample
            raises error if sample_weight is passed and estimator
            fit method does not support it.

            .. versionadded:: 0.18

        Returns
        -------
        self : object
            Fitted `RANSACRegressor` estimator.

        Raises
        ------
        ValueError
            If no valid consensus set could be found. This occurs if
            `is_data_valid` and `is_model_valid` return False for all
            `max_trials` randomly chosen sub-samples.
        """
        # Need to validate separately here. We can't pass multi_output=True
        # because that would allow y to be csr. Delay expensive finiteness
        # check to the estimator's own input validation.
        check_X_params = dict(accept_sparse="csr", force_all_finite=False)
        check_y_params = dict(ensure_2d=False)
        X, y = self._validate_data(
            X, y, validate_separately=(check_X_params, check_y_params)
        )
        check_consistent_length(X, y)

        if self.estimator is not None:
            estimator = clone(self.estimator)
        else:
            estimator = LinearRegression()

        if self.min_samples is None:
            if not isinstance(estimator, LinearRegression):
                raise ValueError(
                    "`min_samples` needs to be explicitly set when estimator "
                    "is not a LinearRegression."
                )
            min_samples = X.shape[1] + 1
        elif 0 < self.min_samples < 1:
            min_samples = np.ceil(self.min_samples * X.shape[0])
        elif self.min_samples >= 1:
            min_samples = self.min_samples
        if min_samples > X.shape[0]:
            raise ValueError(
                "`min_samples` may not be larger than number "
                "of samples: n_samples = %d." % (X.shape[0])
            )

        if self.residual_threshold is None:
            # MAD (median absolute deviation)
            residual_threshold = np.median(np.abs(y - np.median(y)))
        else:
            residual_threshold = self.residual_threshold

        if self.loss == "absolute_error":
            if y.ndim == 1:
                loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)
            else:
                loss_function = lambda y_true, y_pred: np.sum(
                    np.abs(y_true - y_pred), axis=1
                )
        elif self.loss == "squared_error":
            if y.ndim == 1:
                loss_function = lambda y_true, y_pred: (y_true - y_pred) ** 2
            else:
                loss_function = lambda y_true, y_pred: np.sum(
                    (y_true - y_pred) ** 2, axis=1
                )

        elif callable(self.loss):
            loss_function = self.loss

        random_state = check_random_state(self.random_state)

        try:  # Not all estimator accept a random_state
            estimator.set_params(random_state=random_state)
        except ValueError:
            pass

        estimator_fit_has_sample_weight = has_fit_parameter(estimator, "sample_weight")
        estimator_name = type(estimator).__name__
        if sample_weight is not None and not estimator_fit_has_sample_weight:
            raise ValueError(
                "%s does not support sample_weight. Samples"
                " weights are only used for the calibration"
                " itself." % estimator_name
            )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        n_inliers_best = 1
        score_best = -np.inf
        inlier_mask_best = None
        X_inlier_best = None
        y_inlier_best = None
        inlier_best_idxs_subset = None
        self.n_skips_no_inliers_ = 0
        self.n_skips_invalid_data_ = 0
        self.n_skips_invalid_model_ = 0

        # number of data samples
        n_samples = X.shape[0]
        sample_idxs = np.arange(n_samples)

        self.n_trials_ = 0
        max_trials = self.max_trials
        while self.n_trials_ < max_trials:
            self.n_trials_ += 1

            if (
                    self.n_skips_no_inliers_
                    + self.n_skips_invalid_data_
                    + self.n_skips_invalid_model_
            ) > self.max_skips:
                break

            # choose random sample set
            subset_idxs = sample_without_replacement(
                n_samples, min_samples, random_state=random_state
            )
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # check if random sample set is valid
            if self.is_data_valid is not None and not self.is_data_valid(
                    X_subset, y_subset
            ):
                self.n_skips_invalid_data_ += 1
                continue

            # fit model for current random sample set
            if sample_weight is None:
                estimator.fit(X_subset, y_subset, coef_clip=coef_clip)
            else:
                estimator.fit(
                    X_subset, y_subset, sample_weight=sample_weight[subset_idxs], coef_clip=coef_clip
                )

            # check if estimated model is valid
            if self.is_model_valid is not None and not self.is_model_valid(
                    estimator, X_subset, y_subset
            ):
                self.n_skips_invalid_model_ += 1
                continue

            # residuals of all data for current random sample model
            y_pred = estimator.predict(X)
            residuals_subset = loss_function(y, y_pred)

            # classify data into inliers and outliers
            inlier_mask_subset = residuals_subset <= residual_threshold
            n_inliers_subset = np.sum(inlier_mask_subset)

            # less inliers -> skip current random sample
            if n_inliers_subset < n_inliers_best:
                self.n_skips_no_inliers_ += 1
                continue

            # extract inlier data set
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs_subset]
            y_inlier_subset = y[inlier_idxs_subset]

            # score of inlier data set
            score_subset = estimator.score(X_inlier_subset, y_inlier_subset)

            # same number of inliers but worse score -> skip current random
            # sample
            if n_inliers_subset == n_inliers_best and score_subset < score_best:
                continue

            # save current random sample as best sample
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            X_inlier_best = X_inlier_subset
            y_inlier_best = y_inlier_subset
            inlier_best_idxs_subset = inlier_idxs_subset

            max_trials = min(
                max_trials,
                _dynamic_max_trials(
                    n_inliers_best, n_samples, min_samples, self.stop_probability
                ),
            )

            # break if sufficient number of inliers or score is reached
            if n_inliers_best >= self.stop_n_inliers or score_best >= self.stop_score:
                break

        # if none of the iterations met the required criteria
        if inlier_mask_best is None:
            if (
                    self.n_skips_no_inliers_
                    + self.n_skips_invalid_data_
                    + self.n_skips_invalid_model_
            ) > self.max_skips:
                raise ValueError(
                    "RANSAC skipped more iterations than `max_skips` without"
                    " finding a valid consensus set. Iterations were skipped"
                    " because each randomly chosen sub-sample failed the"
                    " passing criteria. See estimator attributes for"
                    " diagnostics (n_skips*)."
                )
            else:
                raise ValueError(
                    "RANSAC could not find a valid consensus set. All"
                    " `max_trials` iterations were skipped because each"
                    " randomly chosen sub-sample failed the passing criteria."
                    " See estimator attributes for diagnostics (n_skips*)."
                )
        else:
            if (
                    self.n_skips_no_inliers_
                    + self.n_skips_invalid_data_
                    + self.n_skips_invalid_model_
            ) > self.max_skips:
                warnings.warn(
                    (
                        "RANSAC found a valid consensus set but exited"
                        " early due to skipping more iterations than"
                        " `max_skips`. See estimator attributes for"
                        " diagnostics (n_skips*)."
                    ),
                    ConvergenceWarning,
                )

        # estimate final model using all inliers
        if sample_weight is None:
            estimator.fit(X_inlier_best, y_inlier_best, coef_clip=coef_clip)
        else:
            estimator.fit(
                X_inlier_best,
                y_inlier_best,
                sample_weight=sample_weight[inlier_best_idxs_subset],
                coef_clip=coef_clip
            )

        self.estimator_ = estimator
        self.inlier_mask_ = inlier_mask_best
        return self


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

def tensor_to_mp4(video, savepath, fps, rescale=True, nrow=None):
    """
    video: torch.Tensor, b,c,t,h,w, 0-1
    if -1~1, enable rescale=True
    """
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
    nrow = int(np.sqrt(n)) if nrow is None else nrow
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=nrow, padding=0) for framesheet in video] # [3, grid_h, grid_w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [T, 3, grid_h, grid_w]
    grid = torch.clamp(grid.float(), -1., 1.)
    if rescale:
        grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, 3, grid_h, grid_w] -> [T, grid_h, grid_w, 3]
    torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})