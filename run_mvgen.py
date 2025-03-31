import argparse
import copy
import json
import os
import random
import time
from glob import glob

import cv2
import imagesize
import numpy as np
import torch
import trimesh
from PIL import Image
from diffusers import AutoencoderKL
from easydict import EasyDict
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize
from tqdm import tqdm
from depth_pro.depth_pro import create_model_and_transforms
from depth_pro.utils import load_rgb

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from src.modules.cam_vis import add_scene_cam
from src.modules.position_encoding import global_position_encoding_3d
from src.modules.schedulers import get_diffusion_scheduler
from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def points_padding(points):
    padding = torch.ones_like(points)[..., 0:1]
    points = torch.cat([points, padding], dim=-1)
    return points


def np_points_padding(points):
    padding = np.ones_like(points)[..., 0:1]
    points = np.concatenate([points, padding], axis=-1)
    return points


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


def save_16bit_png_depth(depth: np.ndarray, depth_png: str):
    # Ensure the numpy array's dtype is float32, then cast to float16, and finally reinterpret as uint16
    depth_uint16 = np.array(depth, dtype=np.float32).astype(np.float16).view(np.uint16)

    # Create a PIL Image from the 16-bit depth values and save it
    depth_pil = Image.fromarray(depth_uint16)

    if not depth_png.endswith(".png"):
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError

    try:
        depth_pil.save(depth_png)
    except:
        print("ERROR DEPTH FILE:", depth_png)
        raise NotImplementedError


def load_dataset(args, config, reference_cam, target_cam, reference_list, depth_list):
    ratio_set = json.load(open(f"./{args.model_dir}/ratio_set.json", "r"))
    ratio_dict = dict()
    for h, w in ratio_set:
        ratio_dict[h / w] = [h, w]
    ratio_list = list(ratio_dict.keys())

    # load dataset
    print("Loading dataset...")
    intrinsic = np.array(reference_cam["intrinsic"])
    tar_names = list(target_cam["extrinsic"].keys())
    tar_names.sort()
    if args.target_limit is not None:
        tar_names = tar_names[:args.target_limit]
    tar_extrinsic = [np.array(target_cam["extrinsic"][k]) for k in tar_names]

    if args.cond_num == 1:
        reference_list = [reference_list[0]]
    elif args.cond_num == 2:
        reference_list = [reference_list[0], reference_list[-1]]
    elif args.cond_num == 3:
        reference_list = reference_list[:3]
    else:
        pass

    ref_images = []
    ref_names = []
    ref_extrinsic = []
    ref_intrinsic = []
    ref_depth = []
    h, w = None, None
    for i, im in enumerate(tqdm(reference_list, desc="loading reference images")):
        img = Image.open(im).convert("RGB")
        intrinsic_ = copy.deepcopy(intrinsic)
        im = f"view{str(reference_views[i]).zfill(3)}_ref"
        if im.split("/")[-1] in reference_cam["extrinsic"]:
            extrinsic_ = np.array(reference_cam["extrinsic"][im.split("/")[-1]])
        else:
            extrinsic_ = np.array(reference_cam["extrinsic"][im.split("/")[-1].split(".")[0]])
        ref_extrinsic.append(extrinsic_)
        ref_names.append(im.split('/')[-1])

        origin_w, origin_h = img.size

        # load monocular depth
        if config.model_cfg.get("enable_depth", False):
            depth = depth_list[i]
            depth = cv2.resize(depth, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
        else:
            depth = None

        if h is None or w is None:
            ratio = origin_h / origin_w
            sub = [abs(ratio - r) for r in ratio_list]
            [h, w] = ratio_dict[ratio_list[np.argmin(sub)]]
            print(f'height:{h}, width:{w}.')
        img = img.resize((w, h), Image.LANCZOS if h < origin_h else Image.BICUBIC)
        if depth is not None:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        new_w, new_h = img.size
        # rescale intrinsic
        intrinsic_[0, :] *= (new_w / reference_cam['w'])
        intrinsic_[1, :] *= (new_h / reference_cam['h'])

        img = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img)
        if depth is not None:
            depth = Compose([ToTensor()])(depth)

        ref_images.append(img)
        ref_intrinsic.append(intrinsic_)
        if depth is not None:
            ref_depth.append(depth)

    ref_images = torch.stack(ref_images, dim=0)
    tar_intrinsic = [ref_intrinsic[0]] * len(tar_extrinsic)
    ref_intrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in ref_intrinsic], dim=0)
    tar_intrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in tar_intrinsic], dim=0)
    ref_extrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in ref_extrinsic], dim=0)
    tar_extrinsic = torch.stack([torch.tensor(K, dtype=torch.float32) for K in tar_extrinsic], dim=0)

    # 外参t归一化
    if config.camera_longest_side is not None:
        extrinsic = torch.cat([ref_extrinsic, tar_extrinsic], dim=0)  # [N,4,4]
        c2ws = extrinsic.inverse()
        max_scale = torch.max(c2ws[:, :3, -1], dim=0)[0]
        min_scale = torch.min(c2ws[:, :3, -1], dim=0)[0]
        max_size = torch.max(max_scale - min_scale).item()
        rescale = config.camera_longest_side / max_size if max_size > config.camera_longest_side else 1.0
        ref_extrinsic[:, :3, 3:4] *= rescale
        tar_extrinsic[:, :3, 3:4] *= rescale
    else:
        rescale = 1.0

    if len(ref_depth) > 0:
        ref_depth = [r * rescale for r in ref_depth]
        ref_depth = torch.stack(ref_depth, dim=0)
    else:
        ref_depth = None

    camera_poses = {"h": h, "w": w, "intrinsic": ref_intrinsic[0].numpy().tolist(), "extrinsic": dict()}
    for i in range(len(ref_names)):
        camera_poses['extrinsic'][ref_names[i].split('.')[0].replace('_ref', '') + ".png"] = ref_extrinsic[i].numpy().tolist()
    for i in range(len(tar_names)):
        camera_poses['extrinsic'][tar_names[i].split('.')[0].replace('_ref', '') + ".png"] = tar_extrinsic[i].numpy().tolist()

    return {"ref_images": ref_images, "ref_intrinsic": ref_intrinsic, "tar_intrinsic": tar_intrinsic,
            "ref_extrinsic": ref_extrinsic, "tar_extrinsic": tar_extrinsic, "ref_depth": ref_depth,
            "ref_names": ref_names, "tar_names": tar_names}


def eval(args, config, data, pipeline):
    N_target = data['tar_intrinsic'].shape[0]
    gen_num = config.nframe - args.cond_num

    # save reference images
    for i in range(data['ref_images'].shape[0]):
        ref_img = ToPILImage()((data['ref_images'][i] + 1) / 2)
        ref_img.save(f"{config.save_path}/images/{data['ref_names'][i].split('.')[0]}.png")

    with torch.no_grad(), torch.autocast("cuda"):
        iter_times = N_target // gen_num
        if N_target % gen_num != 0:
            iter_times += 1
        for i in range(iter_times):
            print(f"synthesis target views {np.arange(N_target)[i::iter_times].tolist()}...")
            h, w = data['ref_images'].shape[2], data['ref_images'].shape[3]
            gen_num_ = len(np.arange(N_target)[i::iter_times].tolist())
            print(f"Gen num {gen_num_ + args.cond_num}...")
            image = torch.cat([data["ref_images"], torch.zeros((gen_num_, 3, h, w), dtype=torch.float32)], dim=0).to("cuda")
            intrinsic = torch.cat([data["ref_intrinsic"], data["tar_intrinsic"][i::iter_times]], dim=0).to("cuda")
            extrinsic = torch.cat([data["ref_extrinsic"], data["tar_extrinsic"][i::iter_times]], dim=0).to("cuda")
            if data["ref_depth"] is not None:
                depth = torch.cat([data["ref_depth"], torch.zeros((gen_num_, 1, h, w), dtype=torch.float32)], dim=0).to("cuda")
            else:
                depth = None

            nframe_new = gen_num_ + args.cond_num
            config_copy = copy.deepcopy(config)
            config_copy.nframe = nframe_new
            generator = torch.Generator()
            generator = generator.manual_seed(args.seed)
            st = time.time()
            preds = pipeline(images=image, nframe=nframe_new, cond_num=args.cond_num,
                             key_rescale=args.key_rescale, height=h, width=w, intrinsics=intrinsic,
                             extrinsics=extrinsic, num_inference_steps=50, guidance_scale=args.val_cfg,
                             output_type="np", config=config_copy, tag=["custom"] * image.shape[0],
                             class_label=args.class_label, depth=depth, vae=pipeline.vae, generator=generator).images  # [f,h,w,c]
            print("Time used:", time.time() - st)
            preds = preds[args.cond_num:]
            preds = (preds * 255).astype(np.uint8)

            for j in range(preds.shape[0]):
                cv2.imwrite(f"{config.save_path}/images/{data['tar_names'][i::iter_times][j].split('.')[0]}.png", preds[j, :, :, ::-1])

            if config.model_cfg.get("enable_depth", False) and config.model_cfg.get("priors3d", False):
                color_warps = global_position_encoding_3d(config_copy, depth, intrinsic, extrinsic,
                                                          args.cond_num, nframe=nframe_new, device=device,
                                                          pe_scale=1 / 8, embed_dim=config.model_cfg.get("coord_dim", 192),
                                                          colors=image)[0]

                cv2.imwrite(f"{config.save_path}/warp{np.arange(N_target)[i::iter_times].tolist()}.png", color_warps[:, :, ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="build cam traj")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="check_points/pretrained_model", help="model directory.")
    parser.add_argument("--output_path", type=str, default="outputs/demo")
    parser.add_argument("--val_cfg", type=float, default=2.0)
    parser.add_argument("--key_rescale", type=float, default=None)
    parser.add_argument("--camera_longest_side", type=float, default=5.0)
    parser.add_argument("--nframe", type=int, default=28)
    parser.add_argument("--min_conf_thr", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--class_label", type=int, default=0)
    parser.add_argument("--target_limit", type=int, default=None)
    # single-view parameters
    parser.add_argument("--center_scale", type=float, default=1.0)
    parser.add_argument("--elevation", type=float, default=5.0, help="the initial elevation angle")
    parser.add_argument("--d_theta", type=float, default=0.0, help="elevation rotation angle")
    parser.add_argument("--d_phi", type=float, default=45.0, help="azimuth rotation angle")
    parser.add_argument("--d_r", type=float, default=1.0, help="the distance from camera to the world center")
    parser.add_argument("--x_offset", type=float, default=0.0, help="up moving")
    parser.add_argument("--y_offset", type=float, default=0.0, help="left moving")
    parser.add_argument("--median_depth", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--cam_traj", type=str, default="free",
                        choices=["free", "bi_direction", "disorder", "swing1", "swing2"])

    args = parser.parse_args()
    config = EasyDict(OmegaConf.load(os.path.join(args.model_dir, "config.yaml")))
    if config.nframe != args.nframe:
        print(f"Extend nframe from {config.nframe} to {args.nframe}.")
        config.nframe = args.nframe
        if config.nframe > 28 and args.key_rescale is None:
            args.key_rescale = 1.2
        print("key rescale", args.key_rescale)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda"

    save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)
    if os.path.isfile(args.input_path):
        input_files = [args.input_path]
    else:
        input_files = glob(f"{args.input_path}/*")  # take all figures as conditional views
    args.cond_num = len(input_files)

    ### Step1: get camera trajectory ###
    print("Get camera traj...")
    if args.cond_num > 1:
        # you can put the path to a local checkpoint in model_name if needed
        dust3r = AsymmetricCroCo3DStereo.from_pretrained("./check_points/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth").to(device)
        dust3r_images = load_images(input_files, size=512, square_ok=True)
        pairs = make_pairs(dust3r_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, dust3r, device, batch_size=1)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, same_focals=True)
        loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
        c2ws = scene.get_im_poses().detach().cpu()
        cam_pos = c2ws[:, :3, -1].numpy()  # [N,3]
        max_scale = [cam_pos[:, 0].max(), cam_pos[:, 1].max(), cam_pos[:, 2].max()]
        min_scale = [cam_pos[:, 0].min(), cam_pos[:, 1].min(), cam_pos[:, 2].min()]
        cam_size = np.array(max_scale) - np.array(min_scale)
        max_size = np.max(cam_size)
        rescale = args.camera_longest_side / max_size if max_size > args.camera_longest_side else 1.0

        w2cs = torch.inverse(c2ws)
        w2cs[:, :3, 3:4] *= rescale
        c2ws[:, :3, 3:4] *= rescale

        Ks = scene.get_intrinsics().detach().cpu()
        origin_w, origin_h = None, None
        for i in range(len(input_files)):
            origin_w, origin_h = imagesize.get(input_files[i])
            new_h, new_w = dust3r_images[i]['true_shape'][0, 0], dust3r_images[i]['true_shape'][0, 1]
            Ks[i, 0] *= (origin_w / new_w)
            Ks[i, 1] *= (origin_h / new_h)

        dust3r_depths = scene.get_depthmaps()
        dust3r_depths = [d.detach().cpu() * rescale for d in dust3r_depths]
        scene.min_conf_thr = args.min_conf_thr
        confidence_masks = scene.get_masks()
        confidence_masks = [c.detach().cpu() for c in confidence_masks]

        pts3d = scene.get_pts3d()
        imgs = scene.imgs
        points3d = []
        colors = []
        for i in range(len(input_files)):
            color = imgs[i]
            points = pts3d[i].detach().cpu().numpy() * rescale
            mask = confidence_masks[i].detach().cpu().numpy()
            points3d.append(points[mask])
            colors.append(color[mask])
        points3d = np.concatenate(points3d)
        colors = np.concatenate(colors)
        colors = (np.clip(colors, 0, 1.0) * 255).astype(np.uint8)

        Ks = Ks.numpy()
        K = np.mean(Ks, axis=0)
        w2cs = w2cs.numpy()
        c2ws = c2ws.numpy()
        depth = dust3r_depths
        for i in range(len(depth)):
            depth[i][confidence_masks == False] = 0
            depth[i] = depth[i].numpy()
            depth[i] = cv2.resize(depth[i], (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        # 旋转矩阵转为欧拉角，并插值
        print("Build the novel frames...")
        c2ws_all = [c2ws[0]]
        w2cs_all = [w2cs[0]]
        reference_views = [0]

        nframes = []
        for i in range(len(input_files) - 1):
            if i != len(input_files) - 2:
                nframes.append((args.nframe - len(input_files)) // (len(input_files) - 1))
            else:
                nframes.append((args.nframe - len(input_files)) - int(np.sum(nframes)))

        cam_idx = 1
        for j in range(len(input_files) - 1):
            # offset interpolation
            pos0 = c2ws[j, :3, -1]
            pos1 = c2ws[j + 1, :3, -1]
            R0 = w2cs[j, :3, :3]
            R1 = w2cs[j + 1, :3, :3]
            rotation0 = Rotation.from_matrix(R0)
            rotation1 = Rotation.from_matrix(R1)
            euler_angles0 = rotation0.as_euler('xyz', degrees=True)
            euler_angles1 = rotation1.as_euler('xyz', degrees=True)

            # 检查是否有符号骤变
            sign_diff = np.sign(euler_angles0) * np.sign(euler_angles1)
            for i_ in range(len(sign_diff)):
                # 先变为连续角度180°-->360°
                if sign_diff[i_] == -1 and abs(euler_angles0[i_]) + abs(euler_angles1[i_]) > 180:
                    if euler_angles1[i_] > 0:
                        euler_angles1[i_] = -360 + euler_angles1[i_]
                    else:
                        euler_angles1[i_] = 360 + euler_angles1[i_]
            for i in range(nframes[j]):
                coef = (i + 1) / (nframes[j] + 1)
                pos_mid = (1 - coef) * pos0 + coef * pos1
                euler_angles = (1 - coef) * euler_angles0 + coef * euler_angles1
                for i_ in range(len(sign_diff)):
                    # 360°-->180°
                    if sign_diff[i_] == -1 and abs(euler_angles0[i_]) + abs(euler_angles1[i_]) > 180:
                        if euler_angles[i_] > 180:
                            euler_angles[i_] = -360 + euler_angles[i_]
                        elif euler_angles[i_] < -180:
                            euler_angles[i_] = 360 + euler_angles[i_]
                print(j, euler_angles0, euler_angles1, euler_angles)
                # 将欧拉角转换回旋转矩阵
                rotation_from_euler = Rotation.from_euler('xyz', euler_angles, degrees=True)
                R_mid = rotation_from_euler.as_matrix()
                R_mid = R_mid.T
                c2w_mid = np.concatenate([R_mid.reshape((3, 3)), pos_mid.reshape((3, 1))], axis=-1)
                c2w_mid = np.concatenate([c2w_mid, np.zeros((1, 4))], axis=0)
                c2w_mid[-1, -1] = 1
                w2c_mid = np.linalg.inv(c2w_mid)

                c2ws_all.append(c2w_mid)
                w2cs_all.append(w2c_mid)

                cam_idx += 1

            c2ws_all.append(c2ws[j + 1])
            w2cs_all.append(w2cs[j + 1])
            reference_views.append(len(c2ws_all) - 1)

        print("Multi-view trajectory building over...")
    else:
        input_path = input_files[0]
        origin_w, origin_h = imagesize.get(input_path)
        depth_pro_model, transform = create_model_and_transforms(device=torch.device("cuda"))
        depth_pro_model.eval()
        # GET depth and intrinsic information
        print("Inference depth...")
        # Load and preprocess an image.
        image, _, f_px = load_rgb(input_path)
        image = transform(image)
        # Run inference.
        prediction = depth_pro_model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.

        _, h, w = image.shape
        K = torch.tensor([[focallength_px, 0, w / 2],
                          [0, focallength_px, h / 2],
                          [0, 0, 1]], dtype=torch.float32, device=device)
        K_inv = K.inverse()

        points2d = torch.stack(torch.meshgrid(torch.arange(w, dtype=torch.float32),
                                              torch.arange(h, dtype=torch.float32), indexing="xy"), -1).to(device)  # [h,w,2]
        points3d = points_padding(points2d).reshape(h * w, 3)  # [hw,3]
        points3d = (K_inv @ points3d.T * depth.reshape(1, h * w)).T
        colors = ((image + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).reshape(h * w, 3)
        points3d = points3d.cpu().numpy()
        colors = colors.cpu().numpy()

        # 以画面中心depth为新世界坐标系原点
        print("Build the first camera")
        if args.median_depth:
            if args.foreground:
                from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7

                seg_net = TracerUniversalB7(device=device, batch_size=1).to(torch.float16)
                with torch.no_grad(), torch.autocast("cuda"):
                    image_pil = Image.open(input_path)
                    origin_w, origin_h = image_pil.size
                    image_pil = image_pil.resize((512, 512))
                    fg_mask = seg_net([image_pil])[0]
                    fg_mask = fg_mask.resize((origin_w, origin_h))
                fg_mask = np.array(fg_mask)
                fg_mask = fg_mask > 127.5
                fg_mask = torch.tensor(fg_mask)
                if fg_mask.sum() == 0:
                    fg_mask[...] = True
                depth_avg = torch.median(depth[fg_mask]).item()
            else:
                depth_avg = torch.median(depth).item()
        else:
            depth_avg = depth[h // 2, w // 2]  # 以图像中心处的depth(z)为球心旋转
        radius = depth_avg * args.center_scale
        c2w_0 = torch.tensor([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, -radius],
                              [0, 0, 0, 1]], dtype=torch.float32)
        # 俯仰角调整（绕x轴旋转）
        elevation_rad = np.deg2rad(args.elevation)
        R_elevation = torch.tensor([[1, 0, 0, 0],
                                    [0, np.cos(-elevation_rad), -np.sin(-elevation_rad), 0],
                                    [0, np.sin(-elevation_rad), np.cos(-elevation_rad), 0],
                                    [0, 0, 0, 1]], dtype=torch.float32)
        c2w_0 = R_elevation @ c2w_0
        w2c_0 = c2w_0.inverse()

        # DEBUG: 根据d_theta，d_phi, d_r构造多个camera view
        c2ws_all = [c2w_0]
        w2cs_all = [w2c_0]
        reference_views = [0]
        d_theta, d_phi, d_r = [], [], []
        x_offsets, y_offsets = [], []
        if args.cam_traj == "disorder":
            for i in range(args.nframe - 1):
                coef = random.random()
                d_theta.append(args.d_theta * coef)
                coef = random.random()
                if random.random() < 0.5:
                    d_phi.append(args.d_phi * coef)
                else:
                    d_phi.append(-args.d_phi * coef)
                coef = random.random()
                d_r.append(args.d_range[0] + coef * (args.d_range[1] - args.d_range[0]))
        elif args.cam_traj == "bi_direction":
            nframe1 = (args.nframe - 1) // 2
            nframe2 = (args.nframe - 1) - nframe1
            for i in range(nframe1):
                coef = (i + 1) / nframe1
                d_theta.append(args.d_theta * coef)
                d_phi.append(-args.d_phi * coef)
                d_r.append(coef * args.d_r + (1 - coef) * 1.0)
            d_theta = d_theta[::-1]
            d_phi = d_phi[::-1]
            d_r = d_r[::-1]
            for i in range(nframe2):
                coef = (i + 1) / nframe2
                d_theta.append(args.d_theta * coef)
                d_phi.append(args.d_phi * coef)
                d_r.append(coef * args.d_r + (1 - coef) * 1.0)
        elif args.cam_traj == "swing1":
            phis_ = [0, -5, -25, -30, -20, -8, 0]
            thetas_ = [0, -3, -7, -15, -12, -7, 0, 3, 7, 15, 12, 7, 0]
            rs_ = [0, 0]
            d_phi = txt_interpolation(phis_, args.nframe, mode='smooth')
            d_phi[0] = phis_[0]
            d_phi[-1] = phis_[-1]
            d_theta = txt_interpolation(thetas_, args.nframe, mode='smooth')
            d_theta[0] = thetas_[0]
            d_theta[-1] = thetas_[-1]
            d_r = txt_interpolation(rs_, args.nframe, mode='linear')
            d_r = 1.0 + d_r
        elif args.cam_traj == "swing2":
            phis_ = [0, 5, 25, 30, 20, 10, 0]
            thetas_ = [0, -3, -12, -9, 0, 3, 12, 9, 0]
            rs_ = [0, -0.03, -0.13, -0.27, -0.24, -0.13, 0]
            d_phi = txt_interpolation(phis_, args.nframe, mode='smooth')
            d_phi[0] = phis_[0]
            d_phi[-1] = phis_[-1]
            d_theta = txt_interpolation(thetas_, args.nframe, mode='smooth')
            d_theta[0] = thetas_[0]
            d_theta[-1] = thetas_[-1]
            d_r = txt_interpolation(rs_, args.nframe, mode='smooth')
            d_r = 1.0 + d_r
        else:
            for i in range(args.nframe - 1):
                coef = (i + 1) / (args.nframe - 1)
                d_theta.append(args.d_theta * coef)
                d_phi.append(args.d_phi * coef)
                d_r.append(coef * args.d_r + (1 - coef) * 1.0)
                x_offsets.append(radius * args.x_offset * ((i + 1) / args.nframe))
                y_offsets.append(radius * args.y_offset * ((i + 1) / args.nframe))

        for i in range(args.nframe - 1):
            d_theta_rad = np.deg2rad(d_theta[i])
            R_theta = torch.tensor([[1, 0, 0, 0],
                                    [0, np.cos(d_theta_rad), -np.sin(d_theta_rad), 0],
                                    [0, np.sin(d_theta_rad), np.cos(d_theta_rad), 0],
                                    [0, 0, 0, 1]], dtype=torch.float32)
            d_phi_rad = np.deg2rad(d_phi[i])
            R_phi = torch.tensor([[np.cos(d_phi_rad), 0, np.sin(d_phi_rad), 0],
                                  [0, 1, 0, 0],
                                  [-np.sin(d_phi_rad), 0, np.cos(d_phi_rad), 0],
                                  [0, 0, 0, 1]], dtype=torch.float32)
            c2w_1 = R_phi @ R_theta @ c2w_0
            if i < len(x_offsets):
                c2w_1[:3, -1] += torch.tensor([x_offsets[i], y_offsets[i], 0])
            c2w_1[:3, -1] *= d_r[i]
            w2c_1 = c2w_1.inverse()
            c2ws_all.append(c2w_1)
            w2cs_all.append(w2c_1)

        depth = [depth.cpu().numpy()]
        print("Single-view trajectory building over...")

    # save pointcloud and cameras
    scene = trimesh.Scene()
    for i in range(len(c2ws_all)):
        add_scene_cam(scene, c2ws_all[i], CAM_COLORS[i % len(CAM_COLORS)], None, imsize=(512, 512), screen_width=0.03)

    pcd = trimesh.PointCloud(vertices=points3d, colors=colors)
    _ = pcd.export(f"{save_path}/pcd.ply")
    scene.export(file_obj=f"{save_path}/cameras.glb")

    reference_cam = {"h": origin_h, "w": origin_w, "intrinsic": K.tolist()}
    reference_cam["extrinsic"] = dict()
    target_cam = copy.deepcopy(reference_cam)
    for i in range(len(reference_views)):
        reference_cam["extrinsic"][f"view{str(reference_views[i]).zfill(3)}_ref"] = w2cs_all[reference_views[i]].tolist()

    for i in range(len(w2cs_all)):
        if i not in reference_views:
            target_cam["extrinsic"][f"view{str(i).zfill(3)}"] = w2cs_all[i].tolist()

    ### Step2: generate multi-view images ###
    # init model
    print("load model...")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path,
                                        subfolder="vae", local_files_only=True)
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                rank=0,
                                                model_cfg=config.model_cfg,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True,
                                                local_files_only=True)
    unet.requires_grad_(False)
    # load pretained weights
    weights = torch.load(f"{args.model_dir}/ema_unet.pt", map_location="cpu")
    unet.load_state_dict(weights)
    unet.eval()

    weight_dtype = torch.float16
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    scheduler = get_diffusion_scheduler(config, name="DDIM")
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    pipeline = pipeline.to(device)

    # load dataset
    args.dataset_dir = save_path
    config.save_path = save_path
    data = load_dataset(args, config, reference_cam, target_cam, input_files, depth)

    os.makedirs(f"{save_path}/images", exist_ok=True)
    eval(args, config, data, pipeline)

    results = glob(f"{config.save_path}/images/view*.png")
    results.sort(key=lambda x: int(x.split('/')[-1].replace(".png", "").replace("view", "").replace("_ref", "")))
    clip = ImageSequenceClip(results, fps=15)
    clip.write_videofile(f"{config.save_path}/output.mp4", fps=15)
