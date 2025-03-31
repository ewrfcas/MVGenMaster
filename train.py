import argparse
import collections
import copy
import gc
import logging
import math
import os
import random
import shutil

import accelerate
import cv2
import diffusers
import einops
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from easydict import EasyDict
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
from my_diffusers.training_utils import EMAModel
from src.datasets.global_datasets import load_global_dataset
from src.datasets.global_sampler import GlobalConcatSampler
from src.modules.camera import get_camera_embedding
from src.modules.position_encoding import depth_freq_encoding, global_position_encoding_3d, get_3d_priors
from src.modules.schedulers import get_diffusion_scheduler
from utils import get_lpips_score, _seq_name_to_seed

logger = get_logger(__name__, log_level="INFO")


def resort_batch(batch, nframe, bsz):
    sequence_name = np.array(batch["sequence_name"]).reshape(bsz, nframe)
    indices = np.argsort(sequence_name, axis=1)
    batch["image"] = einops.rearrange(batch["image"], "(b f) c h w -> b f c h w", f=nframe)
    batch["intrinsic"] = einops.rearrange(batch["intrinsic"], "(b f) c1 c2 -> b f c1 c2", f=nframe)
    batch["extrinsic"] = einops.rearrange(batch["extrinsic"], "(b f) c1 c2 -> b f c1 c2", f=nframe)

    if "depth" in batch and batch["depth"] is not None:
        batch["depth"] = einops.rearrange(batch["depth"], "(b f) c h w -> b f c h w", f=nframe)

    for i in range(bsz):  # we do not need to sort megascenes and front3d
        if batch["tag"][i * nframe] not in ("megascenes", "front3d"):
            batch["image"][i] = batch["image"][i, indices[i]]
            batch["intrinsic"][i] = batch["intrinsic"][i, indices[i]]
            batch["extrinsic"][i] = batch["extrinsic"][i, indices[i]]
            sequence_name[i] = sequence_name[i, indices[i]]

            if "depth" in batch and batch["depth"] is not None:
                batch["depth"][i] = batch["depth"][i, indices[i]]

    batch["image"] = einops.rearrange(batch["image"], "b f c h w -> (b f) c h w", f=nframe)
    batch["intrinsic"] = einops.rearrange(batch["intrinsic"], "b f c1 c2 -> (b f) c1 c2", f=nframe)
    batch["extrinsic"] = einops.rearrange(batch["extrinsic"], "b f c1 c2 -> (b f) c1 c2", f=nframe)
    batch["sequence_name"] = list(sequence_name.reshape(-1))

    if "depth" in batch and batch["depth"] is not None:
        batch["depth"] = einops.rearrange(batch["depth"], "b f c h w -> (b f) c h w", f=nframe)

    return batch


def slice_vae_encode(vae, image, sub_size):  # vae fails to encode large tensor directly, we need to slice it
    with torch.no_grad(), torch.autocast("cuda", enabled=True):
        if len(image.shape) == 5:  # [B,F,C,H,W]
            b, f, _, h, w = image.shape
            image = einops.rearrange(image, "b f c h w -> (b f) c h w")
        else:
            b, f, h, w = None, None, None, None

        if (image.shape[-1] > 256 and image.shape[0] > sub_size) or (image.shape[0] > 192):
            slice_num = image.shape[0] // sub_size
            if image.shape[0] % sub_size != 0:
                slice_num += 1
            latents = []
            for i in range(slice_num):
                latents_ = vae.encode(image[i * sub_size:(i + 1) * sub_size]).latent_dist.sample()
                latents.append(latents_)
            latents = torch.cat(latents, dim=0)
        else:
            latents = vae.encode(image).latent_dist.sample()

        if f is not None:
            latents = einops.rearrange(latents, "(b f) c h w -> b f c h w", f=f)

        return latents


def get_pipeline(accelerator, config, vae, unet, weight_dtype):
    scheduler = get_diffusion_scheduler(config, name="DDIM")
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    return pipeline


def get_show_images(input_images, pred_images, cond_num, depth=None):
    pred_images = [pred_images[i] for i in range(pred_images.shape[0])]
    pred_images = np.clip(np.concatenate(pred_images, axis=1) * 255, 0, 255).astype(np.uint8)  # [H,W*F,c]
    input_images = (input_images + 1) / 2
    ground_truths = np.concatenate([np.array(ToPILImage()(input_images[i])) for i in range(input_images.shape[0])], axis=1)
    input_images[cond_num:] = 0
    inputs = np.concatenate([np.array(ToPILImage()(input_images[i])) for i in range(input_images.shape[0])], axis=1)
    if depth is not None:
        min_vals = depth.amin(dim=[2, 3], keepdim=True)
        max_vals = depth.amax(dim=[2, 3], keepdim=True)
        depth = (depth - min_vals) / (max_vals - min_vals)
        depth[cond_num:] = 0
        depth = np.concatenate([np.array(ToPILImage()(depth[i]).convert("RGB")) for i in range(depth.shape[0])], axis=1)
        show_image = np.concatenate([inputs, depth, ground_truths, pred_images], axis=0)
    else:
        show_image = np.concatenate([inputs, ground_truths, pred_images], axis=0)

    return show_image


def log_validation(accelerator, config, args, pipeline, val_dataloader, step, device, **kwargs):
    if accelerator.is_main_process:
        logger.info(f"Validation log in step {step}")

    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    show_images = []
    show_save_dict = collections.defaultdict(int)
    with torch.no_grad(), torch.autocast("cuda"):
        for batch in tqdm(val_dataloader, desc=f"Validation rank{accelerator.process_index}..."):
            image = batch["image"].to(device)  # -1~1 tensor [1*F,3,H,W] batch=1 for validation
            if config.get("fix_cond_num", None) is not None:
                cond_num = config.fix_cond_num
            else:
                cond_num = random.Random(_seq_name_to_seed(batch["sequence_name"][0])).randint(1, config.max_cond_num)

            if config.model_cfg.get("enable_depth", False):
                depth_freq = config.model_cfg.get("depth_freq", None)
                if depth_freq is not None:
                    depth = depth_freq_encoding(batch["depth"], device=device, embed_dim=depth_freq)
                else:
                    depth = batch["depth"].to(torch.float32)
            else:
                depth = None

            preds = pipeline(images=image, nframe=config.nframe, cond_num=cond_num,
                             height=image.shape[2], width=image.shape[3],
                             intrinsics=batch['intrinsic'], extrinsics=batch['extrinsic'],
                             num_inference_steps=50, guidance_scale=args.val_cfg,
                             output_type="np", config=config, tag=batch["tag"],
                             sequence_name=batch["sequence_name"], depth=depth, vae=kwargs['vae']).images  # [f,h,w,c]

            if config.model_cfg.get("enable_depth", False) and config.model_cfg.get("priors3d", False):
                color_warps = global_position_encoding_3d(config, depth, batch['intrinsic'],
                                                          batch['extrinsic'], 1,
                                                          nframe=config.nframe, device=accelerator.device,
                                                          pe_scale=1 / 8,
                                                          embed_dim=config.model_cfg.get("coord_dim", 192),
                                                          colors=image)
            else:
                color_warps = None

            if batch['tag'][0] not in show_save_dict or show_save_dict[batch['tag'][0]] < 10:  # 每个dataset显示10个
                show_save_dict[batch['tag'][0]] += 1
                if depth is not None:
                    show_image = get_show_images(image, preds, cond_num, batch["depth"])
                else:
                    show_image = get_show_images(image, preds, cond_num)

                if color_warps is not None:
                    h, w = image.shape[2], image.shape[3]
                    show_image[h:h * 2, cond_num * w:] = color_warps[0][:, cond_num * w:]
                show_images.append(show_image)

            gt_images = (image[cond_num:].permute(0, 2, 3, 1).cpu().numpy() + 1) / 2
            preds = preds[cond_num:]

            for i in range(preds.shape[0]):
                psnr_ = peak_signal_noise_ratio(gt_images[i], preds[i], data_range=1.0)
                psnr_scores.append(psnr_)
                ssim_ = structural_similarity(cv2.cvtColor(gt_images[i], cv2.COLOR_RGB2GRAY),
                                              cv2.cvtColor(preds[i], cv2.COLOR_RGB2GRAY), data_range=1.0)
                ssim_scores.append(ssim_)
                lpips_ = get_lpips_score(loss_fn_alex, gt_images[i], preds[i], device)
                lpips_scores.append(lpips_)

    # unify all results
    psnr_score = torch.tensor(np.mean(psnr_scores), device=device, dtype=torch.float32)
    ssim_score = torch.tensor(np.mean(ssim_scores), device=device, dtype=torch.float32)
    lpips_score = torch.tensor(np.mean(lpips_scores), device=device, dtype=torch.float32)

    psnr_score = accelerator.gather(psnr_score).mean().item()
    ssim_score = accelerator.gather(ssim_score).mean().item()
    lpips_score = accelerator.gather(lpips_score).mean().item()

    if accelerator.is_main_process:
        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
        tracker.add_scalar("val/psnr", psnr_score, global_step=step)
        tracker.add_scalar("val/ssim", ssim_score, global_step=step)
        tracker.add_scalar("val/lpips", lpips_score, global_step=step)

        for j in range(len(show_images)):
            if config.image_size > 256:
                show_images[j] = cv2.resize(show_images[j], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            tracker.add_images(f"val/gt_masked_pred_images{j}", show_images[j], step, dataformats="HWC")

    del loss_fn_alex
    torch.cuda.empty_cache()

    return lpips_score


def log_train(accelerator, config, args, pipeline, weight_dtype, batch, step, **kwargs):
    logger.info(f"Train log in step {step}")
    nframe = kwargs['nframe']

    image = batch["image"][:nframe]  # only show one group
    intrinsic = batch["intrinsic"][:nframe]
    extrinsic = batch["extrinsic"][:nframe]
    tag = batch["tag"][:nframe]
    sequence_name = batch["sequence_name"][:nframe]

    if config.model_cfg.get("enable_depth", False):
        depth = batch["depth"][:nframe].to(torch.float32)
        depth_freq = config.model_cfg.get("depth_freq", None)
        if depth_freq is not None:
            depth = depth_freq_encoding(depth, device=kwargs['device'], embed_dim=depth_freq)
    else:
        depth = None

    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        preds = pipeline(images=image, nframe=nframe, cond_num=kwargs.get("random_cond_num", 1),
                         height=image.shape[2], width=image.shape[3],
                         intrinsics=intrinsic, extrinsics=extrinsic,
                         num_inference_steps=50, guidance_scale=args.val_cfg,
                         output_type="np", config=config, tag=tag, sequence_name=sequence_name,
                         depth=depth, vae=kwargs['vae']).images  # [f,h,w,c]

        if config.model_cfg.get("enable_depth", False) and config.model_cfg.get("priors3d", False):
            color_warps = global_position_encoding_3d(config, depth, intrinsic,
                                                      extrinsic, 1,
                                                      nframe=nframe, device=accelerator.device,
                                                      pe_scale=1 / 8,
                                                      embed_dim=config.model_cfg.get("coord_dim", 192),
                                                      colors=image)
        else:
            color_warps = None

    if depth is not None:
        show_image = get_show_images(image, preds, kwargs.get("random_cond_num", 1), batch["depth"][:nframe])
    else:
        show_image = get_show_images(image, preds, kwargs.get("random_cond_num", 1))

    if color_warps is not None:
        h, w = image.shape[2], image.shape[3]
        show_image[h:h * 2, kwargs.get("random_cond_num", 1) * w:] = color_warps[0][:, kwargs.get("random_cond_num", 1) * w:]

    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
    tracker.add_images("train/gt_masked_pred_images", show_image, step, dataformats="HWC")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="MV-Gen",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--train_log_interval", type=int, default=500)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--val_cfg", type=float, default=1.0)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--restart_global_step", default=0, type=int)
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--min_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay_step", type=int, default=10)
    parser.add_argument("--reset_ema_step", action="store_true")
    parser.add_argument("--only_resume_weight", action="store_true")
    parser.add_argument("--max_nframe", type=int, default=None)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resume_from_checkpoint is not None and args.resume_path is None:
        config = EasyDict(OmegaConf.load(os.path.join(args.output_dir, "config.yaml")))
        cfg = dict()
        for data_name in config.dataset_names:
            data_name = data_name.replace('_easy', '')
            cfg[data_name] = EasyDict(OmegaConf.load(os.path.join(args.output_dir, f"{data_name}.yaml")))
    else:
        config = EasyDict(OmegaConf.load(args.config_file))
        cfg = None

    # Sanity checks
    assert config.dataset_names is not None and len(config.dataset_names) > 0

    # dynamic nframe training
    if args.max_nframe is not None and args.max_nframe > config.nframe:
        config.dynamic_nframe = True
        config.origin_nframe = config.nframe
        config.min_nframe = config.nframe
        config.max_nframe = args.max_nframe
        config.nframe = args.max_nframe
    else:
        config.dynamic_nframe = False

    return args, config, cfg


def main():
    args, config, cfg = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    train_noise_scheduler = get_diffusion_scheduler(config, name="DDPM")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    accelerator.print("Loading model weights...")
    # take text_encoder and vae away from parameter sharding across multi-gpu in ZeRO
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(f"{config.pretrained_model_name_or_path}", subfolder="vae")
        vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                rank=accelerator.process_index,
                                                model_cfg=config.model_cfg,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True)
    unet.train()

    if args.use_ema:
        ema_unet = copy.deepcopy(unet)
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=True,
            decay=args.ema_decay,
            min_decay=args.min_decay,
            ema_decay_step=args.ema_decay_step
        )
    else:
        ema_unet = None

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.opt_cfg.scale_lr:
        config.learning_rate = (
                config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # set trainable parameters
    trainable_params = []
    params = [p for n, p in unet.named_parameters() if p.requires_grad]
    trainable_params.append({'params': params, 'lr': config.opt_cfg.learning_rate})

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.opt_cfg.learning_rate,
        betas=(config.opt_cfg.adam_beta1, config.opt_cfg.adam_beta2),
        weight_decay=config.opt_cfg.adam_weight_decay,
        eps=config.opt_cfg.adam_epsilon,
    )

    # Get the datasets
    if cfg is None:
        cfg = dict()
        for data_name in config.dataset_names:
            if "_easy" in data_name:
                data_name = data_name.replace("_easy", "")
                cfg[data_name] = EasyDict(OmegaConf.load(f"configs/datasets/{data_name}_easy.yaml"))
            else:
                cfg[data_name] = EasyDict(OmegaConf.load(f"configs/datasets/{data_name}.yaml"))
            cfg[data_name]["image_height"] = config.image_size
            cfg[data_name]["image_width"] = config.image_size

    train_dataset, val_dataset = load_global_dataset(config, cfg, rank=accelerator.process_index)

    if accelerator.is_main_process:
        OmegaConf.save(dict(config), os.path.join(args.output_dir, 'config.yaml'))
        for data_name in cfg:
            OmegaConf.save(dict(cfg[data_name]), os.path.join(args.output_dir, f'{data_name}.yaml'))

    sampler = GlobalConcatSampler(train_dataset,
                                  n_frames_per_sample=config.nframe,
                                  shuffle=True,
                                  dynamic_sampling=config.get("dynamic_sampling", False),
                                  multi_scale=config.get("multi_scale", False),
                                  batch_per_gpu=config.train_batch_size // config.nframe,
                                  rank=accelerator.process_index,
                                  num_replicas=accelerator.num_processes,
                                  data_config=cfg)
    val_sampler = GlobalConcatSampler(val_dataset,
                                      n_frames_per_sample=config.nframe,
                                      shuffle=False,
                                      rank=accelerator.process_index,
                                      multi_scale=config.get("multi_scale", False),
                                      batch_per_gpu=1,
                                      num_replicas=accelerator.num_processes,
                                      data_config=cfg)

    assert config.train_batch_size % config.nframe == 0

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=sampler,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        batch_size=config.nframe,
        num_workers=4,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        config.opt_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.opt_cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return [total_num, trainable_num]

    param_info_vae = get_parameter_number(vae)
    accelerator.print(f'########## VAE, Total:{param_info_vae[0] / 1e6}M, Trainable:{param_info_vae[1] / 1e6}M ##################')
    param_info_unet = get_parameter_number(unet)
    accelerator.print(f'########## Unet, Total:{(param_info_unet[0]) / 1e6}M, Trainable:{(param_info_unet[1]) / 1e6}M ##################')

    if args.only_resume_weight:
        if accelerator.is_main_process:
            print("Loading weight only...")
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            resume_path = args.resume_path
        else:
            # Get the most recent checkpoint
            resume_path = args.resume_path if args.resume_path is not None else args.output_dir
            dirs = os.listdir(resume_path)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if args.use_ema:
            if os.path.exists(f"{resume_path}/{path}/ema_unet.pt"):
                print("Find ema weights, load it!")
                weights = torch.load(f"{resume_path}/{path}/ema_unet.pt", map_location="cpu")
                # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
                unet.load_state_dict(weights, strict=False)  # unet load first
            else:
                print("No ema weights, load original weights instead!")
                weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
                # here the weights are different from ema_unet (maybe some new weights are in ema_unet/unet)
                unet.load_state_dict(weights['module'], strict=False)  # unet load first
            ema_unet.load_state_dict({"shadow_params": [p.clone().detach() for p in list(unet.parameters())]})
            if not args.reset_ema_step:
                ema_params = torch.load(f"{resume_path}/{path}/ema_unet_params.pt", map_location="cpu")
                ema_unet.optimization_step = ema_params['optimization_step']

        # reload weights for unet here
        weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
        unet.load_state_dict(weights['module'], strict=False)  # we usually need to resume partial weights here

    # Prepare everything with our `accelerator`.
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config.train_batch_size
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)  # train_dataloader

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name)

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            resume_path = args.resume_path
        else:
            # Get the most recent checkpoint
            resume_path = args.resume_path if args.resume_path is not None else args.output_dir
            dirs = os.listdir(resume_path)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:

            if not args.only_resume_weight:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(resume_path, path))

            if args.restart_global_step != 0:
                global_step = args.restart_global_step
            else:
                global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            if args.use_ema and not args.only_resume_weight:
                # here we load state because we need to restore all ema states, while weights are not changed
                temp_unet = copy.deepcopy(accelerator.unwrap_model(unet)).cpu()
                if os.path.exists(f"{resume_path}/{path}/ema_unet.pt"):
                    print("Find ema weights, load it!")
                    weights = torch.load(f"{resume_path}/{path}/ema_unet.pt", map_location="cpu")
                    temp_unet.load_state_dict(weights)
                    if not args.reset_ema_step:
                        ema_params = torch.load(f"{resume_path}/{path}/ema_unet_params.pt", map_location="cpu")
                        ema_unet.optimization_step = ema_params['optimization_step']
                else:
                    print("No ema weights, load original weights instead!")
                    weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
                    temp_unet.load_state_dict(weights['module'])
                ema_unet.load_state_dict({"shadow_params": [p.clone().detach() for p in list(temp_unet.parameters())]})
                del temp_unet
    else:
        initial_global_step = 0

    if args.use_ema:
        ema_unet.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    device = unet.device
    best_metric = 1000
    first_batch = True
    for epoch in range(first_epoch, config.num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            unet.train()
            with accelerator.accumulate(unet):
                # sort images
                if config.get("sort_train_rate", 0) > 0 and random.random() < config.get("sort_train_rate", 0):
                    batch = resort_batch(batch, config.nframe, batch['image'].shape[0] // config.nframe)

                # Convert images to latent space
                image = batch["image"].to(device)  # -1~1 tensor [B*F,3,H,W]
                image = einops.rearrange(image, "(b f) c h w -> b f c h w", f=config.nframe)
                if config.dynamic_nframe:
                    random_nframe = random.Random(step).randint(config.min_nframe, config.max_nframe)
                    image = image[:, :random_nframe]
                    batch['image'] = einops.rearrange(image, "b f c h w -> (b f) c h w")
                    batch['intrinsic'] = einops.rearrange(batch['intrinsic'], "(b f) c1 c2 -> b f c1 c2", f=config.nframe)[:, :random_nframe]
                    batch['intrinsic'] = einops.rearrange(batch['intrinsic'], "b f c1 c2 -> (b f) c1 c2")
                    batch['extrinsic'] = einops.rearrange(batch['extrinsic'], "(b f) c1 c2 -> b f c1 c2", f=config.nframe)[:, :random_nframe]
                    batch['extrinsic'] = einops.rearrange(batch['extrinsic'], "b f c1 c2 -> (b f) c1 c2")
                    new_tags = []
                    for tag_ in batch['tag'][::config.nframe]:
                        new_tags.extend([tag_] * random_nframe)
                    batch['tag'] = new_tags
                    if "depth" in batch:
                        batch['depth'] = einops.rearrange(batch['depth'], "(b f) c h w -> b f c h w", f=config.nframe)[:, :random_nframe]
                        batch['depth'] = einops.rearrange(batch['depth'], "b f c h w -> (b f) c h w")

                b, f, _, h, w = image.shape

                latents = slice_vae_encode(vae, image.to(weight_dtype), sub_size=16)
                latents = latents * vae.config.scaling_factor
                _, _, _, latent_h, latent_w = latents.shape

                # build masks (random visible frames), valid:0, mask:1
                random_masks = torch.ones((b, f, 1, h, w), device=device, dtype=latents.dtype)
                random_latent_masks = torch.ones((b, f, 1, latent_h, latent_w), device=device, dtype=latents.dtype)
                if config.get("fix_cond_num", None) is not None:
                    random_cond_num = config.fix_cond_num
                else:
                    random_cond_num = random.randint(1, config.max_cond_num)
                random_masks[:, :random_cond_num] = 0
                random_latent_masks[:, :random_cond_num] = 0

                if config.get("adaptive_betas", False):
                    noise_scheduler = train_noise_scheduler[config.nframe - random_cond_num]
                else:
                    noise_scheduler = train_noise_scheduler

                # build cameras
                no_camera_emb = config.model_cfg.get("no_camera_emb", False)
                if no_camera_emb:
                    camera_embedding = None
                else:
                    if random.random() < config.model_cfg.cfg_training_rate:
                        add_channel = np.sum(config.model_cfg.additional_in_channels) - 1  # 减去mask channel
                        if config.model_cfg.get("enable_depth", False) and not config.model_cfg.get("priors3d", False):  # TODO: temp code
                            ex_depth_ch = config.model_cfg.get("depth_freq", None)
                            if ex_depth_ch is None:
                                ex_depth_ch = 1
                            add_channel -= ex_depth_ch
                        if "pixel" in config.model_cfg.get("prior_type", "3dpe"):
                            add_channel -= 4
                        if "h3dpe" in config.model_cfg.get("prior_type", "3dpe"):
                            add_channel -= 48
                        camera_embedding = torch.zeros((b, f, add_channel, h, w), device=device, dtype=latents.dtype)
                    else:
                        camera_embedding = get_camera_embedding(batch["intrinsic"].to(device), batch["extrinsic"].to(device),
                                                                b, f, h, w, config=config).to(device=device, dtype=latents.dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if config.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
                if config.input_perturbation:
                    new_noise = noise + config.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if config.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # ensure valid latents through masks
                noisy_latents = latents * (1 - random_latent_masks) + noisy_latents * random_latent_masks

                # Get the text embedding for conditioning; set prompt to "" in some probability.
                encoder_hidden_states = None

                # Get the target for loss depending on the prediction type
                if config.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=config.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # concat all inputs
                inputs = noisy_latents  # [B,F,4,h,w]
                if camera_embedding is None:
                    add_inputs = random_masks
                else:
                    add_inputs = torch.cat([random_masks, camera_embedding], dim=2)  # [B,F,1+6,H,W]
                coords = None
                if config.model_cfg.get("enable_depth", False):
                    if config.model_cfg.get("priors3d", False):
                        if random.random() < config.model_cfg.cfg_training_rate:  # cfg coords
                            if config.model_cfg.get("prior_type", "3dpe") == "3dpe+pixel":
                                coords = [torch.zeros((b * f, 192 + 1, latent_h, latent_w),
                                                      device=device, dtype=latents.dtype),
                                          torch.zeros((b * f, 4, h, w), device=device, dtype=latents.dtype)]
                            elif config.model_cfg.get("prior_type", "3dpe") == "h3dpe+pixel":
                                coords = [torch.zeros((b * f, 48 + 1, h, w), device=device, dtype=latents.dtype),
                                          torch.zeros((b * f, 4, h, w), device=device, dtype=latents.dtype)]
                            elif config.model_cfg.get("prior_type", "3dpe") in ("3dpe+latent", "3dpe+warp_latent"):
                                coords = [torch.zeros((b * f, 192 + 1, latent_h, latent_w),
                                                      device=device, dtype=latents.dtype),
                                          torch.zeros((b * f, 5, latent_h, latent_w), device=device, dtype=latents.dtype)]
                            elif config.model_cfg.get("prior_type", "3dpe") == "pixel":
                                coords = torch.zeros((b * f, 4, h, w), device=device, dtype=latents.dtype)
                            else:
                                coords = torch.zeros((b * f, config.model_cfg.get("coord_dim", 192) + 1, latent_h, latent_w),
                                                     device=device, dtype=latents.dtype)
                        else:
                            coords = get_3d_priors(config, batch['depth'], batch['intrinsic'], batch['extrinsic'],
                                                   random_cond_num, nframe=f, device=device, colors=batch['image'],
                                                   latents=latents, vae=vae, prior_type=config.model_cfg.get("prior_type", "3dpe"))

                            # mask掉特定view的coords为全0 [(b f) c h w]
                            coords_cfg = config.model_cfg.get("coords_cfg", 0.0)
                            if coords_cfg > 0:
                                cmask = np.ones((b * f))
                                cmask[:int(b * f * coords_cfg)] = 0
                                np.random.shuffle(cmask)
                                cmask = cmask.reshape(b, f, 1, 1, 1)
                                cmask[:, :random_cond_num] = 1  # 确保condition不为0
                                cmask = torch.tensor(cmask, device=device, dtype=torch.float32)

                                if type(coords) == list:
                                    for j in range(len(coords)):
                                        coords[j] = einops.rearrange(coords[j], '(b f) c h w -> b f c h w', f=f)
                                        coords[j] = coords[j] * cmask
                                        coords[j] = einops.rearrange(coords[j], 'b f c h w -> (b f) c h w')
                                else:
                                    coords = einops.rearrange(coords, '(b f) c h w -> b f c h w', f=f)
                                    coords = coords * cmask
                                    coords = einops.rearrange(coords, 'b f c h w -> (b f) c h w')

                    else:
                        depth_freq = config.model_cfg.get("depth_freq", None)
                        if depth_freq is not None:
                            depth = depth_freq_encoding(batch["depth"], device=device, embed_dim=depth_freq)
                        else:
                            depth = batch["depth"].to(torch.float32)
                        depth = einops.rearrange(depth, "(b f) c h w -> b f c h w", b=bsz, f=f).to(device=device)
                        depth[:, random_cond_num:] = 0
                        add_inputs = torch.cat([add_inputs, depth], dim=2)
                        coords = None

                # get class label (domain switcher)
                domain_dict = config.model_cfg.get("domain_dict", None)
                if domain_dict is not None:
                    tags = batch["tag"][::f]
                    class_labels = [domain_dict.get(tag, domain_dict['others']) for tag in tags]
                    class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)
                else:
                    class_labels = None

                # Predict the noise residual and compute loss
                with torch.cuda.amp.autocast(enabled=True, dtype=weight_dtype):
                    model_pred = unet(inputs, timesteps, encoder_hidden_states, add_inputs=add_inputs,
                                      class_labels=class_labels, coords=coords, return_dict=False)[0]  # [BF,C,H,W]
                model_pred = einops.rearrange(model_pred, "(b f) c h w -> b f c h w", f=f)

                loss = F.mse_loss(model_pred.float()[:, random_cond_num:], target.float()[:, random_cond_num:], reduction="mean")

                # DEBUG for NaN loss
                if torch.isnan(loss).item():
                    import pickle
                    with open(f"{args.output_dir}/nan_batch_rank{accelerator.process_index}.pkl", "wb") as w:
                        pickle.dump({"batch": batch, "timesteps": timesteps, "model_pred": model_pred, "target": target}, w)
                    save_path = os.path.join(args.output_dir, "nan.ckpt")
                    torch.save(accelerator.unwrap_model(unet).state_dict(), save_path)
                    raise ValueError("NaN loss, stop training...")

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.opt_cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())

                progress_bar.update(1)
                global_step += 1

                if global_step % 20 == 0 or global_step == 1:
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                    train_loss = avg_loss.item() / config.gradient_accumulation_steps
                    accelerator.log({"train/loss": train_loss}, step=global_step)
                    accelerator.log({"train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    if args.use_ema and ema_unet.cur_decay_value is not None:
                        accelerator.log({"train/ema_decay": ema_unet.cur_decay_value}, step=global_step)
                    # logger.info(f"Loss: {train_loss}")

                if (global_step == 1 or global_step % args.train_log_interval == 0 or first_batch) and accelerator.is_main_process:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
                    log_train(accelerator=accelerator, config=config, args=args,
                              pipeline=pipeline, weight_dtype=weight_dtype, batch=batch,
                              step=global_step, random_cond_num=random_cond_num,
                              device=device, vae=vae, nframe=f)
                    del pipeline
                    torch.cuda.empty_cache()
                    gc.collect()
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())
                    first_batch = False

                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None and config.checkpoints_total_limit > 0:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if args.use_ema and accelerator.is_main_process:
                        ema_unet.save_pretrained(copy.deepcopy(accelerator.unwrap_model(unet)), f"{save_path}/ema_unet.pt")
                    logger.info(f"Saved state to {save_path}")

                if (global_step == 1 or global_step % args.val_interval == 0) and not args.no_val:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    pipeline = get_pipeline(accelerator, config, vae, unet, weight_dtype)
                    res = log_validation(accelerator=accelerator, config=config, args=args,
                                         pipeline=pipeline, val_dataloader=val_dataloader,
                                         step=global_step, device=device, vae=vae)
                    del pipeline
                    torch.cuda.empty_cache()
                    gc.collect()
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())
                    if res <= best_metric:
                        best_metric = res
                        save_path = os.path.join(args.output_dir, "best")
                        accelerator.save_state(save_path)
                        if args.use_ema and accelerator.is_main_process:
                            ema_unet.save_pretrained(copy.deepcopy(accelerator.unwrap_model(unet)), f"{save_path}/ema_unet.pt")
                        logger.info(f"Saved best state to {save_path}")

            logs = {"epoch": epoch + 1, "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
