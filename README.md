# MVGenMaster

[CVPR2025] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model

[[arXiv]](https://arxiv.org/abs/2411.16157) [[Project Page]](https://ewrfcas.github.io/MVGenMaster/)

### Abstract

We introduce **MVGenMaster**, a multi-view diffusion model enhanced with 3D priors to address versatile Novel View Synthesis (NVS) tasks. MVGenMaster leverages 3D priors that are warped using metric depth and camera poses, significantly enhancing both generalization and 3D consistency in NVS.
Our model features a simple yet effective pipeline that can generate up to 100 novel views conditioned on variable reference views and camera poses with a single forward process.
Additionally, we have developed a comprehensive large-scale multi-view image dataset called **MvD-1M**, comprising up to 1.6 million scenes, equipped with well-aligned metric depth to train MVGenMaster.
Moreover, we present several training and model modifications to strengthen the model with scaled-up datasets.
Extensive evaluations across in- and out-of-domain benchmarks demonstrate the effectiveness of our proposed method and data formulation.
Models and codes will be released soon.

## News
- **2025-06-03**: We release partial aligned depth of MvD-1M.
- **2025-05-30**: Our new work, [Uni3C](https://ewrfcas.github.io/Uni3C/), is also released.

### TODO List
- [x] Environment setup
- [x] Training codes
- [x] Inference models and codes
- [x] Releasing partial MvD-1M

## Preparation

### Setup repository and environment
```
git clone https://github.com/ewrfcas/MVGenMaster.git
cd MVGenMaster

conda create -n mvgenmaster python=3.10 -y
conda activate mvgenmaster

apt-get update && apt-get install ffmpeg libsm6 libxext6 libglm-dev -y
pip install -r requirements.txt
```

### Weights
1. [Our model](https://huggingface.co/ewrfcas/MVGenMaster/resolve/main/check_points/pretrained_model.zip) for inference (put it to `./check_points/`).
2. [Dust3R](https://huggingface.co/ewrfcas/MVGenMaster/resolve/main/check_points/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) for inference with multi-view conditions (put it to `./check_points/`).
3. [Depth-Pro](https://huggingface.co/ewrfcas/MVGenMaster/resolve/main/check_points/depth_pro.pt) for inference with single-view condition (put it to `./check_points/`).
4. [data](https://huggingface.co/ewrfcas/MVGenMaster/resolve/main/data.zip) (optional) for training indices (put it to `./`).

### MvD-1M aligned depth

We release some aligned coefficients of MvD-1M in [Link](https://huggingface.co/datasets/ewrfcas/MVGenMaster/tree/main/monocular_depthanythingv2_scaleshift) (ACID, DL3DV, Co3Dv2, Real10k, MVImgNet).

For each `json`, `scale` and `shift` is saved as:
```
{'scale': 1.3232, 'shift': 0.0124}
```
Some cases are failed with `scale==0.0` and `shift==0.0`.
You should load the inverse depth (disp) inferenced by [DepthAnythingV2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth) (depth_anything_v2_vitl) and then apply the following codes to achieve aligned metric depth.
Note that all coefficients are aligned based on the officially given extrinsic cameras.
```
disp = model_depthanythingV2_large.forward(image)
disp = np.clip(disp * scale + shift, 1e-4, 1e4)
depth = np.clip(1 / disp, 0, 1e4)
```

## Inference
```
# single-view condition
CUDA_VISIBLE_DEVICES=0 python run_mvgen.py --input_path demo/flux-0.png --output_path outputs/demo_flux0 --nframe=28 --val_cfg=2.0 --cam_traj bi_direction --d_phi 160.0

# multi-view conditions
CUDA_VISIBLE_DEVICES=0 python run_mvgen.py --input_path demo/test1 --output_path outputs/demo_test1 --nframe=28 --val_cfg=2.0
```


## Training
Training script for stage1 (up to 350k steps). Here we use 4 GPUs for an example, while our model is trained under 16 GPUs.
```
accelerate launch --mixed_precision="fp16" \
                  --num_processes=4 \
                  --num_machines 1 \
                  --main_process_port 29443 \
                  --config_file configs/deepspeed/acc_zero2.yaml train.py \
                  --config_file="configs/mvgenmaster_train_stage1.yaml" \
                  --output_dir="check_points/mvgenmaster_stage1" \
                  --train_log_interval=250 \
                  --val_interval=2000 \
                  --val_cfg=2.0 \
                  --use_ema \
                  --ema_decay=0.9995 \
                  --min_decay=0.5 \
                  --ema_decay_step=30
```

Training script for stage2 (up to 600k steps):
```
accelerate launch --mixed_precision="fp16" \
                  --num_processes=4 \
                  --num_machines 1 \
                  --main_process_port 29443 \
                  --config_file configs/deepspeed/acc_zero2.yaml train.py \
                  --config_file="configs/mvgenmaster_train_stage2.yaml" \
                  --resume_path="check_points/mvgenmaster_stage1" \
                  --resume_from_checkpoint="latest" \
                  --output_dir="check_points/mvgenmaster_stage2" \
                  --train_log_interval=250 \
                  --val_interval=2000 \
                  --val_cfg=2.0 \
                  --use_ema \
                  --ema_decay=0.9995 \
                  --min_decay=0.5 \
                  --ema_decay_step=30 \
                  --restart_global_step=350000
```
NOTE: We additionally use "objaverse" "front3d" "megascenes" "aerial" "streetview" for training, while some of them suffer from license issue and are failed to be released. We provide the config files of them for the reference.

## Cite
If you found our project helpful, please consider citing:

```
@article{cao2024mvgenmaster,
  title={MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model},
  author={Cao, Chenjie and Yu, Chaohui and Liu, Shang and Wang, Fan and Xue, Xiangyang and Fu, Yanwei},
  journal={arXiv preprint arXiv:2411.16157},
  year={2024}
}
```
