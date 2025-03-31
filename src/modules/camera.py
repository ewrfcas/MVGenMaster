import einops
import torch
import torch.nn.functional as F

from src.modules.position_encoding import freq_encoding


@torch.cuda.amp.autocast(enabled=False)
def sample_rays(intrinsic, extrinsic, image_h=None, image_w=None,
                normalize_extrinsic=False, normalize_std=False):
    ''' get rays
    Args:
        intrinsic: [B, 3, 3],
        extrinsic: [B, 4, 4],
        h, w: int
        # normalize: let the first camera R=I
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = intrinsic.device
    B = intrinsic.shape[0]
    if normalize_extrinsic:
        extrinsic = extrinsic[0:1].inverse() @ extrinsic

    c2w = torch.inverse(extrinsic)[:, :3, :4]  # [B,3,4]
    x = torch.arange(image_w, device=device).float() - 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.repeat(points, 'w h c -> b (h w) c', b=B)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic.inverse().transpose(-1, -2) * 1  # depth is 1

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)  # [B,N,3]

    rays_o = c2w[..., :3, 3]  # [B, 3]

    if normalize_std:
        rays_o = rays_o / rays_o.std(dim=0, keepdim=True)

    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [B, N, 3]

    return rays_o, rays_d


@torch.cuda.amp.autocast(enabled=False)
def batch_sample_rays(intrinsic, extrinsic, image_h=None, image_w=None,
                      normalize_extrinsic=False, normalize_t=False, nframe=1):
    ''' get rays
    Args:
        intrinsic: [BF, 3, 3],
        extrinsic: [BF, 4, 4],
        h, w: int
        # normalize: let the first camera R=I
    Returns:
        rays_o, rays_d: [BF, N, 3]
    '''

    device = intrinsic.device
    B = intrinsic.shape[0]
    if normalize_extrinsic:
        c2w_view0 = extrinsic[::nframe].inverse().to(device)  # [B,4,4]
        c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)  # [BF,4,4]
        extrinsic = c2w_view0 @ extrinsic

    c2w = torch.inverse(extrinsic)[:, :3, :4].to(device)  # [BF,3,4]
    x = torch.arange(image_w, device=device).float() - 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.repeat(points, 'w h c -> b (h w) c', b=B)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic.inverse().to(device).transpose(-1, -2) * 1  # depth is 1

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)  # [BF,N,3]
    rays_o = c2w[..., :3, 3]  # [BF, 3]

    if normalize_t:
        rays_o = einops.rearrange(rays_o, "(b f) c -> b f c", f=nframe)
        # normalize the farthest to 1.0 (direct3d)
        farthest, _ = rays_o.abs().max(dim=1, keepdim=True)
        farthest, _ = farthest.max(dim=2, keepdim=True)
        rays_o = rays_o / (farthest + 1e-8)
        rays_o = einops.rearrange(rays_o, "b f c -> (b f) c")

    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [BF, N, 3]

    return rays_o, rays_d

@torch.cuda.amp.autocast(enabled=False)
def embed_rays(rays_o, rays_d, nframe, fourier_embedding=False, fourier_embed_dim=16, camera_longest_side=None):
    if len(rays_o.shape) == 4:  # [b,f,n,3]
        rays_o = einops.rearrange(rays_o, "b f n c -> (b f) n c")
        rays_d = einops.rearrange(rays_d, "b f n c -> (b f) n c")
    cross_od = torch.cross(rays_o, rays_d, dim=-1)
    if fourier_embedding:
        fourier_pe = freq_encoding(cross_od, embed_dim=fourier_embed_dim, camera_longest_side=camera_longest_side)
        cam_emb = torch.cat([rays_d, cross_od, fourier_pe], dim=-1)
    else:
        cam_emb = torch.cat([rays_d, cross_od], dim=-1)
    cam_emb = einops.rearrange(cam_emb, "(b f) n c -> b f n c", f=nframe)
    return cam_emb


def get_camera_embedding(intrinsic, extrinsic, b, f, h, w, config):
    rays_o, rays_d = batch_sample_rays(intrinsic, extrinsic, image_h=h, image_w=w,
                                       normalize_extrinsic=config.get("normalize_extrinsic", True),
                                       normalize_t=config.get("normalize_t", True),
                                       nframe=f)
    camera_embedding = embed_rays(rays_o, rays_d, nframe=f,
                                  fourier_embedding=config.get("fourier_embedding", False),
                                  fourier_embed_dim=config.get("fourier_embed_dim", None),
                                  camera_longest_side=config.get("camera_longest_side", None))
    camera_embedding = einops.rearrange(camera_embedding, "b f (h w) c -> b f c h w", h=h, w=w)

    return camera_embedding
