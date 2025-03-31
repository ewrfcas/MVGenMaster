import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class ExConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=320):
        super().__init__()
        if type(in_channels) == list:
            self.in_channels = in_channels[0]
            self.ex_channels = in_channels[1]
        else:
            self.in_channels = in_channels
            self.ex_channels = None

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6),
                                   nn.SiLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6),
                                   nn.SiLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
                                   nn.SiLU())

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=64, num_channels=256, eps=1e-6),
                                   nn.SiLU())

        if self.ex_channels is not None:
            self.conv_out = zero_module(nn.Conv2d(256 + self.ex_channels, out_channels, kernel_size=3, padding=(1, 1)))
        else:
            self.conv_out = zero_module(nn.Conv2d(256, out_channels, kernel_size=3, padding=(1, 1)))

    def forward(self, x):
        if self.ex_channels is not None:
            ex = x[:, self.in_channels:]
            x = x[:, :self.in_channels]
        else:
            ex = None
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.ex_channels is not None and ex is not None:
            ex = F.interpolate(ex, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=False)
            x = self.conv_out(torch.cat([x, ex], dim=1))
        else:
            x = self.conv_out(x)

        return x


class ExConvEncoder2(nn.Module):
    def __init__(self, in_channels, out_channels=320):
        super().__init__()
        self.in_channels = in_channels[0]
        self.ex_channels = in_channels[1]

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6),
                                   nn.SiLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6),
                                   nn.SiLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
                                   nn.SiLU())

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=64, num_channels=256, eps=1e-6),
                                   nn.SiLU())

        self.conv_out = nn.Sequential(nn.Conv2d(256 + self.ex_channels, 512, kernel_size=3, padding=(1, 1)),
                                      nn.GroupNorm(num_groups=64, num_channels=512, eps=1e-6),
                                      nn.SiLU(),
                                      zero_module(nn.Conv2d(512, out_channels, kernel_size=3, padding=(1, 1))))

    def forward(self, x, ex):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv_out(torch.cat([x, ex], dim=1))

        return x

class ExConvEncoder3(nn.Module):
    def __init__(self, in_channels, out_channels=320, dropout_rate=0.15):
        super().__init__()
        self.in_channels = in_channels[0]
        self.ex_channels = in_channels[1]

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6),
                                   nn.SiLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6),
                                   nn.SiLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
                                   nn.SiLU())

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=64, num_channels=256, eps=1e-6),
                                   nn.SiLU())

        self.coord_proj = nn.Conv2d(self.ex_channels, 256, kernel_size=3, padding=(1, 1))
        self.coord_drop = nn.Dropout(p=dropout_rate)

        self.conv_out = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1)),
                                      nn.GroupNorm(num_groups=64, num_channels=512, eps=1e-6),
                                      nn.SiLU(),
                                      zero_module(nn.Conv2d(512, out_channels, kernel_size=3, padding=(1, 1))))

    def forward(self, x, ex):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        y = self.coord_proj(ex)
        y = self.coord_drop(y)
        x = self.conv_out(torch.cat([x, y], dim=1))

        return x


class ExConvEncoder4(nn.Module):
    def __init__(self, in_channels, out_channels=320):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6),
                                   nn.SiLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6),
                                   nn.SiLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
                                   nn.SiLU())

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=64, num_channels=256, eps=1e-6),
                                   nn.SiLU())

        self.conv_out = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=(1, 1)),
                                      nn.GroupNorm(num_groups=64, num_channels=512, eps=1e-6),
                                      nn.SiLU(),
                                      zero_module(nn.Conv2d(512, out_channels, kernel_size=3, padding=(1, 1))))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv_out(x)

        return x


class ExConvFlattenEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=320):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6),
                                   nn.SiLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6),
                                   nn.SiLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
                                   nn.SiLU())

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=64, num_channels=256, eps=1e-6),
                                   nn.SiLU())

        self.linear_out = zero_module(nn.Linear(1024, out_channels))

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self._pack_latents(x, x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x = self.linear_out(x)

        return x


class MultiExConvFlattenEncoder(nn.Module):
    def __init__(self, in_channels, layers=19, out_channels=3072):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6),
                                   nn.SiLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6),
                                   nn.SiLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
                                   nn.SiLU())

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 1)),
                                   nn.GroupNorm(num_groups=64, num_channels=256, eps=1e-6),
                                   nn.SiLU())

        self.out_linears = []
        for i in range(layers):
            self.out_linears.append(zero_module(nn.Linear(1024, out_channels)))
            # self.out_linears.append(nn.Sequential(nn.Linear(1024, 1024),
            #                                       nn.LayerNorm(1024),
            #                                       nn.GELU(),
            #                                       zero_module(nn.Linear(1024, out_channels))))
        self.out_linears = nn.ModuleList(self.out_linears)

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self._pack_latents(x, x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = []
        for i in range(len(self.out_linears)):
            x_ = self.out_linears[i](x)
            out.append(x_)

        return out


class CoordEncoder(nn.Module):
    def __init__(self, coord_embed, coord_layers, downsample_type="conv"):
        super().__init__()
        # layer sorting
        for c in coord_layers:
            if c['name'].startswith("down_block"):
                c['name'] = "0." + c['name']
            elif c['name'].startswith("mid_block"):
                c['name'] = "1." + c['name']
            elif c['name'].startswith("up_block"):
                c['name'] = "2." + c['name']
        coord_layers.sort(key=lambda x: x['name'])
        self.coord_layers = coord_layers

        self.layers = nn.ModuleList()
        self.gammas = nn.ParameterList()
        for c in self.coord_layers:
            print("Build coordinate PE for", c)
            self.gammas.append(nn.Parameter(torch.tensor(0.0), requires_grad=True))
            if downsample_type == "conv":
                if c['down_scale_factor'] == 2:
                    self.layers.append(nn.Conv2d(coord_embed, c['ch'], kernel_size=3, stride=2, padding=1))
                elif c['down_scale_factor'] == 4:
                    self.layers.append(nn.Sequential(nn.Conv2d(coord_embed, c['ch'] // 2, kernel_size=3, stride=2, padding=1),
                                                     nn.GroupNorm(num_groups=32, num_channels=c['ch'] // 2, eps=1e-6),
                                                     nn.SiLU(),
                                                     nn.Conv2d(c['ch'] // 2, c['ch'], kernel_size=3, stride=2, padding=1)))
                elif c['down_scale_factor'] == 8:
                    self.layers.append(nn.Sequential(nn.Conv2d(coord_embed, c['ch'] // 2, kernel_size=3, stride=2, padding=1),
                                                     nn.GroupNorm(num_groups=32, num_channels=c['ch'] // 2, eps=1e-6),
                                                     nn.SiLU(),
                                                     nn.Conv2d(c['ch'] // 2, c['ch'], kernel_size=3, stride=2, padding=1),
                                                     nn.AvgPool2d(2, stride=2)))
                else:
                    raise NotImplementedError
            elif downsample_type == "resize":
                self.layers.append(nn.Sequential(nn.Conv2d(coord_embed, c['ch'], kernel_size=1, stride=1, padding=0, bias=False),
                                                 nn.AvgPool2d(kernel_size=c['down_scale_factor'], stride=c['down_scale_factor'])))
            else:
                raise NotImplementedError

    def forward(self, x):
        # x: [bf,c,h,w]
        results = []
        for i in range(len(self.layers)):
            y = self.gammas[i] * self.layers[i](x)
            results.append(einops.rearrange(y, "b c h w -> b (h w) c"))

        return results


class CoordCrossEncoder(nn.Module):
    def __init__(self, in_channels, cross_attention_dim):
        super().__init__()
        # layer sorting
        self.down_conv = nn.Sequential(nn.Conv2d(in_channels, cross_attention_dim // 2, kernel_size=3, stride=2, padding=1),
                                       nn.GroupNorm(num_groups=32, num_channels=cross_attention_dim // 2, eps=1e-6),
                                       nn.SiLU())
        self.proj = nn.Linear(cross_attention_dim // 2, cross_attention_dim)

    def forward(self, x):
        # x: [bf,c,h,w]
        x = self.down_conv(x)
        x = einops.rearrange(x, "bf c h w -> bf (h w) c")
        x = self.proj(x)

        return x
