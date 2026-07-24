import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.cgnet_arch import LayerNorm2d, SimpleGate


class GatedFusion(nn.Module):
    """
    Gated Fusion module that fuses 4 directional feature maps
    (Top-Left, Top-Right, Bottom-Left, Bottom-Right) using adaptive spatial gating weights.
    “Gated Fusion 給四個不同的角落計算出來的feature maps.
    """
    def __init__(self, c):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(4 * c, c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c, 4, kernel_size=1)
        )

    def forward(self, feats):
        # feats: list of 4 feature maps [tl, tr, bl, br], each of shape (B, C, H, W)
        cat_feats = torch.cat(feats, dim=1)  # (B, 4*C, H, W)
        gates = self.gate_conv(cat_feats)    # (B, 4, H, W)
        gates = F.softmax(gates, dim=1)      # Adaptive spatial gating weights

        out = (gates[:, 0:1] * feats[0] +
               gates[:, 1:2] * feats[1] +
               gates[:, 2:3] * feats[2] +
               gates[:, 3:4] * feats[3])
        return out


class FourDirDynamicGCELayer(nn.Module):
    """
    A single layer of the Four-Directional Dynamic GCE module.
    1. Splits patches from 4 directions: Top-Left, Top-Right, Bottom-Left, Bottom-Right.
    2. Uses DWConv without padding (padding=0). If H, W is not a multiple of patch size, discards leftover pixels (floor division).
    3. Fuses the 4 output feature maps via Gated Fusion.
    """
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        # 3x3 conv non-overlapping for pattern extraction
        self.pattern_conv = nn.Conv2d(c, c, kernel_size=3, stride=3, padding=0)
        self.q_conv = nn.Conv2d(c, c, kernel_size=1)
        self.k_conv = nn.Conv2d(c, c, kernel_size=1)

        self.T = nn.Parameter(torch.tensor(random.uniform(0.3, 0.7)))

        # DWConv without padding (Requirement 2)
        self.dw_conv = nn.Conv2d(c, c, kernel_size=kernel_size, padding=0, groups=c)
        self.pw_conv = nn.Conv2d(c, c, kernel_size=1)

        # Gated Fusion to fuse 4 directional outputs (Requirement 1)
        self.gated_fusion = GatedFusion(c)

    def _process_patches(self, crop, B, C, n_h, n_w, S_x, target_h, target_w):
        """Helper to patchify, apply local DWConv without padding, and unpatchify."""
        _, _, ch, cw = crop.shape
        pad_h = (S_x - ch % S_x) % S_x
        pad_w = (S_x - cw % S_x) % S_x
        if pad_h > 0 or pad_w > 0:
            crop = F.pad(crop, (0, pad_w, 0, pad_h))
            _, _, ch, cw = crop.shape
            n_h = ch // S_x
            n_w = cw // S_x

        # Shape: (B * n_h * n_w, C, S_x, S_x)
        patches = crop.view(B, C, n_h, S_x, n_w, S_x).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, S_x, S_x)

        # Local conv without padding
        out_patches = F.gelu(self.pw_conv(self.dw_conv(patches)))

        # Output patch size after unpadded DWConv
        s_out = S_x - self.kernel_size + 1
        out = out_patches.view(B, n_h, n_w, C, s_out, s_out)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, n_h * s_out, n_w * s_out)

        # Restore target resolution
        if out.shape[2] != target_h or out.shape[3] != target_w:
            out = F.interpolate(out, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return out

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Extract pattern & determine sub-patch size S_x via attention
        if H >= 3 and W >= 3:
            pad_h = (3 - H % 3) % 3
            pad_w = (3 - W % 3) % 3
            x_pad = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else x

            pattern = self.pattern_conv(x_pad)
            B_p, C_p, Hp, Wp = pattern.shape

            if Hp > 1 and Wp > 1:
                Q = self.q_conv(pattern).view(B, C_p, -1)
                K = self.k_conv(pattern).view(B, C_p, -1)
                A = torch.bmm(Q.transpose(1, 2), K) / (C_p ** 0.5)

                max_s = 1
                for b in range(B):
                    s1 = 1
                    for s in range(1, min(Hp, Wp)):
                        if A[b, 0, s * Wp + s] > self.T:
                            s1 = s + 1
                    s2 = 1
                    for s in range(1, min(Hp, Wp)):
                        if A[b, Wp - 1, s * Wp + (Wp - 1 - s)] > self.T:
                            s2 = s + 1
                    max_s = max(max_s, s1, s2)

                S_x = max_s * 3
            else:
                S_x = 3
        else:
            S_x = max(1, min(H, W))

        # Ensure S_x is at least kernel_size since DWConv has padding=0
        S_x = max(S_x, self.kernel_size)
        if H >= self.kernel_size and W >= self.kernel_size:
            S_x = min(S_x, max(H, W))

        # Floor division to discard leftover pixels (Requirement 2)
        n_h = H // S_x
        n_w = W // S_x

        # If image size is smaller than S_x due to downsampling
        if n_h == 0 or n_w == 0:
            n_h = max(1, n_h)
            n_w = max(1, n_w)
            H_eff = H
            W_eff = W
            crop_tl = x
            crop_tr = x
            crop_bl = x
            crop_br = x
        else:
            H_eff = n_h * S_x
            W_eff = n_w * S_x
            # Extract 4 directional crops discarding leftover pixels (Requirement 1 & 2)
            crop_tl = x[:, :, :H_eff, :W_eff]
            crop_tr = x[:, :, :H_eff, W - W_eff:]
            crop_bl = x[:, :, H - H_eff:, :W_eff]
            crop_br = x[:, :, H - H_eff:, W - W_eff:]

        # Process each directional crop
        out_tl = self._process_patches(crop_tl, B, C, n_h, n_w, S_x, H, W)
        out_tr = self._process_patches(crop_tr, B, C, n_h, n_w, S_x, H, W)
        out_bl = self._process_patches(crop_bl, B, C, n_h, n_w, S_x, H, W)
        out_br = self._process_patches(crop_br, B, C, n_h, n_w, S_x, H, W)

        # 3. Fuse the 4 feature maps via Gated Fusion (Requirement 1)
        fused_out = self.gated_fusion([out_tl, out_tr, out_bl, out_br])

        meta = (B, n_h, n_w, H_eff, W_eff, H, W)
        return fused_out, meta


class FourDirDynamicGCE(nn.Module):
    """
    Hierarchical 3-layer Four-Directional Dynamic GCE module.
    Requirement 3: Convolution kernel sizes arranged from small to large (3, 3, 5).
    """
    def __init__(self, c):
        super().__init__()
        self.layer1 = FourDirDynamicGCELayer(c, kernel_size=3)
        self.layer2 = FourDirDynamicGCELayer(c, kernel_size=3)
        self.layer3 = FourDirDynamicGCELayer(c, kernel_size=5)

    def forward(self, x):
        p1, _ = self.layer1(x)
        p2, _ = self.layer2(p1)
        p3, _ = self.layer3(p2)
        return p1, p2, p3


class FourDirDynamicGazeBlock(nn.Module):
    """
    Core building block incorporating FourDirDynamicGCE.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1, groups=self.dw_channel, bias=True)

        gce_in_channels = self.dw_channel // 2
        self.GCE = FourDirDynamicGCE(c=gce_in_channels)

        proj_in_channels = int(self.dw_channel * 2.5)
        self.project_out = nn.Conv2d(proj_in_channels, c, kernel_size=1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=proj_in_channels, out_channels=proj_in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)

        x_1, x_2 = x.chunk(2, dim=1)
        x1, x2, x3 = self.GCE(x_1 + x_2)

        x = torch.cat([x, x1, x2, x3], dim=1)

        x = self.sca(x) * x
        x = self.project_out(x)
        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


@ARCH_REGISTRY.register()
class FourDirDynamicCGNet(nn.Module):
    """
    Four-Directional Dynamic CGNet architecture.
    """
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[FourDirDynamicGazeBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[FourDirDynamicGazeBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[FourDirDynamicGazeBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp_padded = self.check_image_size(inp)

        x = self.intro(inp_padded)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp_padded

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
