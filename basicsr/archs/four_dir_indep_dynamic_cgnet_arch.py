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


class FourDirIndepDynamicGCELayer(nn.Module):
    """
    A single layer of the Four-Directional Independent Dynamic GCE module.
    1. Independently computes attention along diagonal directions from 4 corners
       (TL->BR, TR->BL, BL->TR, BR->TL) to determine 4 distinct patch sizes: S_tl, S_tr, S_bl, S_br.
    2. Slices feature maps from 4 corners according to their corresponding patch sizes, discarding leftover pixels.
    3. Processes each directional patch crop via unpadded DWConv (padding=0).
    4. Interpolates back to target resolution and fuses the 4 distinct feature maps via Gated Fusion.
    """
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        
        # 3x3 conv non-overlapping for pattern extraction
        self.pattern_conv = nn.Conv2d(c, c, kernel_size=3, stride=3, padding=0)
        self.q_conv = nn.Conv2d(c, c, kernel_size=1)
        self.k_conv = nn.Conv2d(c, c, kernel_size=1)
        
        # Independent learnable thresholds for the 4 directions
        self.T_tl = nn.Parameter(torch.tensor(random.uniform(0.3, 0.7)))
        self.T_tr = nn.Parameter(torch.tensor(random.uniform(0.3, 0.7)))
        self.T_bl = nn.Parameter(torch.tensor(random.uniform(0.3, 0.7)))
        self.T_br = nn.Parameter(torch.tensor(random.uniform(0.3, 0.7)))
        
        # DWConv without padding
        self.dw_conv = nn.Conv2d(c, c, kernel_size=kernel_size, padding=0, groups=c)
        self.pw_conv = nn.Conv2d(c, c, kernel_size=1)
        
        # Gated Fusion to fuse 4 directional outputs
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
        
        # 1. Extract pattern & independently determine sub-patch sizes S_tl, S_tr, S_bl, S_br
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
                
                max_s_tl, max_s_tr, max_s_bl, max_s_br = 1, 1, 1, 1
                limit = min(Hp, Wp)
                for b in range(B):
                    # Top-Left anchor looking towards Bottom-Right
                    s_tl = 1
                    for s in range(1, limit):
                        if A[b, 0, s * Wp + s] > self.T_tl:
                            s_tl = s + 1
                    max_s_tl = max(max_s_tl, s_tl)
                    
                    # Top-Right anchor looking towards Bottom-Left
                    s_tr = 1
                    for s in range(1, limit):
                        if A[b, Wp - 1, s * Wp + (Wp - 1 - s)] > self.T_tr:
                            s_tr = s + 1
                    max_s_tr = max(max_s_tr, s_tr)
                    
                    # Bottom-Left anchor looking towards Top-Right
                    s_bl = 1
                    for s in range(1, limit):
                        if A[b, (Hp - 1) * Wp, (Hp - 1 - s) * Wp + s] > self.T_bl:
                            s_bl = s + 1
                    max_s_bl = max(max_s_bl, s_bl)
                    
                    # Bottom-Right anchor looking towards Top-Left
                    s_br = 1
                    for s in range(1, limit):
                        if A[b, Hp * Wp - 1, (Hp - 1 - s) * Wp + (Wp - 1 - s)] > self.T_br:
                            s_br = s + 1
                    max_s_br = max(max_s_br, s_br)
                
                S_tl, S_tr, S_bl, S_br = max_s_tl * 3, max_s_tr * 3, max_s_bl * 3, max_s_br * 3
            else:
                S_tl = S_tr = S_bl = S_br = 3
        else:
            S_tl = S_tr = S_bl = S_br = max(1, min(H, W))
            
        def _bound_sx(S_x, h, w, ksize):
            S_x = max(S_x, ksize)
            if h >= ksize and w >= ksize:
                S_x = min(S_x, max(h, w))
            return S_x

        S_tl = _bound_sx(S_tl, H, W, self.kernel_size)
        S_tr = _bound_sx(S_tr, H, W, self.kernel_size)
        S_bl = _bound_sx(S_bl, H, W, self.kernel_size)
        S_br = _bound_sx(S_br, H, W, self.kernel_size)
        
        # 2. Extract 4 directional crops independently based on their respective patch sizes
        # Process Top-Left
        n_h_tl = H // S_tl
        n_w_tl = W // S_tl
        if n_h_tl == 0 or n_w_tl == 0:
            n_h_tl, n_w_tl = max(1, n_h_tl), max(1, n_w_tl)
            crop_tl = x
        else:
            crop_tl = x[:, :, :n_h_tl * S_tl, :n_w_tl * S_tl]
        out_tl = self._process_patches(crop_tl, B, C, n_h_tl, n_w_tl, S_tl, H, W)

        # Process Top-Right
        n_h_tr = H // S_tr
        n_w_tr = W // S_tr
        if n_h_tr == 0 or n_w_tr == 0:
            n_h_tr, n_w_tr = max(1, n_h_tr), max(1, n_w_tr)
            crop_tr = x
        else:
            H_eff_tr, W_eff_tr = n_h_tr * S_tr, n_w_tr * S_tr
            crop_tr = x[:, :, :H_eff_tr, W - W_eff_tr:]
        out_tr = self._process_patches(crop_tr, B, C, n_h_tr, n_w_tr, S_tr, H, W)

        # Process Bottom-Left
        n_h_bl = H // S_bl
        n_w_bl = W // S_bl
        if n_h_bl == 0 or n_w_bl == 0:
            n_h_bl, n_w_bl = max(1, n_h_bl), max(1, n_w_bl)
            crop_bl = x
        else:
            H_eff_bl, W_eff_bl = n_h_bl * S_bl, n_w_bl * S_bl
            crop_bl = x[:, :, H - H_eff_bl:, :W_eff_bl]
        out_bl = self._process_patches(crop_bl, B, C, n_h_bl, n_w_bl, S_bl, H, W)

        # Process Bottom-Right
        n_h_br = H // S_br
        n_w_br = W // S_br
        if n_h_br == 0 or n_w_br == 0:
            n_h_br, n_w_br = max(1, n_h_br), max(1, n_w_br)
            crop_br = x
        else:
            H_eff_br, W_eff_br = n_h_br * S_br, n_w_br * S_br
            crop_br = x[:, :, H - H_eff_br:, W - W_eff_br:]
        out_br = self._process_patches(crop_br, B, C, n_h_br, n_w_br, S_br, H, W)
        
        # 3. Fuse the 4 distinct feature maps via Gated Fusion
        fused_out = self.gated_fusion([out_tl, out_tr, out_bl, out_br])
        
        meta = (B, (S_tl, S_tr, S_bl, S_br), H, W)
        return fused_out, meta


class FourDirIndepDynamicGCE(nn.Module):
    """
    Hierarchical 3-layer Four-Directional Independent Dynamic GCE module.
    Convolution kernel sizes arranged from small to large (3, 3, 5).
    """
    def __init__(self, c):
        super().__init__()
        self.layer1 = FourDirIndepDynamicGCELayer(c, kernel_size=3)
        self.layer2 = FourDirIndepDynamicGCELayer(c, kernel_size=3)
        self.layer3 = FourDirIndepDynamicGCELayer(c, kernel_size=5)

    def forward(self, x):
        p1, _ = self.layer1(x)
        p2, _ = self.layer2(p1)
        p3, _ = self.layer3(p2)
        return p1, p2, p3


class FourDirIndepDynamicGazeBlock(nn.Module):
    """
    Core building block incorporating FourDirIndepDynamicGCE.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1, groups=self.dw_channel, bias=True)

        gce_in_channels = self.dw_channel // 2
        self.GCE = FourDirIndepDynamicGCE(c=gce_in_channels)

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
class FourDirIndepDynamicCGNet(nn.Module):
    """
    Four-Directional Independent Dynamic CGNet architecture.
    Each of the 4 directional crops computes its own patch size dynamically based on attention.
    """
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[FourDirIndepDynamicGazeBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[FourDirIndepDynamicGazeBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.encoders_skip_conv = nn.Conv2d(chan * 2, chan, kernel_size=1)
            self.decoders.append(
                nn.Sequential(*[FourDirIndepDynamicGazeBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

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
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
