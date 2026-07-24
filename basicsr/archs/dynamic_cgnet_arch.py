import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.cgnet_arch import NAFBlock0, LayerNorm2d, SimpleGate, UpsampleWithFlops

class DynamicGCELayer(nn.Module):
    """
    A single layer of the hierarchical Dynamic GCE module.
    Partitions input patches into smaller sub-patches via attention and applies local convolution.
    """
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # 3x3 conv non-overlapping for pattern extraction
        self.pattern_conv = nn.Conv2d(c, c, kernel_size=3, stride=3, padding=0)
        self.q_conv = nn.Conv2d(c, c, kernel_size=1)
        self.k_conv = nn.Conv2d(c, c, kernel_size=1)
        
        import random
        self.T = nn.Parameter(torch.tensor(random.uniform(0.3, 0.7)))
        
        # Depthwise Separable Conv applied inside each dynamically partitioned patch
        self.dw_conv = nn.Conv2d(c, c, kernel_size=kernel_size, padding=padding, groups=c)
        self.pw_conv = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x):
        # x shape: (B_curr, C, H, W) where B_curr is B * number_of_parent_patches
        B_curr, C, H, W = x.shape
        
        # 1. Extract pattern & determine sub-patch size S_x via attention
        if H >= 3 and W >= 3:
            pad_h = (3 - H % 3) % 3
            pad_w = (3 - W % 3) % 3
            x_pad = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else x
            
            pattern = self.pattern_conv(x_pad) # B_curr, C, Hp, Wp
            B_p, C_p, Hp, Wp = pattern.shape
            
            if Hp > 1 and Wp > 1:
                Q = self.q_conv(pattern).view(B_curr, C_p, -1)
                K = self.k_conv(pattern).view(B_curr, C_p, -1)
                A = torch.bmm(Q.transpose(1, 2), K) / (C_p ** 0.5)
                
                max_s = 1
                for b in range(B_curr):
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
            
        S_x = min(S_x, max(H, W))
        if S_x <= 0:
            S_x = 1
            
        # 2. Patchify x into non-overlapping sub-patches of size S_x by S_x
        pad_h_sx = (S_x - H % S_x) % S_x
        pad_w_sx = (S_x - W % S_x) % S_x
        x_sx_pad = F.pad(x, (0, pad_w_sx, 0, pad_h_sx)) if (pad_h_sx > 0 or pad_w_sx > 0) else x
        
        H_pad, W_pad = x_sx_pad.shape[2:]
        n_h = H_pad // S_x
        n_w = W_pad // S_x
        
        # Shape: (B_curr * n_h * n_w, C, S_x, S_x)
        x_patches = x_sx_pad.view(B_curr, C, n_h, S_x, n_w, S_x).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, S_x, S_x)
        
        # 3. Perform local convolution inside each sub-patch
        out_patches = F.gelu(self.pw_conv(self.dw_conv(x_patches)))
        
        meta = (B_curr, n_h, n_w, H_pad, W_pad, H, W)
        return out_patches, meta


class DynamicGCE(nn.Module):
    """
    Hierarchical 3-layer GCE module.
    Layer 1 receives input image, partitions into patches, and processes with large kernel (7x7).
    Layer 2 receives Layer 1's patches, recursively subdivides them, and processes with medium kernel (5x5).
    Layer 3 receives Layer 2's sub-patches, subdivides further, and processes with small kernel (3x3).
    """
    def __init__(self, c):
        super().__init__()
        self.layer1 = DynamicGCELayer(c, kernel_size=7)
        self.layer2 = DynamicGCELayer(c, kernel_size=5)
        self.layer3 = DynamicGCELayer(c, kernel_size=3)

    def forward(self, x):
        # Layer 1
        p1, meta1 = self.layer1(x)
        out1 = self.unpatchify(p1, meta1)
        
        # Layer 2 (receives p1 patches directly from Layer 1)
        p2, meta2 = self.layer2(p1)
        out2_sub = self.unpatchify(p2, meta2)
        out2 = self.unpatchify(out2_sub, meta1)
        
        # Layer 3 (receives p2 sub-patches directly from Layer 2)
        p3, meta3 = self.layer3(p2)
        out3_sub2 = self.unpatchify(p3, meta3)
        out3_sub1 = self.unpatchify(out3_sub2, meta2)
        out3 = self.unpatchify(out3_sub1, meta1)
        
        return out1, out2, out3

    def unpatchify(self, patches, meta):
        B_curr, n_h, n_w, H_pad, W_pad, H, W = meta
        C, S_x, _ = patches.shape[1:]
        out = patches.view(B_curr, n_h, n_w, C, S_x, S_x)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B_curr, C, H_pad, W_pad)
        return out[:, :, :H, :W]


class DynamicGazeBlock(nn.Module):
    """
    Mimics CascadedGazeBlock from CGNet, utilizing Range Fuser and ChannelMLP,
    but replaces the GCE module with the DynamicGCE module.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1, groups=self.dw_channel, bias=True)

        # Dynamic GCE module replaces the fixed-size GCE module
        # GCE processes the fused half of the channels (dw_channel // 2)
        gce_in_channels = self.dw_channel // 2
        self.GCE = DynamicGCE(c=gce_in_channels)

        # Output projection for concatenated branches (original + 3 upsampled branches)
        # channels: dw_channel + 3 * (dw_channel // 2) = dw_channel * 2.5
        proj_in_channels = int(self.dw_channel * 2.5)
        self.project_out = nn.Conv2d(proj_in_channels, c, kernel_size=1)
        
        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=proj_in_channels, out_channels=proj_in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )

        self.sg = SimpleGate()

        # Channel MLP
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
        b, c, h, w = x.shape

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)

        # Range fusion
        x_1, x_2 = x.chunk(2, dim=1)
        
        # Dynamic GCE replaces fixed GCE across 3 branches (preserving full resolution)
        x1, x2, x3 = self.GCE(x_1 + x_2)
        
        x = torch.cat([x, x1, x2, x3], dim=1)

        x = self.sca(x) * x
        x = self.project_out(x)
        x = self.dropout1(x)

        # Channel mixing (ChannelMLP)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


@ARCH_REGISTRY.register()
class DynamicCGNet(nn.Module):
    """
    A new architecture based on CGNet that utilizes dynamic patch sizing.
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
                nn.Sequential(*[DynamicGazeBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[DynamicGazeBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[DynamicGazeBlock(chan) for _ in range(num)])
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
