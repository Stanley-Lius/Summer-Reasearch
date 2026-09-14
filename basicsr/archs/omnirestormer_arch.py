import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.cgnet_arch import LayerNorm2d
from basicsr.archs.omnidir_dynamic_cgnet_arch import (
    SimpleGate, 
    FactorizedAdaptiveSpatialChannelGatedFusion, 
    FourDirAsymmetricDWConv, 
    OmniDirDynamicGCE
)

class Attention(nn.Module):
    """
    MDTA (Multi-DConv Head Transposed Self-Attention) from Restormer.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class OmniRestormerBlock(nn.Module):
    """
    OmniRestormerBlock:
    Combines Omni-Directional Dynamic Context Extraction (OmniDirDynamicGCE) 
    with Restormer's MDTA (Multi-DConv Head Transposed Self-Attention).
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_heads=1):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, groups=self.dw_channel, bias=True)

        self.sg1 = SimpleGate()
        gce_in_channels = self.dw_channel // 2

        self.GCE = OmniDirDynamicGCE(c=gce_in_channels)

        agg_channels = gce_in_channels * 4
        self.agg_conv = nn.Conv2d(agg_channels, c, kernel_size=1, bias=True)

        # Replaced SCA with MDTA for global cross-covariance attention
        self.attn = Attention(c, num_heads, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Feed-Forward Network (GDFN-style)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv_ffn_dw = nn.Conv2d(ffn_channel, ffn_channel, kernel_size=3, padding=1, groups=ffn_channel, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)  # Shape: (B, c, H, W)

        x1, x2, x3 = self.GCE(x)
        x_cat = torch.cat([x, x1, x2, x3], dim=1)  # Shape: (B, 4*c, H, W)
        x_agg = self.agg_conv(x_cat)               # Shape: (B, c, H, W)

        # Global Attention
        x_out = self.attn(x_agg)
        x_out = self.dropout1(x_out)

        y = inp + x_out * self.beta

        # Feed-Forward Network
        ffn_x = self.conv4(self.norm2(y))
        ffn_x = self.conv_ffn_dw(ffn_x)
        ffn_x = self.sg2(ffn_x)
        ffn_x = self.conv5(ffn_x)
        ffn_x = self.dropout2(ffn_x)

        return y + ffn_x * self.gamma


@ARCH_REGISTRY.register()
class OmniRestormerNet(nn.Module):
    """
    OmniRestormerNet Architecture.
    Combines the efficient directional causal shifts of OmniDirDynamicCGNet 
    with the global channel-wise attention of Restormer, achieving SOTA image restoration.
    """
    def __init__(self, img_channel=3, width=24, middle_blk_num=2,
                 enc_blk_nums=[2, 2, 2, 2], dec_blk_nums=[2, 2, 2, 2],
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0., heads=[1, 2, 4, 8]):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(*[
                    OmniRestormerBlock(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand, drop_out_rate=drop_out_rate, num_heads=heads[i])
                    for _ in range(num)
                ])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[
            OmniRestormerBlock(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand, drop_out_rate=drop_out_rate, num_heads=heads[-1])
            for _ in range(middle_blk_num)
        ])

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[
                    OmniRestormerBlock(chan, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand, drop_out_rate=drop_out_rate, num_heads=heads[3-i])
                    for _ in range(num)
                ])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

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
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x
