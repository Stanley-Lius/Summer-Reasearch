import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.cgnet_arch import LayerNorm2d, SimpleGate

class IrregularGCELayer(nn.Module):
    """
    A layer that dynamically partitions an image into irregular, non-overlapping 
    rectangular patches (like a k-d tree) based on feature variance, perfectly 
    reconstructing the image. Different patches can be 1x1, 3x3, 4x5, etc.
    """
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # 1x1 conv to extract features for variance-based splitting decision
        self.feat_conv = nn.Conv2d(c, c // 4 if c >= 4 else 1, kernel_size=1)
        
        # Learnable threshold for splitting
        import random
        self.threshold = nn.Parameter(torch.tensor(random.uniform(0.01, 0.1)))
        
        # Local convolution applied inside each irregular patch
        self.dw_conv = nn.Conv2d(c, c, kernel_size=kernel_size, padding=padding, groups=c)
        self.pw_conv = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        
        # Calculate feature map for variance check
        feat = self.feat_conv(x)
        
        # Process each image in the batch independently because they might have different partitions
        for b in range(B):
            # 1. Dynamically partition the image into irregular bounding boxes
            boxes = self._partition_single(feat[b], 0, 0, W, H)
            
            # 2. Process each patch independently and place it back
            for (x1, y1, x2, y2) in boxes:
                patch = x[b:b+1, :, y1:y2, x1:x2]
                
                # Apply local convolutions on this specific irregular patch
                processed_patch = F.gelu(self.pw_conv(self.dw_conv(patch)))
                
                # Place non-overlapping processed patch back to reconstruct the original shape
                out[b:b+1, :, y1:y2, x1:x2] = processed_patch
                
        # Modulate output using feat and threshold so gradients flow seamlessly into partitioning parameters
        attn = torch.sigmoid(feat.mean(dim=1, keepdim=True) - self.threshold)
        return out * (1.0 + attn * 0.1)

    def _partition_single(self, feat, x1, y1, x2, y2):
        """
        Recursively splits the region if its variance is higher than the threshold.
        Enforces that the minimum patch size cannot be smaller than self.kernel_size.
        """
        w = x2 - x1
        h = y2 - y1
        min_size = self.kernel_size
        
        # Base case: check if splitting along width or height would result in sub-patches smaller than kernel_size
        can_split_w = (w // 2 >= min_size)
        can_split_h = (h // 2 >= min_size)
        
        if not can_split_w and not can_split_h:
            return [(x1, y1, x2, y2)]
            
        region = feat[:, y1:y2, x1:x2]
        # Calculate variance across spatial dimensions to measure feature complexity
        var = torch.var(region, dim=(1, 2)).mean()
        
        if var > self.threshold:
            # Split along the longer axis if permitted by min_size constraint
            if w >= h:
                if can_split_w:
                    mid = x1 + w // 2
                    return self._partition_single(feat, x1, y1, mid, y2) + \
                           self._partition_single(feat, mid, y1, x2, y2)
                elif can_split_h:
                    mid = y1 + h // 2
                    return self._partition_single(feat, x1, y1, x2, mid) + \
                           self._partition_single(feat, x1, mid, x2, y2)
                else:
                    return [(x1, y1, x2, y2)]
            else:
                if can_split_h:
                    mid = y1 + h // 2
                    return self._partition_single(feat, x1, y1, x2, mid) + \
                           self._partition_single(feat, x1, mid, x2, y2)
                elif can_split_w:
                    mid = x1 + w // 2
                    return self._partition_single(feat, x1, y1, mid, y2) + \
                           self._partition_single(feat, mid, y1, x2, y2)
                else:
                    return [(x1, y1, x2, y2)]
        else:
            return [(x1, y1, x2, y2)]


class IrregularGCE(nn.Module):
    """
    Hierarchical 3-layer GCE module utilizing the IrregularGCELayer.
    """
    def __init__(self, c):
        super().__init__()
        self.layer1 = IrregularGCELayer(c, kernel_size=7)
        self.layer2 = IrregularGCELayer(c, kernel_size=5)
        self.layer3 = IrregularGCELayer(c, kernel_size=3)

    def forward(self, x):
        # Each layer partitions the image independently based on its own features and thresholds
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out1, out2, out3


class IrregularGazeBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1, groups=self.dw_channel, bias=True)

        gce_in_channels = self.dw_channel // 2
        self.GCE = IrregularGCE(c=gce_in_channels)

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
        b, c, h, w = x.shape

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
class IrregularCGNet(nn.Module):
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
                nn.Sequential(*[IrregularGazeBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[IrregularGazeBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[IrregularGazeBlock(chan) for _ in range(num)])
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
