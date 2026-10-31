import json
import math
import os
from collections import OrderedDict
from os import path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.utils.logger import get_root_logger
from basicsr.utils.softmax_util import _SOFTMAX_HOOKS


def build_2d_coords(h, w, device=None, dtype=torch.float32):
    """Build 2D pixel coordinates (H, W, 2) with (y, x) in [0, H-1] x [0, W-1]."""
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([grid_y, grid_x], dim=-1)


def get_sws_geometry(h, w, win_size, sub_win_size):
    """Compute padded feature dimensions and window/sub-window sizes for SWS attention."""
    win_h = max(h // win_size, 1)
    win_w = max(w // win_size, 1)
    sub_win_h = min(sub_win_size, win_h)
    sub_win_w = min(sub_win_size, win_w)

    win_h += (sub_win_h - win_h % sub_win_h) % sub_win_h
    win_w += (sub_win_w - win_w % sub_win_w) % sub_win_w

    mod_pad_h = (win_h - h % win_h) % win_h
    mod_pad_w = (win_w - w % win_w) % win_w
    h_pad, w_pad = h + mod_pad_h, w + mod_pad_w

    nWh, nWw = h_pad // win_h, w_pad // win_w
    nW = nWh * nWw
    sub_nWh, sub_nWw = win_h // sub_win_h, win_w // sub_win_w
    sub_nW = sub_nWh * sub_nWw
    N = sub_win_h * sub_win_w

    return {
        'h': h, 'w': w,
        'h_pad': h_pad, 'w_pad': w_pad,
        'win_h': win_h, 'win_w': win_w,
        'sub_win_h': sub_win_h, 'sub_win_w': sub_win_w,
        'nWh': nWh, 'nWw': nWw, 'nW': nW,
        'sub_nWh': sub_nWh, 'sub_nWw': sub_nWw, 'sub_nW': sub_nW,
        'N': N,
        'image_size': math.sqrt(h_pad * w_pad),
    }


def build_sws_coords(h, w, win_size, sub_win_size, pooled_q=True, device=None, dtype=torch.float32):
    """Build 2D query and key coordinates for Shuffled Window Sub-Window (SWS) attention.

    Note:
        Because all sub-window slices s in [0, sub_nW - 1] are rigid 2D translations of slice 0,
        pairwise Euclidean distances ||p_i - p_j||_2 are identical across all sub_nW slices.
        Returning slice-0 coordinates (N_Q, 2) and (N_K, 2) avoids redundant sub_nW memory.
    """
    geom = get_sws_geometry(h, w, win_size, sub_win_size)
    coords = build_2d_coords(geom['h_pad'], geom['w_pad'], device=device, dtype=dtype)

    win_coords = rearrange(
        coords, '(nWh wH) (nWw wW) d -> (nWh nWw) wH wW d',
        wH=geom['win_h'], wW=geom['win_w']
    )
    sub_win_coords = rearrange(
        win_coords, 'nW (sWh swH) (sWw swW) d -> nW (sWh sWw) (swH swW) d',
        swH=geom['sub_win_h'], swW=geom['sub_win_w']
    )
    # Take slice 0 across all nW windows: shape (nW, N, 2)
    slice0_coords = sub_win_coords[:, 0, :, :]
    k_coords = slice0_coords.reshape(geom['nW'] * geom['N'], 2)
    q_coords = slice0_coords.mean(dim=1) if pooled_q else k_coords
    return q_coords, k_coords, geom


def compute_pixel_attention_distance(attn_weights, q_coords, k_coords, image_size=None):
    """Compute per-token Mean Attention Distance (D_i) and Normalized Mean Attention Distance (D_bar).

    Formulas (Dosovitskiy et al., ICLR 2021):
        D_i = sum_{j=1}^{N_K} A_{i,j} * ||p_i - p_j||_2
        D_bar = (1 / Image_Size) * (1 / N_Q) * sum_{i=1}^{N_Q} D_i

    Args:
        attn_weights (torch.Tensor): Softmax attention weights of shape (..., num_heads, N_Q, N_K)
            satisfying sum_j A_{i,j} = 1.
        q_coords (torch.Tensor): 2D coordinates of queries, shape (N_Q, 2).
        k_coords (torch.Tensor): 2D coordinates of keys, shape (N_K, 2).
        image_size (float, optional): Normalization factor (Image Size). If None, inferred from max coordinate span.

    Returns:
        dict:
            - 'd_i': Per-query weighted spatial distance D_i, shape (..., num_heads, N_Q).
            - 'd_i_norm': Normalized per-query distance D_i / Image_Size, shape (..., num_heads, N_Q).
            - 'd_bar_per_head': Normalized Mean Attention Distance per head, shape (num_heads,).
            - 'd_bar': Scalar Normalized Mean Attention Distance averaged over all heads.
    """
    attn = attn_weights.detach().float()
    q_c = q_coords.to(device=attn.device, dtype=torch.float32)
    k_c = k_coords.to(device=attn.device, dtype=torch.float32)

    # Pairwise 2D Euclidean distance matrix ||p_i - p_j||_2: shape (N_Q, N_K)
    dist_matrix = torch.cdist(q_c, k_c, p=2)

    if image_size is None:
        span = torch.maximum(k_c.max(dim=0).values, q_c.max(dim=0).values) + 1.0
        image_size = float(torch.sqrt(span[0] * span[1]).item())

    # D_i = sum_j A_{i,j} * ||p_i - p_j||_2 -> shape (..., num_heads, N_Q)
    d_i = torch.einsum('...hqk,qk->...hq', attn, dist_matrix)
    d_i_norm = d_i / image_size

    # Average over all leading batch/window dims and query dim N_Q -> shape (num_heads,)
    flat_d_i_norm = d_i_norm.reshape(-1, d_i_norm.shape[-2], d_i_norm.shape[-1])
    d_bar_per_head = flat_d_i_norm.mean(dim=(0, 2))
    d_bar = d_bar_per_head.mean()

    return {
        'd_i': d_i,
        'd_i_norm': d_i_norm,
        'd_bar_per_head': d_bar_per_head,
        'd_bar': d_bar,
    }


def _short_layer_name(name):
    """Shorten BasicSR module path for clean plot labels (e.g. 'encoders.0.3.attn' -> 'enc0.b3')."""
    name = name.replace('module.', '')
    for suffix in ['.attn.attn', '.attn']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    parts = name.split('.')
    if len(parts) >= 3 and parts[0] == 'encoders':
        return f'enc{parts[1]}.b{parts[2]}'
    if len(parts) >= 2 and parts[0] == 'middle_blks':
        return f'mid.b{parts[1]}'
    if len(parts) >= 3 and parts[0] == 'decoders':
        return f'dec{parts[1]}.b{parts[2]}'
    return name


def _stage_color(short_name):
    """Assign color by network stage (Encoder / Middle / Decoder) for ViT scatter plot."""
    if short_name.startswith('enc'):
        return '#1f77b4'  # Blue
    if short_name.startswith('mid'):
        return '#d62728'  # Red
    if short_name.startswith('dec'):
        return '#2ca02c'  # Green
    return '#9467bd'      # Purple


def plot_attention_distance(layer_stats, save_path, title=None, ref_image_size=256):
    """Plot Normalized Mean Attention Distance across network depth (Dosovitskiy et al., 2021, Fig. 7).

    Args:
        layer_stats (list[dict]): List of per-layer dicts with keys:
            - 'name': Layer name (str)
            - 'head_distances': 1D sequence/array of per-head normalized distances D_bar^{(h)}
            - 'mean_distance': Layer mean normalized distance D_bar
        save_path (str): Output image path (.png / .pdf).
        title (str, optional): Plot title.
        ref_image_size (int): Reference input image size for secondary pixel-distance Y-axis.
    """
    if not layer_stats:
        return

    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    num_layers = len(layer_stats)
    fig_w = max(8.0, min(18.0, num_layers * 0.75 + 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5), dpi=200)

    x_indices = np.arange(num_layers)
    short_names = [_short_layer_name(item['name']) for item in layer_stats]
    layer_means = np.array([float(item['mean_distance']) for item in layer_stats])

    # Plot per-head scatter points (Dosovitskiy et al., 2021 style)
    plotted_stages = set()
    rng = np.random.default_rng(42)
    for idx, (item, s_name) in enumerate(zip(layer_stats, short_names)):
        heads = np.asarray(item['head_distances'], dtype=np.float64)
        color = _stage_color(s_name)
        stage_label = (
            'Encoder Heads' if s_name.startswith('enc')
            else 'Middle Heads' if s_name.startswith('mid')
            else 'Decoder Heads' if s_name.startswith('dec')
            else 'Attention Heads'
        )
        label = stage_label if stage_label not in plotted_stages else None
        plotted_stages.add(stage_label)

        jitter = rng.uniform(-0.12, 0.12, size=len(heads)) if len(heads) > 1 else np.zeros(1)
        ax.scatter(
            np.full_like(heads, idx, dtype=np.float64) + jitter,
            heads,
            color=color,
            alpha=0.75,
            s=36,
            edgecolors='white',
            linewidths=0.5,
            zorder=3,
            label=label,
        )

    # Plot layer-mean line
    ax.plot(
        x_indices,
        layer_means,
        color='#111111',
        linestyle='-',
        marker='D',
        markersize=5.5,
        linewidth=2.0,
        zorder=4,
        label=r'Layer Mean $\bar{D}$',
    )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=9)
    ax.set_xlabel('Network Attention Layer (Depth)', fontsize=11)
    ax.set_ylabel(r'Normalized Mean Attention Distance ($\bar{D}$)', fontsize=11)

    y_max = max(0.5, min(1.42, float( max(np.max(layer_means), max(np.max(x['head_distances']) for x in layer_stats)) * 1.15 )))
    ax.set_ylim(0.0, y_max)
    ax.grid(True, linestyle='--', alpha=0.35, zorder=0)

    # Secondary Y-axis in equivalent image pixels (matching ViT Fig. 7 right)
    if ref_image_size and ref_image_size > 0:
        sec_ax = ax.secondary_yaxis(
            'right',
            functions=(lambda d: d * ref_image_size, lambda px: px / ref_image_size),
        )
        sec_ax.set_ylabel(f'Mean Attention Distance (Pixels @ {ref_image_size}px)', fontsize=10)

    plot_title = title or 'Normalized Mean Attention Distance across Network Depth (ViT Style)'
    ax.set_title(plot_title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_spatial_attention_distance_map(spatial_maps, save_path):
    """Plot 2D spatial heatmaps of per-pixel normalized attention distance D_i / Image_Size."""
    if not spatial_maps:
        return

    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    n = len(spatial_maps)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), dpi=180, squeeze=False)

    for idx, (name, d_map) in enumerate(spatial_maps.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        im = ax.imshow(d_map, cmap='magma', vmin=0.0, vmax=max(0.5, float(np.max(d_map))))
        ax.set_title(f'{_short_layer_name(name)} (mean={np.mean(d_map):.3f})', fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    fig.suptitle(r'Per-Pixel Normalized Attention Distance Map ($D_i / \mathrm{Image\ Size}$)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


class _SoftmaxInterceptor:
    """Temporarily intercepts Softmax outputs inside a hooked attention module's forward pass."""

    def __init__(self):
        self.captured = []
        self._orig_f_softmax = None
        self._orig_t_softmax = None
        self._orig_sdpa = None
        self._hook_fn = None

    def start(self):
        self.captured.clear()
        self._orig_f_softmax = F.softmax
        self._orig_t_softmax = torch.Tensor.softmax
        self._orig_sdpa = getattr(F, 'scaled_dot_product_attention', None)

        def _record(out, dim):
            if isinstance(out, torch.Tensor) and out.ndim >= 4 and dim in (-1, out.ndim - 1):
                self.captured.append(out.detach())

        def _f_softmax(input, dim=None, _stacklevel=3, dtype=None):
            out = self._orig_f_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
            _record(out, dim)
            return out

        def _t_softmax(tensor, dim, dtype=None):
            out = self._orig_t_softmax(tensor, dim, dtype=dtype)
            _record(out, dim)
            return out

        F.softmax = _f_softmax
        torch.Tensor.softmax = _t_softmax
        self._hook_fn = _record
        _SOFTMAX_HOOKS.append(self._hook_fn)

        if self._orig_sdpa is not None:
            def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                s = (query.size(-1) ** -0.5) if scale is None else scale
                logits = torch.matmul(query, key.transpose(-2, -1)) * s
                if attn_mask is not None:
                    logits = logits + attn_mask
                probs = self._orig_f_softmax(logits, dim=-1)
                _record(probs, -1)
                return self._orig_sdpa(
                    query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
                )
            F.scaled_dot_product_attention = _sdpa

    def stop(self):
        if self._orig_f_softmax is not None:
            F.softmax = self._orig_f_softmax
        if self._orig_t_softmax is not None:
            torch.Tensor.softmax = self._orig_t_softmax
        if self._orig_sdpa is not None:
            F.scaled_dot_product_attention = self._orig_sdpa
        if self._hook_fn in _SOFTMAX_HOOKS:
            _SOFTMAX_HOOKS.remove(self._hook_fn)
        self._orig_f_softmax = None
        self._orig_t_softmax = None
        self._orig_sdpa = None
        self._hook_fn = None


class AttentionDistanceMonitor:
    """Non-intrusive hook monitor for computing & plotting Normalized Mean Attention Distance.

    Supports:
        - SWSformer / KimiSWSformer / SGNLiteNoPoolKimiSWSformer / LoopLiteNoPoolKimiSWSformer / etc.
        - Standard 2D spatial / window attention modules (e.g., SwinIR WindowAttention, ViT Attention).
    """

    def __init__(self, net, opt=None):
        self.opt = opt or {}
        self.max_images = int(self.opt.get('max_images', 16))
        self.save_plot = bool(self.opt.get('save_plot', True))
        self.save_spatial_map = bool(self.opt.get('save_spatial_map', False))
        self.save_json = bool(self.opt.get('save_json', True))
        self.ref_image_size = int(self.opt.get('ref_image_size', 256))
        self.layer_filter = self.opt.get('layer_names', None)

        self._handles = []
        self._interceptor = _SoftmaxInterceptor()
        self._dist_cache = {}
        self._stats = OrderedDict()
        self._spatial_sums = OrderedDict()
        self._active_module = None
        self._first_layer_name = None
        self._processed_images = 0

        nets = net if isinstance(net, (list, tuple)) else [net]
        for n in nets:
            if n is not None:
                self._register_hooks(n)

    @staticmethod
    def _is_target_attention_module(name, module):
        # 1. SWS / KimiSWS Vision Attention wrapper with is_full_attn=True or SWSAttention
        if hasattr(module, 'win_size') and hasattr(module, 'sub_win_size'):
            if hasattr(module, 'is_full_attn'):
                return bool(module.is_full_attn)
            return True
        # 2. SwinIR WindowAttention (has window_size and softmax)
        if hasattr(module, 'window_size') and hasattr(module, 'num_heads') and hasattr(module, 'softmax'):
            return True
        return False

    def _register_hooks(self, net):
        bare_net = net.module if hasattr(net, 'module') else net
        for name, module in bare_net.named_modules():
            if self.layer_filter and not any(k in name for k in self.layer_filter):
                continue
            if self._is_target_attention_module(name, module):
                if self._first_layer_name is None:
                    self._first_layer_name = name
                h_pre = module.register_forward_pre_hook(self._make_pre_hook(name))
                h_post = module.register_forward_hook(self._make_post_hook(name))
                self._handles.extend([h_pre, h_post])

    def _make_pre_hook(self, name):
        def _pre_hook(module, inputs):
            if self.max_images > 0 and self._processed_images >= self.max_images:
                return
            if self._active_module is None:
                self._active_module = name
                self._interceptor.start()
        return _pre_hook

    def _make_post_hook(self, name):
        def _post_hook(module, inputs, output):
            if self._active_module != name:
                return
            self._interceptor.stop()
            self._active_module = None

            if not self._interceptor.captured or not inputs or not isinstance(inputs[0], torch.Tensor):
                return

            x = inputs[0]
            attn_weights = self._interceptor.captured[-1]
            self._interceptor.captured.clear()

            if name == self._first_layer_name:
                self._processed_images += x.shape[0]
                if x.ndim == 4:
                    self.ref_image_size = int(math.sqrt(x.shape[2] * x.shape[3]))

            with torch.no_grad():
                self._accumulate_layer(name, module, x, attn_weights)
        return _post_hook

    def _get_cached_sws(self, h, w, win_size, sub_win_size, n_q, n_k, device):
        key = ('sws', h, w, win_size, sub_win_size, n_q, n_k, str(device))
        if key not in self._dist_cache:
            pooled_q = (n_q != n_k)
            q_c, k_c, geom = build_sws_coords(h, w, win_size, sub_win_size, pooled_q=pooled_q, device=device)
            if q_c.shape[0] != n_q or k_c.shape[0] != n_k:
                return None
            dist_matrix = torch.cdist(q_c, k_c, p=2)
            self._dist_cache[key] = (dist_matrix, geom)
        return self._dist_cache[key]

    def _get_cached_grid(self, n_q, n_k, image_size, device):
        key = ('grid', n_q, n_k, image_size, str(device))
        if key not in self._dist_cache:
            side = int(math.isqrt(n_k))
            if side * side != n_k or n_q != n_k:
                return None
            coords = build_2d_coords(side, side, device=device).reshape(n_k, 2)
            dist_matrix = torch.cdist(coords, coords, p=2)
            self._dist_cache[key] = (dist_matrix, float(image_size or side))
        return self._dist_cache[key]

    def _accumulate_layer(self, name, module, x, attn):
        attn = attn.detach().float()
        n_q, n_k = attn.shape[-2], attn.shape[-1]
        num_heads = attn.shape[-3]

        # Case 1: SWS / KimiSWS Attention (4D: [B*sub_nW, H, N_Q, N_K] or 5D: [B, sub_nW, H, N_Q, N_K])
        if hasattr(module, 'win_size') and hasattr(module, 'sub_win_size') and x.ndim == 4:
            B, _, H, W = x.shape
            cached = self._get_cached_sws(H, W, module.win_size, module.sub_win_size, n_q, n_k, attn.device)
            if cached is None:
                return
            dist_matrix, geom = cached
            image_size = geom['image_size']

            # Average over leading batch/sub_nW dims first to keep VRAM footprint minimal
            attn_flat = attn.reshape(-1, num_heads, n_q, n_k)
            attn_mean = attn_flat.mean(dim=0)  # (num_heads, N_Q, N_K)
            d_i_norm_head = torch.einsum('hqk,qk->hq', attn_mean, dist_matrix) / image_size
            head_vals = d_i_norm_head.mean(dim=-1).cpu().numpy()  # (num_heads,)

            if self.save_spatial_map:
                sub_nW, nW, N = geom['sub_nW'], geom['nW'], geom['N']
                attn_5d = attn.reshape(B, sub_nW, num_heads, n_q, n_k).mean(dim=(0, 2))  # (sub_nW, N_Q, N_K)
                d_sub = torch.einsum('sqk,qk->sq', attn_5d, dist_matrix) / image_size     # (sub_nW, N_Q)
                if n_q == nW:
                    d_sub = d_sub.unsqueeze(-1).expand(sub_nW, nW, N)
                else:
                    d_sub = d_sub.reshape(sub_nW, nW, N)
                d_map = d_sub.transpose(0, 1)
                d_map = rearrange(
                    d_map, 'nW (sWh sWw) (swH swW) -> nW (sWh swH) (sWw swW)',
                    sWh=geom['sub_nWh'], swH=geom['sub_win_h'], swW=geom['sub_win_w']
                )
                d_map = rearrange(
                    d_map, '(nWh nWw) wH wW -> (nWh wH) (nWw wW)',
                    nWh=geom['nWh'], wH=geom['win_h'], wW=geom['win_w']
                )[:H, :W].cpu().numpy()
                if name not in self._spatial_sums:
                    self._spatial_sums[name] = [d_map * B, B]
                else:
                    self._spatial_sums[name][0] += d_map * B
                    self._spatial_sums[name][1] += B
        else:
            # Case 2: Standard 2D Grid / Window Attention
            B = x.shape[0]
            img_sz = math.sqrt(x.shape[1]) if x.ndim == 3 else None
            cached = self._get_cached_grid(n_q, n_k, img_sz, attn.device)
            if cached is None:
                return
            dist_matrix, image_size = cached
            attn_mean = attn.reshape(-1, num_heads, n_q, n_k).mean(dim=0)
            d_i_norm_head = torch.einsum('hqk,qk->hq', attn_mean, dist_matrix) / image_size
            head_vals = d_i_norm_head.mean(dim=-1).cpu().numpy()

        if name not in self._stats:
            self._stats[name] = {'sum': head_vals * B, 'count': B}
        else:
            self._stats[name]['sum'] += head_vals * B
            self._stats[name]['count'] += B

    def get_results(self):
        """Return aggregated per-layer and per-head Normalized Mean Attention Distance statistics."""
        layer_stats = []
        for name, record in self._stats.items():
            head_dists = record['sum'] / max(record['count'], 1)
            layer_stats.append({
                'name': name,
                'short_name': _short_layer_name(name),
                'num_heads': int(len(head_dists)),
                'head_distances': [float(v) for v in head_dists],
                'mean_distance': float(np.mean(head_dists)),
                'std_distance': float(np.std(head_dists)),
            })
        return layer_stats

    def finalize_and_save(self, save_dir, tag='', current_iter=None, tb_logger=None):
        """Log Normalized Mean Attention Distance summary and save plots / JSON."""
        layer_stats = self.get_results()
        if not layer_stats:
            return None

        overall_mean = float(np.mean([item['mean_distance'] for item in layer_stats]))
        logger = get_root_logger()
        summary_parts = [f"{item['short_name']}: {item['mean_distance']:.4f}" for item in layer_stats]
        logger.info(
            f"[Attention Distance ({tag or 'eval'})] Overall D_bar: {overall_mean:.4f} | "
            + ', '.join(summary_parts)
        )

        if tb_logger is not None and isinstance(current_iter, int):
            prefix = f'attn_distance/{tag}' if tag else 'attn_distance'
            tb_logger.add_scalar(f'{prefix}/overall_D_bar', overall_mean, current_iter)
            for item in layer_stats:
                tb_logger.add_scalar(f"{prefix}/{item['short_name']}", item['mean_distance'], current_iter)

        if save_dir:
            out_dir = osp.join(save_dir, 'attn_distance')
            os.makedirs(out_dir, exist_ok=True)
            suffix = f'_{tag}_{current_iter}' if (tag and current_iter is not None) else (f'_{tag}' if tag else '')

            if self.save_plot:
                plot_path = osp.join(out_dir, f'attn_distance{suffix}.png')
                plot_attention_distance(
                    layer_stats,
                    save_path=plot_path,
                    title=f'Normalized Mean Attention Distance ({tag or "Model"})',
                    ref_image_size=self.ref_image_size,
                )
                logger.info(f'Saved ViT Mean Attention Distance plot to {plot_path}')

            if self.save_spatial_map and self._spatial_sums:
                spatial_maps = OrderedDict(
                    (k, v[0] / max(v[1], 1)) for k, v in self._spatial_sums.items()
                )
                map_path = osp.join(out_dir, f'attn_distance_spatial{suffix}.png')
                plot_spatial_attention_distance_map(spatial_maps, save_path=map_path)
                logger.info(f'Saved Spatial Attention Distance map to {map_path}')

            if self.save_json:
                json_path = osp.join(out_dir, f'attn_distance{suffix}.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        {
                            'tag': tag,
                            'iter': current_iter,
                            'overall_mean_distance': overall_mean,
                            'ref_image_size': self.ref_image_size,
                            'layers': layer_stats,
                        },
                        f,
                        indent=2,
                    )

        return {'overall_mean': overall_mean, 'layers': layer_stats}

    def close(self):
        self._interceptor.stop()
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
