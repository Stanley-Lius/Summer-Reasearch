# SWSFormer Optimization Session Summary

**Date:** 2026-10-04
**Project:** BasicSR - SWSFormer Variants Optimization
**Goal:** Reduce computational complexity (FLOPs/MACs/VRAM) of KimiSWSformer while preserving/improving image restoration performance (PSNR/SSIM).

## 🚀 Key Achievements & Architectural Innovations

### 1. LiteKimiSWSformer (The FLOPs Breakthrough)
We successfully reduced the KimiSWSformer complexity from **70.84 GMACs** to **15.35 GMACs** (approaching the original SWSformer's 12.39 GMACs).
*   **MLA Q-Expansion Redundancy Fix (The 64x Optimization):** 
    *   *Problem:* The original Kimi Full Attention (MLA) expanded the pooled Query ($Q$) to match the Key/Value sequence length *before* matrix multiplication. Because 64 pixels inside a sub-window shared the exact same $Q$, it resulted in computing a massive $16384 \times 16384$ attention matrix with identically repeated rows, wasting VRAM and FLOPs.
    *   *Solution:* We computed the attention using the unexpanded $Q$ ($256 \times 16384$), performed the Value multiplication, and then expanded the *output* back to 16,384 pixels. This is mathematically 100% equivalent but scales down the attention bottleneck by 64x.
*   **1D Convolution Pruning:** Removed redundant `ShortConvolution` layers on $Q, K, V$, relying entirely on the existing `PEG` module for local positional features.
*   **Bidirectional Channel Split:** Sliced the channels ($C/2$) for the forward and backward sweeps inside the linear attention (KDA) instead of duplicating the full channels, halving the state-computation FLOPs.

### 2. MS-KimiSWSformer (Multi-Scale Kimi)
*   Introduced a dual-branch architecture combining large and small receptive fields.
*   **Split:** Channels are split symmetrically ($C/2$). Branch A uses `sub_win_size=8`, Branch B uses `sub_win_size=4`.
*   **Fusion:** Concatenated outputs are fused dynamically using a `SpatialGating` mechanism (Sigmoid Gating + 1x1 Convolution).

### 3. MSSWSformer (Multi-Scale Standard SWSformer)
*   *Concept:* Bringing the Multi-Scale branch mechanism back to the highly stable and hardware-friendly original `SWSformer`.
*   **Architecture:** Uses standard `SWSAttention`. No heavy linear KDA gating.
*   **Fusion Strategy:** Employs a **Simple 1x1 Conv** fusion for maximum efficiency, stripping away the Sigmoid gating to keep parameter count low.
*   **Implementation:** Developed in `basicsr/archs/ms_swsformer_arch.py` with matched Train/Test YAML configurations.

## 🐛 Bug Fixes & Refactoring
*   **NaN Loss Divergence Fixed:** The original KimiSWSformer diverged around iteration 100 due to garbage memory allocation for `dt_bias`. Replaced `torch.empty` with `torch.zeros`.
*   **Code Complexity Audit (Ponytail Review):** Stripped 115 lines of redundant code, including custom LR schedulers reinventing the PyTorch standard library and obsolete transform hooks.
*   **Automated Evaluation:** Wrote `/tmp/batch_test_30.py` to evaluate the last 30 checkpoints concurrently in the background, extracting both **PSNR** and **SSIM** while managing process deadlocks during heavy VRAM utilization.

## 📁 Generated Artifacts & Files
*   `basicsr/archs/lite_kimi_swsformer_arch.py`
*   `basicsr/archs/ms_kimi_swsformer_arch.py`
*   `basicsr/archs/ms_swsformer_arch.py`
*   YAML Configurations mapped in `options/train/SIDD/` and `options/test/SIDD/`.
*   Testing Results logged to `30_test_results.txt` at the repository root.

## 🔮 Next Steps
*   Train the `MSSWSformer` and `LiteKimiSWSformer` variants.
*   Compare evaluation metrics (PSNR/SSIM) vs FLOPs against the baseline SWSformer.
*   Monitor GPU memory and inference speeds on the RTX 5060 (16GB).
