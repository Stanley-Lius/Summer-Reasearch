# CGNet 官方訓練設定 vs 自研架構訓練設定 — 差異分析

> **比較基準**
> - 官方：`CascadedGaze-SIDD-with60`（`width: 60`，8 GPU，SIDD）
> - 自研：`options/train/CGnet/` 下三個 SIDD 設定檔
>   - `train_VectorizedDynamicCGNet_SIDD.yml`（最新，主要比較對象）
>   - `train_DynamicCGNet_SIDD.yml`（`OmniDirDynamicCGNet`）
>   - `train_IrregularCGNet_SIDD.yml`
>
> **驗證方式**：所有「有效／無效」判定皆以本 repo 實際程式碼為準
> （`basicsr/models/cgnet_model.py`、`basicsr/models/sr_model.py`、
> `basicsr/data/paired_image_dataset.py`、`basicsr/models/lr_scheduler.py`、`basicsr/losses/`）
>
> **影響程度**：極高 / 高 / 中 / 低 / 無
> **緊急程度**：P0（立即）/ P1（本週）/ P2（有空）/ P3（可忽略）

---

## ⚠️ 版本註記：v2 修正（重要）

本文件 v1 的 dB 估計**過度樂觀**，v2 已修正。三項具體錯誤：

| # | v1 的錯誤 | v2 修正 |
|:--|:--|:--|
| 1 | §11 堆疊估計寫 **+0.9 ~ +2.3 dB**。但 NAFNet 在 SIDD 上 `width 32 → 39.97 dB`、`width 64 → 40.30 dB`，**兩個訓練良好的模型之間全部差距僅約 0.33 dB**。v1 的總和超過整個已知空間數倍，**算術上不可能成立** | 病因是把各項估計「當獨立效果相加」；實際上 batch / patch / iteration 都在推同一件事（逼近容量上限），高度重疊且會飽和。§11 已改為天花板推算 |
| 2 | §B1 寫「等效資料曝光量只有 1/64」，暗示模型**資料飢餓** | SIDD medium 僅 320 對影像 ≈ **30k crops**。batch 4 × 400k iter ≈ **52 epoch**（每個 crop 看過 52 遍），官方 ≈ 850 epoch。從 52 → 850 是深度收斂區的打磨，**不是資料不夠**。原框架方向錯誤 |
| 3 | §B1 寫「梯度方差約為官方 16 倍」，並把梯度累積列為「最大單項增益」 | 損失是**逐像素平均**：batch 4 × 128² = **65,536 個像素級樣本**參與每次梯度估計，梯度信噪比遠優於分類任務。線性外推不成立。**batch size 對本任務不是畫質旋鈕**，梯度累積已從 P2 首位**降級為性價比最差項** |

**同時修正的方法論問題**：v1 對一個**沒有量測值的基線**給了一整欄 dB 估計（因為 `val.metrics` 缺失，你的 PSNR 目前無法測量）。所有 dB 數字皆為**文獻外推**，非本專案實測。修好 §E1 後應以自己的曲線取代全部估計。

**未受影響的結論**：`val.metrics` 缺失（P0）、`lr`/`grad_clip` 設定衝突（P1，非規模問題）、mixup 歸因問題（P1）、`gt_size` 對上下文消融的影響（P1，此項理由與 PSNR 無關）。

> **已核實**：`vectorized_dynamic_cgnet_arch.py:194-195` 使用 `LayerNorm2d`，**全網無 BatchNorm** → batch size 對正規化統計**毫無影響**，這進一步降低 batch size 的重要性。

---

## 目錄

0. [⚠️ v2 修正說明](#️-版本註記v2-修正重要)
1. [總覽表（按性價比排序）](#1-總覽表按性價比排序)
2. [A 類：實作缺口與相容性](#2-a-類實作缺口與相容性)
3. [B 類：訓練規模與資料吞吐](#3-b-類訓練規模與資料吞吐)
4. [C 類：優化器與學習率](#4-c-類優化器與學習率)
5. [D 類：資料增強策略](#5-d-類資料增強策略)
6. [E 類：驗證與記錄](#6-e-類驗證與記錄)
7. [F 類：路徑與雜項](#7-f-類路徑與雜項)
8. [無效鍵清單](#8-無效鍵清單寫了但程式不讀)
9. [行動清單](#9-行動清單)
10. [建議設定檔（可直接套用）](#10-建議設定檔可直接套用)
11. [預期成效總結](#11-預期成效總結)

---

## 1. 總覽表（按性價比排序）

**排序原則**：以「性價比」而非「絕對規模差距」排序 —— 即 (預期改善 ÷ 投入成本) × 理由紮實度。

| # | 差異項目 | 官方 | 我的設定 | 影響 | 緊急 | 修正後預期 | 一句話結論 |
|:--|:--|:--|:--|:--:|:--:|:--:|:--|
| **E1** | `val.metrics` | psnr + ssim | **完全缺少** | **極高** | **P0** | 0 dB | 驗證不算任何指標，等於盲跑；**沒有它以下全部無法驗證** |
| **F1** | `resume_state`（Dynamic） | `~` | **Linux 絕對路徑** | **中** | **P0** | 可執行性 | 在本機直接 FileNotFoundError |
| **C1+C2** | `lr` × `grad_clip_norm` | 1e-3 @ batch 64 | **1e-3 @ batch 4 + clip 0.01** | **高** | **P1** | **+0.05~0.2 dB** | **設定衝突，非規模問題**；clip 幾乎每步觸發，抹平整條 cosine 曲線。改 2 個數字、零算力 |
| **D1** | `mixing_augs.mixup` | **不使用** | **開啟** | **中** | **P1** | ~0 dB | 理由不是 dB 而是**歸因**：同時改架構又加 mixup，好壞都無法歸因 |
| **B2** | `gt_size` | 256 | **128** | **高** | **P1** | **+0.1~0.25 dB** | 深層特徵圖僅 8×8，`context_max_radius: 4` 已覆蓋整張圖 → **上下文消融測不到你想測的東西**（此理由與 PSNR 無關） |
| **B3a** | Irregular `width` | 60 | **16** | **高** | P2 | 可比性 | 僅其他兩變體 1/4 容量 → **三變體無法橫向比較** |
| **A1** | `pixel_opt.type` | `PSNRLoss` | **`L1Loss`** | 中 | P2 | +0.05~0.15 dB<br>*(信心低)* | 沒有直接優化評測指標。**此項是我證據最弱的估計** |
| **E2** | `save_checkpoint_freq` | 2000 | **20000**（Irregular 5000） | 中 | P2 | +0.02~0.05 dB | 挑最佳權重粒度粗 10 倍；且需先修 E1 才會產生 best ckpt |
| **A4** | 兩階段微調 | 280k 後開增強＋降 LR | **未實作** | 中 | P2 | +0.05~0.1 dB | 分兩次跑即可，無需寫程式 |
| **B3b** | `width`（最終數字） | **60** | 32 | **高** | P2 | **+0.2~0.35 dB** | **證據最紮實**（NAFNet 32→64 實測 +0.33 dB），但需大量算力 |
| **B1** | 有效 batch size | 8×8 = **64** | 1×4 = **4** | 中低 | **P3**<br>↓降級 | +0.05~0.2 dB | **逐像素損失下 batch 非畫質旋鈕**（每 step 已有 65k 像素樣本）；無 BatchNorm；需寫程式＋重訓 → **性價比最差** |
| **D2** | 幾何增強時機 | 前 280k 關閉 | **全程開啟** | 中 | P3 | 方向不明 | **真正的 trade-off**：你資料曝光低，全程增強可能更適合 |
| **C3** | `eta_min`（Irregular） | 1e-7 | 1e-6 | 低 | P3 | ~0 dB | 順手統一，消除無意義變異 |
| **A3** | flip 鍵名 | `use_flip` | `use_hflip` | 低 | P3 | — | **你的才是對的**，官方那份在本 repo 會 KeyError |
| **B4** | `num_worker_per_gpu` | 8 | 4 / 6 | 低 | P3 | 0 dB | batch 4 非 I/O bound |
| **E3** | `val_freq` | 2000 | 1000 | 低 | P3 | 0 dB | 驗證更頻繁，訓練略慢 |
| **A2** | scheduler | `TrueCosineAnnealingLR` | `CosineAnnealingRestartLR` | **無** | P3 | 0 dB | 單週期無重啟，**數學上等價** |
| **A5** | `model_type` | `ImageRestorationModel` | `CGNetModel` | **無** | P3 | 0 dB | 本 repo 中前者是後者的空別名 |
| **E4** | `print_freq` | 200 | 100 | **無** | P3 | 0 dB | 純日誌 |
| **—** | 上下文超參 | `GCE_CONVS_nums` | `context_num_scales/max_radius` | — | — | — | 機制不同，非強弱問題 |
| **—** | `enc/middle/dec_blk_nums` | [2,2,4,6]/10/[2,2,2,2] | **完全相同** | **無** | — | — | 深度對齊，可比性良好 |
| **—** | `optim_g` 參數本身 | AdamW/wd 0/betas[0.9,0.9] | **完全相同** | **無** | — | — | 一致（問題在 lr 未隨 batch 縮放，見 C1） |

> **所有 dB 數字皆為文獻外推，非本專案實測**（因 `val.metrics` 缺失，目前無任何量測值）。修好 E1 後請以自己的曲線取代。

---

## 2. A 類：實作缺口與相容性

### A1. 損失函數：`PSNRLoss` → `L1Loss`

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方用 `PSNRLoss`（log 域 MSE，`-10·log10(MSE)`）；本 repo `basicsr/losses/` **沒有** `PSNRLoss`，只有 `L1Loss` |
| **優點** | `L1Loss` 對離群像素較穩健、梯度尺度穩定、感知上稍平滑；不需改動程式碼 |
| **缺點** | **沒有直接優化評測指標**。`PSNRLoss` 會對「已接近乾淨」的樣本給更大梯度，後期收斂更積極；NAFNet / CascadedGaze 在 SIDD 上都用它 |
| **影響程度** | 中 |
| **緊急程度** | P2（v1 誤列為 P1） |
| **建議設定** | 在 `basicsr/losses/basic_loss.py` 新增並註冊 `PSNRLoss`，然後改為 `type: PSNRLoss` |
| **預期改善** | **+0.05 ~ +0.15 dB** ⚠️ **本文件中信心最低的估計**（v1 寫 +0.1~0.2 dB） |
| **建議** | 約 20 行程式碼。但要誠實說明：**我沒有找到「同架構、同配方、只換 L1 ↔ PSNRLoss」的乾淨消融數據**。NAFNet / CascadedGaze 選用 PSNRLoss 是事實，但那不等於「換掉就掉 0.1~0.2 dB」—— 兩者在 log 域與線性域的梯度形狀差異在後期才明顯，實際增益可能更小。**建議把它當「與官方對齊」的理由，而非「換取 dB」的理由**。實作時注意是對 batch 平均後的 `mse` 取 log，而非逐張取 log 再平均 |

### A2. 排程器：`TrueCosineAnnealingLR` → `CosineAnnealingRestartLR`

| 項目 | 內容 |
|:--|:--|
| **差異** | 本 repo `lr_scheduler.py` 只有 `MultiStepRestartLR`、`CosineAnnealingRestartLR`。你用 `periods: [400000]`、`restart_weights: [1]` 取代 |
| **優點** | **數學上完全等價** —— 單一週期、無重啟，就是同一條 cosine 曲線 |
| **缺點** | 無 |
| **影響程度** | 無 |
| **緊急程度** | P3 |
| **建議設定** | 維持現狀 |
| **預期改善** | 0 dB |
| **建議** | 這是正確的等價替換，不需處理。唯一要注意的是 `periods` 總和必須等於 `total_iter`（目前 400000 = 400000 ✓） |

### A3. 增強鍵名：`use_flip` vs `use_hflip`

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方寫 `use_flip: false`；`paired_image_dataset.py:84` 實際讀的是 `self.opt['use_hflip']`（**直接索引，非 `.get()`**） |
| **優點** | **你的寫法才是對的**。官方那份 yaml 直接放進本 repo 會在第一個 training step 就 `KeyError: 'use_hflip'` |
| **缺點** | 無 |
| **影響程度** | 低（只在照抄官方 yaml 時才會踩到） |
| **緊急程度** | P3 |
| **建議設定** | 維持 `use_hflip` |
| **預期改善** | 0 dB（但避免一次 crash） |
| **建議** | 不需處理。記錄下來，避免日後照抄官方設定時被雷到 |

### A4. 兩階段微調策略未實作 ⚠️

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方 `finetune_augment: true` / `finetune_start_iteration: 280000` / `finetune_start_iteration_lr: 0.00002` —— 前 280k 不做幾何增強快速擬合，後 120k 開增強並把 LR 壓到 2e-5 收尾。**本 repo 的 `ImageRestorationModel`（`cgnet_model.py:117`）只是 `CGNetModel` 的空別名（`pass`），完全沒讀這三個鍵 → 靜默失效** |
| **優點** | 不實作也能跑；你改用「全程開增強」是一個合理的替代策略 |
| **缺點** | 放棄了官方刻意設計的收尾技巧。「先無增強快速收斂 → 最後用增強＋極低 LR 微調」通常比全程固定策略多拿一小截分數 |
| **影響程度** | 中 |
| **緊急程度** | P2 |
| **建議設定** | 若要實作，在 `CGNetModel` 加 iteration hook 切換 dataset 的增強 flag 並改寫 optimizer LR；或**簡易替代**：先跑 280k 無增強，手動 resume 後改 yaml 開增強＋降 LR 續跑 120k |
| **預期改善** | **+0.05 ~ +0.10 dB** |
| **建議** | 優先度低於 A1/E1/C1。**簡易替代方案（分兩次跑）成本極低**，不需寫程式，只要在 280k 時停下來改 yaml 再 resume。建議等基礎設定修好、確認 baseline 穩定後再做 |

### A5. `model_type`：`ImageRestorationModel` vs `CGNetModel`

| 項目 | 內容 |
|:--|:--|
| **差異** | 名稱不同，但 `cgnet_model.py:117` 中 `ImageRestorationModel(CGNetModel): pass` |
| **優點/缺點** | 完全等價，無差異 |
| **影響程度** | 無 |
| **緊急程度** | P3 |
| **建議設定** | 維持 `CGNetModel`（語義更清楚） |
| **預期改善** | 0 dB |
| **建議** | 不需處理。但要意識到：**這個空別名是 A4 和 E1 失效的根因** —— 官方 repo 的 `ImageRestorationModel` 有 `grids`、TTA、`finetune_augment`、`use_image` 等功能，你這裡一個都沒有 |

---

## 3. B 類：訓練規模與資料吞吐

### B1. 有效 batch size：64 → 4 — **v2 已大幅降級**

> **v1 把這條列為「最大的性能差距來源」，這是錯的。** 以下是修正後的分析。

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方 `num_gpu: 8` × `batch_size_per_gpu: 8` = **64**；你 `num_gpu: 1` × `4` = **4**（Irregular 為 `1×8 = 8`） |
| **~~每 step 像素量差 64 倍~~** | ~~官方 4.19 M vs 你 65.5 K~~ → **此框架誤導**，見下方「為何降級」 |
| **優點** | 單卡跑得動、VRAM 需求低、每 step 極快；**梯度雜訊本身有正則化效果**，小 batch 有時反而略好 |
| **缺點** | batch 4 確實在偏低的一端（主流配置：EDSR 16、SwinIR 32、Restormer 64、HAT 32），對 **loss 曲線平滑度**的影響比對最終 PSNR 明顯 |
| **影響程度** | **中低**（v1 誤判為極高） |
| **緊急程度** | **P3**（v1 誤判為 P1） |
| **建議設定** | 若真要處理：`batch_size_per_gpu: 8` 或梯度累積到有效 batch 16~24（「夠穩就好」），**不必追到 64** |
| **預期改善** | **+0.05 ~ +0.20 dB**（v1 寫 +0.3~0.8 dB，**過度樂觀**） |
| **建議** | **不要優先做這個。** 需要寫程式（約 15 行梯度累積）＋完整重訓，換來的是整份清單裡最小的確定性增益 —— **性價比最差**。若手上多出算力，優先給 `gt_size`（B2）或延長訓練，不要全砸在 batch 上 |

#### 為何從「極高」降級到「中低」——三個理由

**理由 1：不是「1/64 資料量」，是 52 epoch vs 850 epoch**

SIDD medium 訓練集僅 **320 對影像**，標準預處理切成約 **30k crops**：

| 設定 | 每 step 樣本 | 400k iter 總樣本 | **等效 epoch** |
|:--|:--:|:--:|:--:|
| 官方 batch 64 | 64 | 25.6 M | **≈ 850** |
| 你 batch 4 | 4 | 1.6 M | **≈ 52** |

v1 寫「你看過的像素只等於官方約 6k iter」，暗示模型**資料飢餓**。但 52 epoch 意味著**每個 crop 已經看過 52 遍**。從 52 → 850 epoch 是深度收斂區的打磨，邊際收益很小。**SIDD 是小資料集，「資料不夠」這個框架整個是錯的方向。**

**理由 2：逐像素損失的梯度估計早就夠準了**

v1 寫「梯度方差約為官方 16 倍」—— 這個線性外推**對密集預測任務不成立**。你的損失是逐像素平均的：

```
batch 4 × 128 × 128 = 65,536 個像素級樣本 / 每次梯度估計
```

這跟分類任務裡「batch 4 = 4 個樣本」完全不是一回事。低階視覺的梯度信噪比本來就遠優於分類，**batch 增加的是「梯度估計準確度」，而這一項早已飽和**。這正是 batch size 在本任務中不是畫質旋鈕的根本原因。

**理由 3：本網路無 BatchNorm（已核實）**

`vectorized_dynamic_cgnet_arch.py:194-195` 使用 `LayerNorm2d`，全網 **無任何 BatchNorm**。若有 BN，batch 4 的統計量會極不穩定，升到 64 可能實質改善 0.1 dB 以上 —— 但 LayerNorm 對 batch size **完全無感**。這條路徑不存在。

#### 正確的歸因方式：固定 iteration vs 固定樣本數

| 比較方式 | 4 → 32 的觀測結果 | 說明 |
|:--|:--|:--|
| **固定 iteration**（你的情況） | +0.05 ~ +0.2 dB | 總樣本數變 8 倍，**這是「訓練量」的功勞，不是 batch size 的** |
| **固定總樣本數 / epoch**（公平比較） | **±0.05 dB，在雜訊內** | 有時小 batch 還略好（梯度雜訊的正則化效果） |
| **沒調 LR**（大 batch + 原 LR） | **大概率變差** | 等效步長變小，同 iteration 下收斂不足 |

第三列正是 C1 的鏡像問題：**batch 和 LR 必須一起動**。若你把 batch 從 4 加到 32 卻不調 LR，很可能白花算力。要調的話 linear scaling 是 ×8、sqrt scaling 是 ×2.8；低階視覺實務上偏保守，常取 **×2~3 並加 500~1000 iteration warmup**。

### B2. `gt_size`：256 → 128

| 項目 | 內容 |
|:--|:--|
| **差異** | 訓練 patch 從 256×256 縮到 128×128 |
| **深層特徵圖尺寸** | 官方 256/16 = **16×16**；你 128/16 = **8×8** |
| **優點** | 每 step 計算量降 4 倍、VRAM 降約 4 倍，是單卡的必要妥協 |
| **缺點** | **對你的研究主題傷害最大。** CGNet 系列的核心賣點就是 GCE / dynamic-context 的大感受野。你的 `context_max_radius: 4` 在 8×8 特徵圖上**已覆蓋整張圖** → 「大範圍上下文建模」這個能力**既訓練不出來，也驗證不到**。等於在弱化你正想證明的東西 |
| **影響程度** | **中高**（對一般 PSNR）／**極高**（對「證明架構有效」這個目的） |
| **緊急程度** | P1 |
| **建議設定** | 若 VRAM 允許，**優先把 `gt_size` 提回 192 或 256**，寧可降 `batch_size_per_gpu` 換取。`gt_size: 192` + `batch_size_per_gpu: 4` 是不錯的折衷（深層特徵圖 12×12） |
| **預期改善** | `gt_size: 256`：**+0.1 ~ +0.25 dB**，且**上下文模組的消融結論才有意義**（這比 dB 數字更重要） |
| **建議** | **在「加大 batch」和「加大 patch」之間，你的研究應該優先選 patch。** 因為 batch 小只是讓分數低一點，patch 小會讓你的架構貢獻**完全看不出來** |

#### 為何 patch size 不隨 B1 一起降級

B1 降級了，但 B2 **維持高優先**，理由與 PSNR 無關，因此不受 batch size 那套論證影響：

**理由 1：顯存固定時，patch 比 batch 划算（業界共識）**

顯存固定下 `batch 64@128` 與 `batch 16@256` 成本相當。兩者買到的東西不同：

| 加大什麼 | 買到什麼 | 是否已飽和 |
|:--|:--|:--:|
| **patch size** | **模型能看到的上下文範圍** | ❌ 遠未飽和 |
| batch size | 梯度估計的準確度 | ✅ 早就夠準（見 B1 理由 2） |

一般公認的優先順序大致是：

```
patch size  >  總訓練 iteration  >  EMA / 學習率排程  >  batch size
```

**理由 2：這是你研究主題的命門，不只是分數問題**

深層特徵圖尺寸 = `gt_size / 16`：

| `gt_size` | 深層特徵圖 | `context_max_radius: 4` 的覆蓋狀況 |
|:--:|:--:|:--|
| 128（現況） | **8×8** | **半徑 4 已覆蓋整張圖 → 動態上下文機制無作用空間** |
| 192 | 12×12 | 開始有區分度 |
| 256（官方） | 16×16 | 機制能被正常訓練與評估 |

**你正在弱化你想證明的東西。** 這條理由與「batch size 是不是畫質旋鈕」的爭論**完全正交** —— 就算 patch size 對 PSNR 只值 0.1 dB，它對「消融實驗能不能測到你的貢獻」是決定性的。

**理由 3：這可能是一個論文論點，不只是一個修正**

建議把「`gt_size` 128 vs 256 對照實驗」排在前面。如果你的架構在 256 下相對 baseline 的優勢**明顯放大**，那本身就是很好的論述（「本方法的增益隨輸入尺度提升而放大」），比追一個絕對 PSNR 數字更有價值 —— 而且這條路對單卡預算友善得多。

### B3. `width`：60 → 32 / 16

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方 60；Vectorized/Dynamic 為 32；Irregular 為 16 |
| **參數量比例（≈ width²）** | 32 → **≈ 0.28×**；16 → **≈ 0.07×** |
| **優點** | 訓練快、單卡跑得動、迭代實驗週期短，適合架構探索期 |
| **缺點** | 容量上限明顯較低，PSNR 天花板必然低一截。`width: 16` 只夠當「這個 idea 有沒有訊號」的探針，**不適合拿來跟官方數字對比** |
| **影響程度** | **高** |
| **緊急程度** | P2 |
| **建議設定** | 探索期維持 32；**確定架構有效後，做一次 `width: 60` 的完整訓練**作為最終數字。Irregular 的 16 建議至少提到 32，否則與另兩個變體不可橫向比較 |
| **預期改善** | `width` 32→60：**+0.2 ~ +0.35 dB** |
| **建議** | **深度（blk_nums）你三個檔都與官方完全一致，這點做得很好**，消融的可比性良好。但 Irregular 的 `width: 16` 破壞了三變體間的橫向可比性 —— 目前無法判斷它表現差是因為架構還是因為容量只有其他兩者的 1/4。**建議統一為 32** |

#### 這是本文件證據最紮實的一項

NAFNet 在 SIDD 上的公開數字（**同一套訓練配方，唯一變數是寬度**）：

| 模型 | SIDD PSNR |
|:--|:--:|
| NAFNet-width32 | **≈ 39.97 dB** |
| NAFNet-width64 | **≈ 40.30 dB** |
| **差距（2× 寬度）** | **≈ +0.33 dB** |

這個數據點有兩個用途：

1. **它讓 `width 32 → 60` 的估計有實測依據** —— 60 略小於 64，故取 **+0.2 ~ +0.35 dB**。
2. **它是本文件所有 dB 估計的天花板校驗器。** 兩個「訓練良好」的模型、寬度差一倍，總差距只有 0.33 dB。任何聲稱「調整訓練設定可得 +1 dB 以上」的估計，都必須先解釋為何能超過這個架構層級的差距 —— v1 的 +0.9~2.3 dB 過不了這個檢驗，這正是它被推翻的方式。

### B4. `num_worker_per_gpu`：8 → 4 / 6

| 項目 | 內容 |
|:--|:--|
| **優點** | 降低本機記憶體與行程開銷，Windows 上更穩定 |
| **缺點** | 理論吞吐略降 |
| **影響程度** | 低 |
| **緊急程度** | P3 |
| **建議設定** | 維持 4（Windows 上 `num_workers` 開太高容易出問題） |
| **預期改善** | 0 dB（僅影響 wall-clock） |
| **建議** | 不需處理。batch 4 + lmdb 幾乎不是 I/O bound |

---

## 4. C 類：優化器與學習率

### C1. `lr: 1e-3` 未隨 batch size 縮放 ⚠️ 隱性問題

| 項目 | 內容 |
|:--|:--|
| **差異** | `optim_g` 兩邊**完全相同**（AdamW, lr 1e-3, wd 0, betas [0.9, 0.9]）—— 但官方的 1e-3 是為 **batch 64** 調的，你在 **batch 4** 沿用 |
| **優點** | 與官方設定「看起來一致」，省去調參 |
| **缺點** | batch 從 64 降到 4 是 **1/16**。依 **linear scaling** LR 應降到 `6.25e-5`；依 **sqrt scaling** 應降到 `2.5e-4`。低階視覺實務上偏保守，通常介於兩者之間。**沿用 1e-3 等於超調 4~16 倍** → 這正是你必須開激進 grad clip 才跑得動的根因 |
| **影響程度** | **高** |
| **緊急程度** | P1 |
| **建議設定** | **方案一（推薦，≈ sqrt scaling）**：`lr: 2e-4` + `grad_clip_norm: 1.0`<br>**方案二（保守，介於兩法則之間）**：`lr: 1e-4` + `grad_clip_norm: 0.5`<br>**方案三（配合 B1）**：若做梯度累積到有效 batch 16~32，則 `lr: 4e-4` + `grad_clip_norm: 1.0` + **加 500~1000 iter warmup**（`warmup_iter: 1000`）—— 大 batch 不加 warmup 很可能白花算力 |
| **預期改善** | **+0.1 ~ +0.3 dB**，且訓練曲線明顯更乾淨 |
| **建議** | **C1 和 C2 必須一起改，不能只改一個。** 目前「lr 1e-3 + clip 0.01」是一組互相抵銷的設定：LR 開很大，再用極緊的 clip 硬壓回來。改成「lr 2e-4 + clip 1.0」是同一件事的乾淨版本，且 LR 排程才會真正發揮作用 |

### C2. `grad_clip_norm: 0.01` ⚠️ 讓 LR 排程實質失效

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方 SIDD yaml **未設定**（本 repo `cgnet_model.py:57-58` 預設 `use_grad_clip=True`、`grad_clip_norm=0.01`）；你三個檔都明確寫 `0.01` |
| **機制** | 這是對**全模型梯度總範數**的裁剪（非 per-layer）。以 width 32、含 10 個 middle block 的網路，健康訓練時總範數遠大於 0.01 → **幾乎每個 step 都被裁剪** |
| **優點** | 在「batch 4 + lr 1e-3」的組合下確實防止 loss 爆掉，讓訓練跑得完 |
| **缺點** | 實質上把「AdamW + lr 1e-3 + cosine 退火」變成「幾乎固定步長的正規化更新」。**LR 排程的作用被大幅削弱，cosine 的精細收尾幾乎無效** → 末期榨不出最後那 0.1 dB |
| **影響程度** | **高** |
| **緊急程度** | P1 |
| **建議設定** | `grad_clip_norm: 1.0`（搭配 `lr: 2e-4`）或 `0.5`（搭配 `lr: 1e-4`） |
| **預期改善** | 與 C1 合計 **+0.1 ~ +0.3 dB** |
| **建議** | **改動成本最低、確定性最高的一項** —— 只改兩個數字，不用寫程式。建議**先做這個**，跑 20k iter 看 loss 曲線是否更平滑、驗證 PSNR 是否上升更快，再決定要不要投入 B1/B2 的大工程 |

### C3. `eta_min`：1e-7 vs 1e-6（僅 Irregular）

| 項目 | 內容 |
|:--|:--|
| **差異** | Irregular 用 `1e-6`，官方與另兩個檔用 `1e-7` |
| **優點** | 幾乎沒有 |
| **缺點** | 末期 LR 高 10 倍，最後收斂階段抖動略大 |
| **影響程度** | 低 |
| **緊急程度** | P2 |
| **建議設定** | 改為 `1e-7`，與其他兩檔統一 |
| **預期改善** | **+0.00 ~ +0.02 dB**（實務上幾乎不可測） |
| **建議** | 改動成本為零，順手統一即可。主要價值是**消除三變體間的無意義變異**，讓消融更乾淨 |

---

## 5. D 類：資料增強策略

### D1. `mixing_augs.mixup: true` ⚠️ 建議關閉

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方 SIDD **不使用 mixup**；你三個檔都開了（`mixup: true`, `mixup_beta: 1.2`, `use_identity: true`） |
| **機制** | `cgnet_model.py:17-33` 把兩組 (noisy, clean) 配對做線性混合。`use_identity: true` 讓約 50% 的 step 跳過混合 |
| **優點** | 額外正則化，在你這種「小 batch + 低資料曝光」的情境下理論上有助於抗過擬合 |
| **缺點** | **SIDD 是真實相機雜訊（訊號相關）。線性混合出來的「雜訊」不再服從真實感測器雜訊分佈。** Restormer 在**合成高斯**去噪上用 mixup 有效，但 NAFNet / CascadedGaze 的 SIDD 設定**刻意不用**。更關鍵的是：它引入的變異會**混淆「架構改動有沒有效」這個結論** |
| **影響程度** | 中（對 PSNR）／**高**（對消融結論的可信度） |
| **緊急程度** | P1 |
| **建議設定** | `mixup: false`（或整段 `mixing_augs` 移除） |
| **預期改善** | **+0.0 ~ +0.1 dB**，但**主要價值是讓消融結論可信** |
| **建議** | **做架構比較時務必關掉。** 你現在同時改了架構、又加了官方沒有的 mixup —— 如果結果變好，你無法區分是架構的功勞還是 mixup 的功勞；如果變差，也無法歸因。**先關掉建立乾淨 baseline，之後想加再單獨做一次 mixup 消融** |

### D2. 幾何增強時機：兩階段 vs 全程開啟

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方前 280k 關閉、後 120k 開啟；你 `use_hflip: true` / `use_rot: true` **全程開啟** |
| **優點** | 正則化更強，在低資料曝光量的情境下較不易過擬合；設定簡單，不需分段 |
| **缺點** | 早期收斂較慢；且放棄了官方「先無增強快速擬合 → 最後用增強＋極低 LR 微調」的收尾增益 |
| **影響程度** | 中 |
| **緊急程度** | P2 |
| **建議設定** | **短期維持全程開啟**（在你的低曝光情境下這個選擇其實有道理）。若要複製官方策略，見 A4 的「分兩次跑」方案 |
| **預期改善** | 改為兩階段：**+0.05 ~ +0.10 dB** |
| **建議** | 這一項是**真正的 trade-off，不是單純的錯誤**。你的 batch 小、資料曝光低，全程增強的正則化其實可能比官方策略更適合你的情境。建議**先不動**，等其他項目修好後再做 A/B 對照 |

---

## 6. E 類：驗證與記錄

### E1. `val.metrics` 完全缺少 ⚠️⚠️ 最緊急

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方有 `psnr` + `ssim`；**你三個 SIDD train yaml 全部沒有 `metrics` 區塊** |
| **機制** | `sr_model.py:186` 是 `with_metrics = self.opt['val'].get('metrics') is not None`。**沒有這個區塊 → 驗證迴圈只跑 forward，不計算、不記錄、不寫 tensorboard 任何 PSNR/SSIM** |
| **優點** | 驗證稍快（省下 metric 計算時間） |
| **缺點** | **你現在是完全盲跑，只能看 training loss。** 無法知道驗證 PSNR、無法判斷有沒有過擬合、無法挑最佳 checkpoint、無法比較三個架構變體。這使**目前所有訓練的產出都無法用來下任何結論** |
| **影響程度** | **極高** |
| **緊急程度** | **P0 — 立即修** |
| **建議設定** | 三個檔的 `val:` 區塊補上（見下方 §10） |
| **預期改善** | 0 dB（不改變訓練），但**讓實驗從「不可用」變成「可用」** |
| **建議** | **這是整份清單裡最該馬上做的一件事，改動成本 = 貼上 10 行 YAML。** 有趣的是你的 **test yaml 三個檔都有正確的 metrics 區塊** —— 顯然是 train yaml 漏了。修完後建議重跑（或至少 resume）一次，才有數字可看 |

### E2. `save_checkpoint_freq`：2000 → 20000 / 5000

| 項目 | 內容 |
|:--|:--|
| **差異** | 官方 2000；Vectorized/Dynamic 為 `2e4`；Irregular 為 `5e3` |
| **優點** | 大幅節省磁碟（width 32 的 checkpoint 每個約數十 MB） |
| **缺點** | ①**挑最佳權重的粒度粗 10 倍** —— 真正的最佳點可能落在兩個存檔之間；②斷點恢復最多損失 2 萬 iter；③與 test yaml 用 `net_g_latest.pth` 的習慣疊加，很可能拿到非最佳權重去測 |
| **影響程度** | 中 |
| **緊急程度** | P2 |
| **建議設定** | `save_checkpoint_freq: !!float 5e3`，三檔統一 |
| **預期改善** | **+0.02 ~ +0.05 dB**（純粹來自「挑到更好的權重」，非訓練改善） |
| **建議** | 5000 是磁碟與粒度的好折衷。**搭配 E1 修好後，記得改用 `net_g_best_*.pth` 而非 `net_g_latest.pth` 做測試** —— 沒有 metrics 就不會產生 best checkpoint，所以這兩件事要一起做 |

### E3. `val_freq`：2000 → 1000

| 項目 | 內容 |
|:--|:--|
| **優點** | 驗證曲線解析度更高，更早發現問題 |
| **缺點** | 訓練 wall-clock 略增（修好 E1 之後這個成本才會真正出現） |
| **影響程度** | 低 |
| **緊急程度** | P3 |
| **建議設定** | 維持 `1e3`；若修好 E1 後發現驗證太耗時，再放寬到 `2e3` |
| **預期改善** | 0 dB |
| **建議** | 不需處理 |

### E4. `print_freq`：200 → 100

| 項目 | 內容 |
|:--|:--|
| **影響程度** | 無 |
| **緊急程度** | P3 |
| **建議** | 不需處理，純日誌密度偏好 |

---

## 7. F 類：路徑與雜項

### F1. `resume_state` 指向 Linux 絕對路徑 ⚠️

| 項目 | 內容 |
|:--|:--|
| **差異** | `train_DynamicCGNet_SIDD.yml:56`：<br>`resume_state: /media/stanley/AA5E237A5E233F07/BasicSR/experiments/train_OmniDirDynamicCGNet_SIDD_multiFeaturemaps/training_states/140000.state` |
| **缺點** | 這是舊 Linux 機器的絕對路徑，**在目前的 Windows 環境下會直接 FileNotFoundError**，訓練起不來 |
| **影響程度** | 中（單一檔案完全無法啟動） |
| **緊急程度** | **P0** |
| **建議設定** | 改為 `~`，或改成相對路徑 `experiments/<name>/training_states/140000.state`（且確認該檔存在） |
| **預期改善** | 從「無法執行」變成「可執行」 |
| **建議** | 順手修掉。同時注意：這個檔案的 `name:` 是 `train_OmniDirDynamicCGNet_SIDD_multiFeaturemaps`，而**對應的 test yaml 指向 `..._doublegate_...` 目錄** —— 兩者實驗名不一致，建議一併釐清 |

### F2. 其他確認無差異的項目

| 項目 | 說明 |
|:--|:--|
| `manual_seed` 10 vs 100 | 無影響 |
| `scale: 1` | 相同 |
| `dataroot_gt` / `dataroot_lq` / `io_backend: lmdb` | **完全相同**，訓練與驗證資料來源一致 —— 這是可比性的基礎，做對了 |
| `filename_tmpl: '{}'` | 相同 |
| `dataset_enlarge_ratio: 1` / `prefetch_mode: ~` | 相同 |
| `dist_params` (nccl/29500) | 單卡用不到 |
| `enc/middle/dec_blk_nums` | **三檔與官方完全一致**（[2,2,4,6] / 10 / [2,2,2,2]），深度對齊，消融可比性良好 |
| `optim_g` 全部參數 | **完全一致** |
| 上下文超參 | 官方 `GCE_CONVS_nums: [3,3,2,2]`（逐階段遞減）vs 你的 `context_num_scales: 4` + `context_max_radius: 4`（各階段固定）。**這是刻意的機制替換，不是缺項**。唯一可議之處：官方隨深度遞減，你固定 4 尺度，在深層 8×8 特徵圖上多出的尺度是冗餘計算 |

---

## 8. 無效鍵清單（寫了但程式不讀）

以下鍵在本 repo 中**完全沒有被讀取**，不影響訓練，但會誤導你以為有做某些處理：

| 鍵 | 出現位置 | 你以為它在做什麼 | 實況 |
|:--|:--|:--|:--|
| `window_size: 8` | 你三個 train yaml 的 `val:` | 把輸入 pad 成 8 的倍數 | `sr_model.py` 不讀。目前僥倖沒事，因 SIDD crops 是 256×256（16 整除）。**換非 16 倍數的資料集會炸** |
| `rgb2bgr: true` | 你三個 train yaml | 通道轉換 | 不讀。`tensor2img` 內建預設就輸出 BGR |
| `use_image: false` | 你三個 train yaml | 指標算在 float tensor 上 | 不讀。`sr_model.py:210` **永遠**先 `tensor2img()` 再算指標 → 你的 PSNR 一律算在 **8-bit 量化影像**上，官方 `use_image: false` 是算在 **float tensor** 上，**系統性偏低約 0.01~0.03 dB** |
| `max_minibatch: 4` | Gaussian yaml | 驗證時分批 | 不讀 |
| `dist: false` | 你三個 train yaml | 關閉分散式 | `parse_options` 會依 launcher 覆寫，寫了也是白寫（無害） |
| `finetune_augment` 等三鍵 | 官方 yaml | 兩階段微調 | 見 A4，本 repo 未實作 |
| `grids` | 官方 test yaml | 大圖切塊推論 | `grep -rn "grids" basicsr/models/` **零命中** |

> **`use_image` 的實質後果**：你的 PSNR 與論文表格**定義不同**（量化 vs 未量化），嚴格說不可直接對比。好消息是三個變體一致，**內部橫向比較完全成立**。

---

## 9. 行動清單

### 第一批 — P0（今天做完，總計約 15 分鐘）

| 動作 | 檔案 | 成本 | 效果 |
|:--|:--|:--:|:--|
| ✅ 補上 `val.metrics`（psnr + ssim） | 三個 train yaml | 貼 10 行 | **讓實驗從不可用變可用** |
| ✅ 修掉 Linux 絕對路徑 `resume_state` → `~` | `train_DynamicCGNet_SIDD.yml:56` | 改 1 行 | 讓該檔能啟動 |
| ✅ 釐清 Dynamic 的實驗名 `doublegate` 不一致問題 | train + test yaml | 查證 | 避免張冠李戴 |

### 第二批 — P1（本週做完，先跑 20k iter 驗證方向）

| 動作 | 設定 | 成本 | 修正後預期 |
|:--|:--|:--:|:--|
| ✅ 修正 LR / clip 組合 | `lr: 2e-4` + `grad_clip_norm: 1.0` | 改 2 個數字 | **+0.05 ~ +0.2 dB** |
| ✅ 關閉 mixup | `mixup: false` | 改 1 行 | ~0 dB，但**消融可信度大增** |
| ⬜ 提高 `gt_size: 128 → 192` 或 `256` | 視 VRAM 調 batch | 改 1~2 行 + 重訓 | **+0.1 ~ +0.25 dB**，且**上下文消融才有意義** |

> **先跑 20k iter 就能看出方向**：修好 E1 後，比較「舊設定」與「LR/clip 修正版」的驗證曲線。若後者 loss 更平滑、PSNR 爬升更快，再決定要不要投入 patch/width 的重訓成本。

### 第三批 — P2（baseline 穩定後再投入）

| 動作 | 成本 | 修正後預期 |
|:--|:--:|:--|
| ⬜ 統一 Irregular 的 `width: 16 → 32` | 重訓 | 讓三變體可橫向比較（**非 dB，是可比性**） |
| ⬜ 實作並改用 `PSNRLoss` | ~20 行程式 | +0.05 ~ +0.15 dB（**信心低**，見 §A1） |
| ⬜ `save_checkpoint_freq → 5e3` + 改用 `net_g_best_*.pth` | 改 2 行 | +0.02 ~ +0.05 dB |
| ⬜ 兩階段微調（分兩次跑，280k 時改 yaml resume） | 無需寫程式 | +0.05 ~ +0.10 dB |
| ⬜ 最終數字用 `width: 60` 重訓一次 | 大量算力 | **+0.2 ~ +0.35 dB**（證據最紮實） |

### 第四批 — P3（低優先，含 v2 降級項）

| 動作 | 成本 | 修正後預期 | 為何降級 |
|:--|:--:|:--|:--|
| ⬜ **梯度累積（有效 batch → 16~24）** | ~15 行程式 **+ 完整重訓** | +0.05 ~ +0.2 dB | **v1 誤列為「最大單項增益」。逐像素損失下梯度估計早已飽和、無 BatchNorm、且 52 epoch 非資料飢餓 → 性價比最差**（見 §B1） |
| ⬜ 統一 Irregular 的 `eta_min → 1e-7` | 改 1 行 | ~0 dB | 順手做，消除無意義變異 |
| ⬜ 兩階段幾何增強（vs 全程開啟） | 分兩次跑 | 方向不明 | **真正的 trade-off**，你資料曝光低，全程增強可能更適合 |

### 不需處理

`use_hflip` 鍵名（你的是對的）、`CosineAnnealingRestartLR`（數學等價）、`model_type`（空別名）、`num_worker_per_gpu`、`val_freq`、`print_freq`、無效鍵（可選擇移除以免日後誤導）

---

## 10. 建議設定檔（可直接套用）

以下是 `train_VectorizedDynamicCGNet_SIDD.yml` 的建議修改版，**只含 P0 + P1 中不需寫程式的部分**（標 `# ← 改動` 者為變更點）：

```yaml
# general settings
name: train_VectorizedDynamicCGNet_SIDD
model_type: CGNetModel
scale: 1
num_gpu: 1
dist: false
manual_seed: 100

datasets:
  train:
    name: SIDD_Train
    type: PairedImageDataset
    dataroot_gt: ./datasets/SIDD/train/gt_crops.lmdb
    dataroot_lq: ./datasets/SIDD/train/input_crops.lmdb
    filename_tmpl: '{}'
    io_backend:
      type: lmdb

    gt_size: 192            # ← 改動：128 → 192（VRAM 不足則退回 128）
    use_hflip: true
    use_rot: true

    use_shuffle: true
    num_worker_per_gpu: 4
    batch_size_per_gpu: 4
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: SIDD_Val
    type: PairedImageDataset
    dataroot_gt: ./datasets/SIDD/val/gt_crops.lmdb
    dataroot_lq: ./datasets/SIDD/val/input_crops.lmdb
    io_backend:
      type: lmdb

network_g:
  type: VectorizedDynamicCGNet
  img_channel: 3
  width: 32
  middle_blk_num: 10
  enc_blk_nums: [2, 2, 4, 6]
  dec_blk_nums: [2, 2, 2, 2]
  context_num_scales: 4
  context_max_radius: 4

path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

train:
  total_iter: 400000
  warmup_iter: -1
  use_grad_clip: true
  grad_clip_norm: 1.0      # ← 改動：0.01 → 1.0（配合下方 lr）

  scheduler:
    type: CosineAnnealingRestartLR
    periods: [400000]
    restart_weights: [1]
    eta_min: !!float 1e-7

  mixing_augs:
    mixup: false           # ← 改動：true → false（建立乾淨 baseline）
    mixup_beta: 1.2
    use_identity: true

  optim_g:
    type: AdamW
    lr: !!float 2e-4       # ← 改動：1e-3 → 2e-4（依 batch 4 縮放）
    weight_decay: 0
    betas: [0.9, 0.9]

  pixel_opt:
    type: L1Loss           # 註：可選擇實作 PSNRLoss 與官方對齊（P2，增益信心低，見 §A1）
    loss_weight: 1
    reduction: mean

# validation settings
val:
  val_freq: !!float 1e3
  save_img: false
  use_image: false         # 註：本 repo 不讀此鍵，指標一律算在 uint8 上
  metrics:                 # ← 改動：新增整段（原本完全缺少）
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false

logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3   # ← 改動：2e4 → 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~
```

> 同樣的改動套用到 `train_DynamicCGNet_SIDD.yml`（另外把 `resume_state` 改為 `~`）
> 與 `train_IrregularCGNet_SIDD.yml`（另外把 `width: 16 → 32`、`eta_min: 1e-6 → 1e-7`）。

---

## 11. 預期成效總結

### ⚠️ v1 的堆疊表已刪除（不可能成立）

v1 在此處列了一張「各批次累積效果」表，最終聲稱 **+0.9 ~ +2.3 dB**。**該表已刪除**，因為它過不了天花板校驗：

```
NAFNet-width32 (SIDD) ≈ 39.97 dB
NAFNet-width64 (SIDD) ≈ 40.30 dB
─────────────────────────────────
兩個「訓練良好」模型的全部差距 ≈ 0.33 dB
                                （唯一變數是 2 倍寬度）
```

**若「調訓練設定」就能拿 +0.9~2.3 dB，那將遠超過「架構寬度加倍」帶來的全部收益 —— 這不合理。**

錯誤成因：把各項估計**當成獨立效果相加**。實際上 batch / patch / iteration / LR 全都在推同一件事 —— **讓模型逼近它容量所允許的收斂點**。一旦接近那個上限，再加什麼都吃不到。這些增益高度重疊且會飽和，不能相加。

### 修正後的天花板推算

| 階段 | 內容 | 合理期待 | 依據 |
|:--|:--|:--:|:--|
| 現狀 | 盲跑，**PSNR 無量測值** | **未知** | 因 `val.metrics` 缺失 |
| **P0** | 有 PSNR/SSIM 可看 | **0 dB** | 不改變訓練，但**讓實驗從不可用變可用** |
| **P0 + P1 + P2**<br>（`width` 固定 32） | 把設定從「目前」修到「調好」 | **+0.2 ~ +0.5 dB**<br>*(信心不高)* | 各項增益重疊飽和，非相加 |
| **+ `width: 60`** | 上述 + 架構容量提升 | **再 +0.2 ~ +0.35 dB** | NAFNet 32→64 實測 +0.33 dB |

> **請把上表當「量級參考」而非承諾。** 真正的答案要等你補上 `val.metrics` 後看自己的曲線 —— 這也正是為什麼 E1 是唯一的 P0。

### 四句話總結

1. **目前這三個設定檔是「單卡可行性設定」，不是「性能複現設定」。** 但差距的來源要歸因正確：**主要是 `width` 32 vs 60（架構容量）和評測定義不同（uint8 vs float）**，而**不是** v1 強調的 batch size —— 逐像素損失下 batch 4 每步已有 65k 像素樣本，梯度估計早已飽和，且全網無 BatchNorm。

2. **架構之間的相對比較仍然成立 —— 只要設定彼此一致。** 這點你做得不錯：三個檔的 `blk_nums`、`optim_g`、scheduler、資料來源都對齊了。**唯二的破口是 Irregular 的 `width: 16`（其他兩者的 1/4 容量）和 `eta_min: 1e-6`**，建議統一。

3. **最該立刻做的兩件事，成本都不到半小時**：補上 `val.metrics`（否則你連自己的實驗結果都看不到，本文件所有 dB 估計也都只是文獻外推），以及把 `lr: 1e-3 + clip 0.01` 改成 `lr: 2e-4 + clip 1.0`。後者**不是規模問題而是設定衝突** —— LR 開很大再用極緊的 clip 硬壓回來，等於把整條 cosine 退火曲線抹平。

4. **不要把多出來的算力全砸在 batch size 上。** 優先順序是 `patch size > 總訓練 iteration > LR 排程 / EMA > batch size`。若真要加 batch，記得 LR 跟著調（sqrt scaling ×2~3）並加 500~1000 iter warmup，否則很可能白花。

### 給研究方向的一點建議

你的 `gt_size: 128` 讓深層特徵圖只有 8×8，而 `context_max_radius: 4` 在 8×8 上已覆蓋整張圖 —— **這意味著你正在弱化你想證明的東西**（大範圍動態上下文建模）。

建議把「`gt_size` 128 vs 256 的對照實驗」排在較前面。如果你的架構在 256 下相對 baseline 的優勢明顯放大，**這本身就是一個很好的論文論點**（「本方法的增益隨輸入尺度提升而放大」），比單純追一個絕對 PSNR 數字更有價值 —— 而且這條路對單卡預算友善得多。
