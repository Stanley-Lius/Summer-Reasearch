# BasicSR 專案轉移與環境部署指南

這份指南說明瞭如何將現有的 BasicSR 專案（包含訓練好的權重與設定）推送到 GitHub，並在另一台新電腦上完整重建環境並接續訓練。

## 1. 在原電腦上推送程式碼 (Push to GitHub)

在將程式碼推送至 GitHub 前，請確認已經忽略了大型檔案與資料集，以避免超過 GitHub 的容量限制。

*   `.gitignore` 中已經設定忽略了 `datasets/`, `experiments/`, `results/`, `tb_logger/`, `wandb/`, `tmp/` 以及 `SIDD/` 等大型資料夾。
*   確認無誤後，即可執行以下指令推送到 GitHub：

```bash
git add .
git commit -m "Update BasicSR and prepare for migration"
git push origin master
```

---

## 2. 在新電腦上部署環境

為了確保依賴套件不會與系統環境衝突，**強烈建議（且必須）使用 Conda 來建立獨立的虛擬環境 (Virtual Environment)**。

### 步驟 2.1: 準備 Conda 虛擬環境
如果您還沒有安裝 Conda，請先安裝 Anaconda 或 Miniconda。
打開終端機，建立並啟動一個新的 conda 環境（建議使用 Python 3.9 或 3.10）：

```bash
# 建立名為 basicsr 的 conda 虛擬環境
conda create -n basicsr python=3.9 -y

# 啟動虛擬環境
conda activate basicsr
```

### 步驟 2.2: 下載專案程式碼
將您推送到 GitHub 上的專案 Clone 下來，並進入專案目錄：

```bash
git clone <您的_github_repo_網址>
cd BasicSR
```

### 步驟 2.3: 安裝 PyTorch
在安裝 BasicSR 之前，請先根據新電腦的顯示卡型號（CUDA 版本），到 [PyTorch 官網](https://pytorch.org/) 查詢對應的安裝指令。例如，針對 CUDA 11.8 或 12.1 的安裝指令可能如下：

```bash
# 範例 (請依照新電腦的 CUDA 版本調整)：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 步驟 2.4: 安裝 BasicSR 及專案依賴
在確認 `conda activate basicsr` 已經啟動的狀態下，安裝此專案所需的套件，並以開發者模式安裝 BasicSR：

```bash
# 安裝 requirements.txt 內定義的依賴
pip install -r requirements.txt

# 以開發者模式安裝 BasicSR (這會執行 setup.py)
python setup.py develop
```
*(註：BasicSR 內的架構與模型，如 Restormer 或 CGNet 等依賴，均會在此階段一併準備就緒。如果您有用到 `einops` 等套件，`requirements.txt` 中會自動處理。)*

---

## 3. 準備資料集 (Datasets)

由於 GitHub 不會備份資料集檔案，您必須手動將訓練用的資料集轉移到新電腦上。

1.  將原電腦上的 `SIDD/` 資料夾或 `datasets/` 目錄，透過隨身碟、網路硬碟或 `scp` / `rsync` 複製到新電腦的 `BasicSR` 根目錄下。
2.  確保檔案結構與舊電腦一致。例如：
    *   `./SIDD/val/...`
    *   `./datasets/REDS4/...`
3.  檢查您要執行的設定檔（例如 `options/train/CGNet/...yml`），確保 `dataroot_gt` 與 `dataroot_lq` 的路徑與您放置資料集的位置相符。

---

## 4. 驗證與開始訓練

環境部署與資料集放置完成後，您就可以在新電腦上繼續進行訓練或測試了。

**執行訓練範例指令：**
```bash
# 單 GPU 訓練範例
python basicsr/train.py -opt options/train/CGNet/train_OmniDirDynamicCGNet.yml

# 多 GPU 訓練範例 (例如使用 2 張卡)
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/CGNet/train_OmniDirDynamicCGNet.yml --launcher pytorch
```

**若要接續先前的訓練 (Resume Training):**
請記得將原電腦上 `experiments/` 裡最新的 checkpoint (`.pth` 檔) 複製到新電腦對應的 `experiments/` 路徑下，並在 yaml 設定檔中將 `pretrain_network_g` 或 `resume_state` 指向該檔案。

祝您訓練順利！
