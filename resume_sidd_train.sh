#!/bin/bash
# 續傳並解壓 SIDD 訓練集指令腳本 (SIDD Train Dataset Resume & Extract Script)

echo "正在嘗試接續下載 SIDD 訓練集 (剩餘約 6.8 GB)..."
./venv/bin/gdown --continue 1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw -O SIDD/train.zip

if [ $? -eq 0 ]; then
    echo "下載完成！正在解壓縮 SIDD 訓練集..."
    ./venv/bin/python -c "import shutil; shutil.unpack_archive('SIDD/train.zip', 'SIDD')"
    
    echo "正在整理目錄架構..."
    mv SIDD/train/SIDD/* SIDD/train/ 2>/dev/null
    rmdir SIDD/train/SIDD 2>/dev/null
    
    echo "🎉 SIDD 訓練集解壓整理與連結完成！模型現可透過 datasets/SIDD 呼叫。"
else
    echo "⚠️ Google Drive 匿名下載流量限制目前仍未解除（通常需等待數小時或 24 小時）。"
    echo "💡 解決方案：您也可以在瀏覽器中開啟 https://drive.google.com/uc?id=1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw 登入 Google 帳號下載 train.zip，放入 SIDD/ 目錄後再次執行本腳本即可自動解壓與整理！"
fi
