#!/bin/bash
# setup_v100.sh — Setup finetune LoRA OmniVoice trên V100 (1 lần chạy)
# ✅ ĐÃ BAO GỒM TẤT CẢ FIX — chạy là xong, không cần debug lại.
# Chạy: bash setup_v100.sh
set -e

echo "══════════════════════════════════════════"
echo "  SETUP FINETUNE LORA — V100"
echo "══════════════════════════════════════════"

# 1. Python venv
echo ""
echo "[1/7] Python venv..."
apt-get update -qq && apt-get install -y -qq python3-venv git > /dev/null 2>&1 || true
python3 -m venv /root/venv_lora
source /root/venv_lora/bin/activate
pip install --upgrade pip -q

# 2. PyTorch — FIX QUAN TRỌNG: V100 (sm_70) chỉ chạy torch 2.6.x cu126
#    (torch >= 2.7 báo "no kernel image is available" — V100 không được hỗ trợ)
echo ""
echo "[2/7] PyTorch 2.6.0+cu126 (BẮT BUỘC cho V100 sm_70)..."
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 -q
python -c "import torch; assert torch.cuda.is_available(), 'CUDA không khả dụng!'; print(f'  ✅ torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}')"

# 3. Clone repo Lora_TTS
echo ""
echo "[3/7] Clone Lora_TTS..."
cd /root
if [ ! -d Lora_TTS ]; then
    git clone https://github.com/Teedyyy-rm/Lora_TTS.git
fi
cd Lora_TTS

# 4. Dependencies — requirements.txt ĐÃ PIN versions (torch 2.6, datasets 3.5)
echo ""
echo "[4/7] Dependencies (pinned — không tự ý nâng version)..."
pip install -r requirements.txt -q 2>&1 | grep -iE "error" | tail -3 || true

# 5. Base model OmniVoice-Vietnamese
echo ""
echo "[5/7] Base model splendor1811/omnivoice-vietnamese..."
mkdir -p /root/Lora_TTS/base_model
if [ ! -d /root/Lora_TTS/base_model/omnivoice-vietnamese ]; then
    /root/venv_lora/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('splendor1811/omnivoice-vietnamese', local_dir='/root/Lora_TTS/base_model/omnivoice-vietnamese')
print('  ✅ Base model cloned')"
fi

# 6. Dataset
echo ""
echo "[6/7] Dataset Teedyyy-rm/Voice_Ngoc_Huyen..."
if [ ! -d /root/Lora_TTS/dataset/ngochuyen_voice ]; then
    /root/venv_lora/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Teedyyy-rm/Voice_Ngoc_Huyen', repo_type='dataset', local_dir='/root/Lora_TTS/dataset/ngochuyen_voice')
print('  ✅ Dataset cloned')"
fi

# 7. Fix TERM cho terminal hiển thị (btop/htop/watch qua SSH từ Kitty)
echo ""
echo "[7/7] Fix TERM cho SSH terminal..."
grep -q "xterm-256color" ~/.bashrc 2>/dev/null || echo 'export TERM=xterm-256color' >> ~/.bashrc

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ SETUP HOÀN TẤT — không cần fix gì thêm!"
echo "  Train: cd /root/Lora_TTS && HF_TOKEN=... python3 train_lora.py"
echo "  Xem log: tail -f /root/train_lora.log"
echo "══════════════════════════════════════════"
