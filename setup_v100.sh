#!/bin/bash
# setup_v100.sh — Setup finetune LoRA OmniVoice trên V100 (1 lần chạy)
# Chạy: bash setup_v100.sh
set -e

echo "══════════════════════════════════════════"
echo "  SETUP FINETUNE LORA — V100"
echo "══════════════════════════════════════════"

# 1. Cập nhật pip + cài venv
echo ""
echo "[1/6] Python venv..."
apt-get update -qq && apt-get install -y -qq python3-venv git > /dev/null 2>&1 || true
python3 -m venv /root/venv_lora
source /root/venv_lora/bin/activate
pip install --upgrade pip -q

# 2. Cài torch CUDA
echo ""
echo "[2/6] PyTorch (CUDA 13.0)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
python -c "import torch; print(f'  ✅ torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')"

# 3. Clone repo Lora_TTS
echo ""
echo "[3/6] Clone Lora_TTS..."
cd /root
if [ ! -d Lora_TTS ]; then
    git clone https://github.com/Teedyyy-rm/Lora_TTS.git
fi
cd Lora_TTS

# 4. Cài dependencies
echo ""
echo "[4/6] Dependencies (transformers, peft, datasets, omnivoice)..."
pip install -r requirements.txt -q 2>&1 | grep -iE "error|successfully installed" | tail -3 || true

# 5. Clone base model OmniVoice-Vietnamese
echo ""
echo "[5/6] Base model splendor1811/omnivoice-vietnamese..."
mkdir -p /root/Lora_TTS/base_model
if [ ! -d /root/Lora_TTS/base_model/omnivoice-vietnamese ]; then
    export HF_TOKEN="${HF_TOKEN:-}"
    huggingface-cli download splendor1811/omnivoice-vietnamese \
        --local-dir /root/Lora_TTS/base_model/omnivoice-vietnamese 2>&1 | tail -2 || \
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('splendor1811/omnivoice-vietnamese', local_dir='/root/Lora_TTS/base_model/omnivoice-vietnamese')
print('  ✅ Base model cloned')"
fi

# 6. Clone dataset (sẽ tự clone khi train — nhưng clone sẵn cho chắc)
echo ""
echo "[6/6] Dataset Teedyyy-rm/Voice_Ngoc_Huyen..."
if [ ! -d /root/Lora_TTS/dataset/ngochuyen_voice ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Teedyyy-rm/Voice_Ngoc_Huyen', repo_type='dataset', local_dir='/root/Lora_TTS/dataset/ngochuyen_voice')
print('  ✅ Dataset cloned')"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ SETUP HOÀN TẤT!"
echo "  Train: cd /root/Lora_TTS && HF_TOKEN=... python3 train_lora.py"
echo "══════════════════════════════════════════"
