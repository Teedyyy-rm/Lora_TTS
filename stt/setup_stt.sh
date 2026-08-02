#!/bin/bash
# setup_stt.sh — cài môi trường cho STT (Whisper LoRA) trên V100
# Chạy: bash setup_stt.sh  (trong /root/Lora_TTS/stt)
set -e
source /root/venv_lora/bin/activate

echo "=== STT setup: Whisper large-v3 + PEFT ==="
pip install -q "transformers>=4.44" "datasets>=3.5" "peft>=0.12" "accelerate" \
    "soundfile" "librosa" "huggingface_hub"

# Verify
python -c "
import torch, transformers, peft
print(f'✅ torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}')
from transformers import WhisperForConditionalGeneration, WhisperProcessor
print('✅ Whisper classes OK')
"
echo "✅ STT setup xong — chạy: python3 train_whisper_lora.py --config config_stt.yaml"
