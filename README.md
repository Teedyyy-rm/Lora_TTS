# Lora_TTS — Finetune LoRA OmniVoice (Giọng Ngọc Huyền)

Finetune **LoRA** cho **OmniVoice-Vietnamese** (base `splendor1811/omnivoice-vietnamese`)
để clone giọng đọc **Ngọc Huyền** — dùng cho pipeline TTS StoryCast.

## 📋 Kiến trúc

```
OmniVoice (Qwen3Model 605M backbone)
   └── LoRA trên m.llm (7 modules: q,k,v,o_proj + gate,up,down_proj)
        └── ~40M trainable / 605M ≈ 6.6%
```

## 🚀 Cách chạy (trên V100 / máy train)

```bash
# 1. Clone repo
git clone https://github.com/Teedyyy-rm/Lora_TTS.git
cd Lora_TTS

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Login HF (để clone dataset + push LoRA)
huggingface-cli login   # hoặc export HF_TOKEN=...

# 4. Train — TỰ ĐỘNG:
#    - Clone dataset Teedyyy-rm/Voice_Ngoc_Huyen nếu chưa có
#    - Clone base model splendor1811/omnivoice-vietnamese (đặt trong ./base_model/)
#    - Train + push adapter lên HF
python3 train_lora.py
```

> **Lưu ý:** base model cần ở `./base_model/omnivoice-vietnamese`
> (clone trước: `huggingface-cli download splendor1811/omnivoice-vietnamese --local-dir ./base_model/omnivoice-vietnamese`)

## ⚙️ Cấu hình (train_lora.py)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| **Dataset** | `Teedyyy-rm/Voice_Ngoc_Huyen` | Dataset mới: nhiều giờ audio, có tên riêng truyện, pre-process sạch |
| **LoRA rank** | 128 | alpha = 256 (2×rank) — như bản V100 cũ chạy tốt |
| **LoRA dropout** | 0.05 | |
| **Target modules** | q,k,v,o + gate,up,down | 7 modules |
| **Epochs** | 4 | 10 quá nhiều với dataset 14000+ → overfit |
| **Batch** | 8 × 4 grad accum | effective 32 (V100 dư VRAM) |
| **Validation** | 5% (VAL_RATIO=0.05) | eval mỗi epoch, `load_best_model_at_end` chống overfit |
| **Learning rate** | 2e-5 | cosine scheduler, warmup 100 |
| **Precision** | FP16 | V100 |
| **Hub output** | `Teedyyy-rm/LoRa_Ngoc_Huyen_2.0` | **PRIVATE** — Ngọc Huyền 2.0, KHÔNG đè bản cũ |

## 📂 Các file

| File | Chức năng |
|---|---|
| `train_lora.py` | Script train chính (HF Trainer API) |
| `callbacks.py` | Log đẹp: Loss/LR/Epoch/VRAM/% hoàn thành (không phá logs gốc) |
| `validate_lora.py` | Test generation sau train |
| `requirements.txt` | Dependencies (torch ≥2.6, transformers ≥5.14, peft ≥0.14) |

## 🎯 Kết quả mong đợi

- Dataset mới giải quyết: **nuốt chữ tên riêng** (có tên truyện: Thẩm Hữu Vi, Mai Mồi...)
- Data nhiều giờ hơn → giảm **overfit/nhiễu**
- So sánh A/B với bản cũ `omnivoice-ngochuyen-lora` (rank 128, data TIN Vbee)

## 🧪 Validate sau train

```bash
python3 validate_lora.py --lora-path ./omnivoice_ngochuyen_lora_v2/final_lora --text "Câu test..."
```

## 🐛 Troubleshooting — CÁC LỖI ĐÃ GẶP & FIX (đừng debug lại!)

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `CUDA error: no kernel image is available` | torch ≥2.7 không hỗ trợ V100 (sm_70) | `pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126` |
| `ImportError: To support decoding audio data, please install 'torchcodec'` | datasets ≥4.x ép dùng torchcodec | `pip install datasets==3.5.0` (dùng soundfile — không cần torchcodec) |
| `libnppicc.so.12: cannot open shared object file` | torchcodec cần NVIDIA NPP (container minimized thiếu) | Không cài torchcodec — dùng datasets 3.5.0 |
| `FileNotFoundError: ... neither a Dataset directory nor DatasetDict` | Logic load dataset sai (local dir có dataset_info.json là HF parquet format) | `load_dataset(local_dir)` — chỉ `load_from_disk` khi có `dataset_dict.json` |
| `Error opening terminal: xterm-kitty` | V100 minimized không có terminfo Kitty | `export TERM=xterm-256color` (đã thêm vào ~/.bashrc) |
| `Error opening terminal: unknown` khi dùng `watch` qua SSH | watch cần tty | `ssh -t -p 40202 ... "watch ..."` hoặc dùng `nvidia-smi -l 1` |

## 🆚 V2 vs bản cũ

| | Bản cũ (`omnivoice-ngochuyen-lora`) | **V2 (repo này)** |
|---|---|---|
| Dataset | `pnnbao-ump/ngochuyen_voice` (TIN Vbee, 7540 mẫu) | **`Teedyyy-rm/Voice_Ngoc_Huyen`** (mới, sạch, có tên riêng) |
| LoRA rank | 128 (qlora_train_v100.py cũ) | 128 (config này) |
| Epochs | ~15000 steps | 4 epochs (~3500 steps) + eval 5% |
| Trainer | Custom loop | **HF Trainer** + eval/early-stop |
| Output | `omnivoice-ngochuyen-lora` | `LoRa_Ngoc_Huyen_2.0` |
