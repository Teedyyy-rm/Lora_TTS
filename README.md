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
| **LoRA rank** | 64 | alpha = 128 (2×rank) |
| **LoRA dropout** | 0.05 | |
| **Target modules** | q,k,v,o + gate,up,down | 7 modules |
| **Epochs** | 10 | ~2048 steps với 3277 mẫu |
| **Batch** | 4 × 4 grad accum | effective 16 |
| **Learning rate** | 2e-5 | cosine scheduler, warmup 200 |
| **Precision** | FP16 | V100 |
| **Hub output** | `Teedyyy-rm/omnivoice-ngochuyen-lora-v2` | V2 — KHÔNG đè bản cũ |

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

## 🆚 V2 vs bản cũ

| | Bản cũ (`omnivoice-ngochuyen-lora`) | **V2 (repo này)** |
|---|---|---|
| Dataset | `pnnbao-ump/ngochuyen_voice` (TIN Vbee, 7540 mẫu) | **`Teedyyy-rm/Voice_Ngoc_Huyen`** (mới, sạch, có tên riêng) |
| LoRA rank | 128 (qlora_train_v100.py cũ) | 64 (config này) |
| Trainer | Custom loop | **HF Trainer** |
| Output | `omnivoice-ngochuyen-lora` | `omnivoice-ngochuyen-lora-v2` |
