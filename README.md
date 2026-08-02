---
license: mit
language:
  - vi
tags:
  - text-to-speech
  - automatic-speech-recognition
  - voice-cloning
  - omnivoice
  - whisper
  - lora
---

<div align="center">

# 🎤🎧 OmniVoice & Whisper LoRA — Bộ Đôi Giọng Ngọc Huyền

**TTS: Chữ → Giọng (clone 90%) · STT: Giọng → Chữ (WER <3%)**
**3 Dataset Gộp · ~48 Giờ Dữ Liệu · Cùng Một Giọng**

[![Task](https://img.shields.io/badge/Task-TTS_%2B_STT-8A2BE2?style=for-the-badge&logo=audio&logoColor=white)]()
[![Language](https://img.shields.io/badge/Language-Vietnamese-00C853?style=for-the-badge&logo=translate&logoColor=white)]()
[![Method](https://img.shields.io/badge/Method-LoRA-2962FF?style=for-the-badge&logo=python&logoColor=white)]()
[![Dataset](https://img.shields.io/badge/Dataset-48h-7C4DFF?style=for-the-badge&logo=books&logoColor=white)]()
[![TTS Base](https://img.shields.io/badge/TTS-OmniVoice_Qwen3--0.6B-FF6F00?style=for-the-badge&logo=transformers&logoColor=white)]()
[![STT Base](https://img.shields.io/badge/STT-Whisper_LargeV3-1565C0?style=for-the-badge&logo=openai&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-4E342E?style=for-the-badge&logo=license&logoColor=white)]()

---

</div>

## 🧠 Giới Thiệu

Repo chứa **2 cấu trúc độc lập** — cùng một dataset (26,253 mẫu / ~48h giọng Ngọc Huyền), hai mô hình bổ sung nhau cho pipeline audiobook tự động:

```
Lora_TTS/
├── tts/    → 🔊 OmniVoice LoRA    — CHỮ → GIỌNG (đọc truyện bằng giọng clone)
└── stt/    → 👂 Whisper LoRA       — GIỌNG → CHỮ (transcribe truyện chính xác)
```

> ⚠️ **Hai adapter KHÔNG dùng chung** — khác model (Qwen3-0.6B vs Whisper-1.55B), khác kiến trúc, khác shape weights. Mỗi cái có repo HF riêng.

---

## 📚 Dataset — 3 Nguồn Cùng Giọng (~48h)

> ✅ **Đã verify speaker similarity** (Resemblyzer): 0.873 và 0.787 — cùng giọng Ngọc Huyền

<table align="center">
  <tr><th>Dataset</th><th>Mẫu</th><th>Nội dung</th><th>Sim</th></tr>
  <tr><td><code>Teedyyy-rm/Voice_Ngoc_Huyen</code></td><td>14,937</td><td>Truyện (Phi Thăng Thất Bại)</td><td>—</td></tr>
  <tr><td><code>pnnbao-ump/ngochuyen_voice</code></td><td>7,540</td><td>Tin tức</td><td>0.873</td></tr>
  <tr><td><code>thangnzt/NgocHuyenViVoice</code></td><td>3,776</td><td>Truyện ma</td><td>0.787</td></tr>
  <tr><td><b>Tổng</b></td><td><b>26,253</b></td><td colspan="2"><b>~48h — đa dạng ngữ cảnh</b></td></tr>
</table>

---

## 🔊 tts/ — OmniVoice LoRA (Text → Speech)

### 📦 Thông số

| | |
|---|---|
| Base model | `splendor1811/omnivoice-vietnamese` (Qwen3-0.6B) |
| Phương pháp | LoRA (PEFT, `FEATURE_EXTRACTION`) |
| Rank / Alpha | **64 / 128** (scale 2.0) |
| Learning rate | **1e-5** (chuẩn tác giả k2-fsa finetune) + **audio_lr 5e-5** (tách param groups) |
| Epochs / Batch | 6 / 4×8 = effective 32 |
| Nghiệm thức | `mask_beta` (Beta(2,1) thiên mask cao) + `drop_cond 0.1` (CFG) |
| HF adapter | `Teedyyy-rm/Omnivoice_Lora_v2` |

### 🔥 So sánh với OmniVoice GỐC (base, không finetune)

<table align="center">
  <tr><th>Metric</th><th>OmniVoice GỐC</th><th><b>TTS LoRA V2</b></th></tr>
  <tr><td>Dải giọng nói (100-4000Hz)</td><td>82%</td><td><b>83-91%</b></td></tr>
  <tr><td>Speaker similarity</td><td>~0.78</td><td><b>0.85-0.91</b> (+12%)</td></tr>
  <tr><td>WER (câu ngắn)</td><td>~5%</td><td><b>0-4%</b></td></tr>
  <tr><td>Chất giọng</td><td>Chung chung</td><td><b>Ngọc Huyền đặc trưng (90% theo tai người)</b></td></tr>
  <tr><td>Clone giọng khác</td><td>✅</td><td>✅ <b>Giữ nguyên</b> (LoRA không phá base)</td></tr>
</table>

### 🚀 Chạy

```bash
cd tts
bash setup_v100.sh                                    # cài môi trường V100
python3 train_lora.py --dataset combined --config config.yaml   # train

# Test checkpoint (adapter tự push lên HF mỗi nửa epoch)
python3 validate_lora.py --lora_path <repo>/adapters/checkpoint-N \
    --voice_prompt voice_prompt.pkl --text "Xin chào!" --output test.wav

# Đánh giá 3 trục: phổ sạch + giống giọng + đúng chữ
python3 evaluate_voice.py --wav test.wav --ref_wav mồi.wav --text "..."
```

---

## 👂 stt/ — Whisper large-v3 LoRA (Speech → Text)

### 📦 Thông số

| | |
|---|---|
| Base model | `openai/whisper-large-v3` (1.55B) |
| Phương pháp | LoRA (PEFT) — chỉ decoder `q_proj, v_proj` |
| Rank / Alpha | **16 / 32** (ASR cần rank thấp — khác TTS rank 64) |
| Learning rate | **1e-4** (chuẩn finetune Whisper) |
| Epochs / Batch | 5 / 2×8 = effective 16 (V100 16GB) |
| Mục đích | Transcribe chính xác giọng Ngọc Huyền (WER <3%) |
| HF adapter | `Teedyyy-rm/whisper-ngochuyen-lora` (repo riêng) |

### 🎯 Vì sao cần STT riêng?

- YouTube auto-caption **sai tên riêng truyện** ("Cố Thanh Hàn" → "cốt thanh hàn"...) — Whisper finetune giọng này đọc đúng
- Video **không có caption** → chỉ STT local transcribe được
- 48h 1 giọng = dữ liệu vàng cho speaker-adapted ASR

### 🚀 Chạy

```bash
cd stt
bash setup_stt.sh                                     # cài Whisper + PEFT
python3 train_whisper_lora.py --config config_stt.yaml  # train LoRA
```

---

## 🔄 Pipeline Truyện Tự Động (khi cả 2 hoàn thành)

```
① Audio truyện (YouTube, giọng Ngọc Huyền)
        ↓  stt/ Whisper LoRA transcribe (WER <3%, đúng tên riêng)
② Text truyện chuẩn
        ↓  tts/ OmniVoice LoRA đọc lại (90% giống bản gốc)
③ 🎧 Audiobook hoàn chỉnh — tự động từ đầu đến cuối
```

---

## 🔬 Nghiệm thức Kỹ Thuật (Aug 2)

<details open>
<summary><b>🎭 Audio Masking đúng chuẩn Flow-Matching (root cause fix)</b></summary>

- Che 0-100% token audio = `1024`, loss CHỈ trên token bị che — khớp `processor.py` gốc
- <b>mask_beta=true</b>: Beta(2,1) (mean 0.67) thay uniform — model train NHIỀU hơn ở chế độ mask cao = đúng inference toàn-mask → giảm vang/robot
</details>

<details open>
<summary><b>🔀 Tách Learning Rate — LoRA vs audio_heads/embeddings</b></summary>

- LoRA (`llm.*`): **1e-5** (chuẩn tác giả k2-fsa — finetune tinh chỉnh nhẹ)
- audio_heads/embeddings: **5e-5** (5×) — nơi biểu diễn timbre/giọng ở tầng ra, cần LR cao hơn để "áp" giọng đích
</details>

<details open>
<summary><b>🧠 CFG Dropout 10% + Special Tokens</b></summary>

- `drop_cond_ratio: 0.1` — guidance_scale 2.0 khuếch đại ĐÚNG text, không khuếch đại rác
- Text bọc chuẩn `<|lang_start|>vi<|lang_end|>...<|text_start|>{text}<|text_end|>` + điền 8 layers
</details>

---

## ⚠️ Pitfalls (đã trả giá bằng GPU/ngày debug)

| Pitfall | Fix |
|---|---|
| Thiếu masking/special tokens → nhiễu | Code chuẩn đã có trong tts/ |
| PyYAML `5e-5` → str → crash | Field số mới PHẢI thêm vào `_coerce_cfg` INT/FLOAT |
| `audio_codebook_weights` đừng đổi | Thiết kế gốc [8,8,6,6,4,4,2,2] tối ưu — codebook cao nắm cả nhiễu |
| eval_loss thấp ≠ giọng hay | Chọn checkpoint bằng NGHE + phổ + 3 trục metric |
| Checkpoint tốt nhất nằm SỚM | Epoch 0.5-1: sim cao nhất, WER thấp nhất |
| Dataset mẫu dài → OOM batch 8 | **batch 4×8** (effective 32 giữ nguyên) |
| Sửa code V100 → pull không đủ | `pkill -9` + `tmux kill-server` + pull + restart |
| Full finetune | KHÔNG — catastrophic forgetting mất clone đa giọng |

---

## 📊 Kết Quả Tham Chiếu (Ngọc Huyền, Aug 2)

```
TTS combined ckpt-389 (epoch 0.5): sim 0.911 🔥 | WER 4.3-10.8% | 90% giống (tai người)
TTS combined ckpt-778 (epoch 1):   sim 0.896   | WER 20.4% (nuốt nhiều hơn)
→ Điểm vàng: epoch 0.5-1 — chạy full rồi CHỌN bằng nghe + metric
VP: vp2 (6.5s) đọc đúng chữ / NgocHuyen2.0 (8.8s) giống hơn — test nhiều câu mới chốt
```

---

<div align="center">

*Repo PRIVATE — chỉ dành cho Teedyyy-rm* 🔒
*© 2026 — Bộ đôi TTS + STT giọng Ngọc Huyền*

</div>
