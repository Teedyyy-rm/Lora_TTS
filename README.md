# 🎤🎧 Lora_TTS — Bộ đôi TTS + STT cho giọng Ngọc Huyền

Repo gồm **2 cấu trúc độc lập** — cùng dataset, 2 adapter RIÊNG BIỆT (không dùng chung):

```
Lora_TTS/
├── tts/    → 🔊 OmniVoice LoRA — CHỮ → GIỌNG (đọc truyện bằng giọng Ngọc Huyền)
└── stt/    → 👂 Whisper large-v3 LoRA — GIỌNG → CHỮ (transcribe truyện chính xác)
```

## 🔊 tts/ — OmniVoice LoRA (Text → Speech)

| | |
|---|---|
| Model | `splendor1811/omnivoice-vietnamese` (Qwen3-0.6B) + LoRA r64 |
| Chức năng | Đọc text → giọng Ngọc Huyền clone (90% giống bản gốc ✅) |
| Config chuẩn | rank 64, LR 1e-5, audio_lr 5e-5, mask_beta, drop_cond 0.1 |
| Train | `train_lora.py --dataset combined --config config.yaml` |
| Test | `validate_lora.py` + `evaluate_voice.py` (3 trục: phổ/sim/WER) |
| HF adapter | `Teedyyy-rm/Omnivoice_Lora_v2` |

```bash
cd tts
bash setup_v100.sh            # cài môi trường
python3 train_lora.py --dataset combined --config config.yaml
```

> ⚠️ Đọc `../.hermes/skills/mlops/omnivoice-finetuning/references/quickstart-lora-reuse.md` (skill) cho toàn bộ pitfalls đã trả giá.

## 👂 stt/ — Whisper large-v3 LoRA (Speech → Text)

| | |
|---|---|
| Model | `openai/whisper-large-v3` (1.55B) + LoRA r16 |
| Chức năng | Nghe audio truyện → text chuẩn (WER <3% giọng Ngọc Huyền) |
| Mục đích | Tự động hóa pipeline: audio truyện → text → TTS đọc lại |
| Train | `train_whisper_lora.py --config config_stt.yaml` |
| HF adapter | `Teedyyy-rm/whisper-ngochuyen-lora` (repo riêng) |

```bash
cd stt
bash setup_stt.sh             # cài môi trường (dùng chung venv V100)
python3 train_whisper_lora.py --config config_stt.yaml
```

## 🔄 Pipeline truyện tự động (khi cả 2 xong)

```
① Audio truyện (YouTube) ──→ ② STT transcribe text chuẩn ──→ ③ TTS đọc lại bằng giọng Ngọc Huyền
```

## ⚠️ Quan trọng

- **2 adapter KHÔNG dùng chung** — khác model (Qwen3-0.6B vs Whisper-1.55B), khác kiến trúc, khác shape weights.
- Dataset 3 repo (26,253 mẫu / 48h) dùng chung cho cả 2 — đã verify cùng giọng (sim ≥0.75).
- Thêm field số mới vào yaml → PHẢI khai báo trong `_coerce_cfg` INT/FLOAT lists (PyYAML `5e-5` → str crash).
