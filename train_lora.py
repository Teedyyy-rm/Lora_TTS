"""
Fine-Tune LoRA OmniVoice-Vietnamese (Ngoc Huyen Voice) — V2
--------------------------------------------------------------
Base model: splendor1811/omnivoice-vietnamese
Dataset:   Teedyyy-rm/Voice_Ngoc_Huyen  (dataset MỚI: nhiều giờ audio,
           có tên riêng truyện, pre-process sạch — thay pnnbao-ump/ngochuyen_voice cũ)
Architecture: OmniVoice → LoRA on m.llm (Qwen3Model)
Task:      Audio language modeling (text → audio tokens)

V2: output_dir + hub_model_id đổi tên (KHÔNG đè bản đang chạy trong omnivoice).
    Dataset tự clone từ HF nếu chưa có.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from datasets import load_from_disk
from transformers import (
    TrainingArguments,
    Trainer,
    Qwen3Model,           # Backbone bên trong OmniVoice
)
from peft import get_peft_model, LoraConfig, TaskType
import logging
from callbacks import DetailedLogCallback
import soundfile as sf
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Hyperparameters V2 (tối ưu cho dataset mới 14000+ mẫu) ──
NUM_EPOCHS = 4          # 10 → 4: data lớn, 10 epochs = overfit chắc chắn
VAL_RATIO = 0.05        # 5% validation set (phát hiện overfit, giữ best)
BATCH_SIZE = 8          # 4 → 8: V100 dư VRAM, gradient ổn định
GRAD_ACCUM = 4          # effective batch = 32 (như bản V100 cũ đã chạy tốt)


@dataclass
class DataCollatorForOmniVoice:
    """Collate samples into batches for OmniVoice."""
    pad_token_id: int = 0
    audio_mask_id: int = 1024

    def __call__(self, features):
        # features is a list of dicts with 3D tensors
        input_ids = [f["input_ids"] for f in features]  # list of [8, seq_len]
        labels = [f["labels"] for f in features]        # list of [8, seq_len]
        audio_masks = [f["audio_mask"] for f in features]
        attn_masks = [f["attention_mask"] for f in features]

        # Pad sequences along last dim (seq_len)
        max_len = max(ids.shape[1] for ids in input_ids)
        batch_input_ids = []
        batch_labels = []
        batch_audio_mask = []
        batch_attention_mask = []

        for ids, labs, am, attn in zip(input_ids, labels, audio_masks, attn_masks):
            pad_len = max_len - ids.shape[1]
            batch_input_ids.append(
                torch.cat([ids, torch.zeros((8, pad_len), dtype=torch.long)], dim=1)
            )
            batch_labels.append(
                torch.cat([labs, torch.full((8, pad_len), -100, dtype=torch.long)], dim=1)
            )
            batch_attention_mask.append(
                torch.cat([attn, torch.zeros(pad_len, dtype=torch.bool)])
            )
            batch_audio_mask.append(
                torch.cat([am, torch.zeros(pad_len, dtype=torch.bool)])
            )

        return {
            "input_ids": torch.stack(batch_input_ids),        # [batch, 9, max_len]
            "labels": torch.stack(batch_labels),              # [batch, 9, max_len]
            "attention_mask": torch.stack(batch_attention_mask),  # [batch, max_len]
            "audio_mask": torch.stack(batch_audio_mask),          # [batch, max_len]
        }


def preprocess_dataset(dataset, audio_tokenizer, text_tokenizer):
    """Convert raw audio + text to OmniVoice input format (on CPU).

    Dataset features: audio (array), transcription (str), file_name (str)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed = []
    from tqdm import tqdm

    for i, sample in enumerate(tqdm(dataset, desc="Encoding audio→tokens", unit="samples")):
        audio_field = sample["audio"]
        # HF Audio feature trả về AudioDecoder (hỗ trợ ["array"]/["sampling_rate"]),
        # dict, hoặc array thuần — xử lý cả 3
        if hasattr(audio_field, "array"):
            audio_array = audio_field["array"]
            sr = audio_field["sampling_rate"]
        elif isinstance(audio_field, dict):
            audio_array = audio_field.get("array")
            sr = audio_field.get("sampling_rate", 24000)
        else:
            audio_array = audio_field
            sr = 24000
        text = sample["transcription"]

        # Convert audio to tensor
        audio_t = torch.from_numpy(audio_array).float()
        if sr != 24000:
            import torchaudio.functional as F
            audio_t = F.resample(audio_t, sr, 24000)

        # Normalize RMS
        rms = torch.sqrt(torch.mean(audio_t ** 2))
        if rms > 0:
            audio_t = audio_t * (0.13 / rms)

        # Encode audio → tokens: [1, 8, time] — encode on CPU, avoid VRAM leak
        with torch.no_grad():
            audio_input = audio_t.unsqueeze(0).unsqueeze(0).to(device)
            enc = audio_tokenizer.encode(audio_input)
            audio_tokens = enc.audio_codes[0]  # [8, time]

        # Tokenize text
        text_tokens = text_tokenizer.encode(text, add_special_tokens=False)

        num_text = len(text_tokens)
        num_audio = audio_tokens.shape[1]
        seq_len = num_text + num_audio

        # Build input_ids: [8, seq_len]
        # Layer 0: text tokens + audio codebook 0
        # Layers 1-7: audio codebooks 1-7
        # codebook_layer_offsets: [0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]
        input_ids = torch.zeros((8, seq_len), dtype=torch.long)
        input_ids[0, :num_text] = torch.tensor(text_tokens, dtype=torch.long)
        input_ids[:, num_text:] = audio_tokens  # [8, time]

        # audio_mask: 0=text, 1=audio
        audio_mask = torch.zeros(seq_len, dtype=torch.bool)
        audio_mask[num_text:] = True

        # labels: -100 for text positions, audio tokens for audio positions
        labels = input_ids.clone()
        labels[:, :num_text] = -100  # ignore text positions (all layers)

        processed.append({
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "attention_mask": torch.ones(seq_len, dtype=torch.bool),
        })

    return processed

    return processed


def main():
    # ── Paths (điều chỉnh cho V100 Docker) ──
    base_model_path = "./base_model/omnivoice-vietnamese"
    dataset_path = "./dataset/ngochuyen_voice"     # sau khi clone dataset mới
    output_dir = "./omnivoice_ngochuyen_lora_v2"   # V2 — KHÔNG đè bản cũ

    # ── Clone dataset MỚI từ HuggingFace (nếu chưa có) ──
    # Dataset mới: Teedyyy-rm/Voice_Ngoc_Huyen (nhiều giờ audio, có tên riêng truyện,
    # pre-process sạch: trim silence, ép thở, volume chuẩn, relative path)
    if not os.path.isdir(dataset_path):
        logger.info("Cloning dataset từ HuggingFace: Teedyyy-rm/Voice_Ngoc_Huyen ...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Teedyyy-rm/Voice_Ngoc_Huyen",
            repo_type="dataset",
            local_dir=dataset_path,
            token=os.environ.get("HF_TOKEN"),
        )
        logger.info(f"✅ Dataset cloned → {dataset_path}")

    # ── Load OmniVoice model ──
    logger.info("Loading OmniVoice model...")
    import omnivoice
    m = omnivoice.OmniVoice.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    logger.info(f"Loaded: OmniVoice (LLM={type(m.llm).__name__})")

    # ── Freeze everything EXCEPT m.llm ──
    for name, param in m.named_parameters():
        if "llm" not in name:
            param.requires_grad = False

    # ── Apply LoRA to m.llm (Qwen3Model, base transformer) ──
    # V2 tối ưu: rank 128 (như bản V100 cũ chạy tốt — học chi tiết giọng,
    # giảm "giọng khô") + alpha = 2×rank
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    m.llm = get_peft_model(m.llm, lora_config)
    m.llm.print_trainable_parameters()
    # Trainable: ~40M / 605M ≈ 6.6%

    # ── Load dataset ──
    logger.info("Loading dataset...")
    from datasets import load_dataset
    # Dataset trên HF là parquet (audio bytes nhúng) — dùng load_dataset,
    # KHÔNG dùng load_from_disk (chỉ đọc được DatasetDict lưu disk cũ)
    if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path, token=os.environ.get("HF_TOKEN"))
    split = "train" if "train" in dataset else list(dataset.keys())[0]
    train_data = dataset[split]
    logger.info(f"Raw dataset: {len(train_data)} samples")

    # Precompute audio tokens (GPU encode for speed, clear cache between loops)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tokenizer = m.audio_tokenizer.to(device)
    torch.cuda.empty_cache()
    text_tokenizer = m.text_tokenizer if hasattr(m, "text_tokenizer") else None

    if text_tokenizer is None:
        # Use HuggingFace tokenizer from the base model
        from transformers import AutoTokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

    logger.info("Preprocessing dataset (encoding audio → tokens)...")
    processed = preprocess_dataset(train_data, audio_tokenizer, text_tokenizer)
    audio_tokenizer.to("cpu")
    logger.info(f"Processed: {len(processed)} samples")

    # ── Tách Validation set (5%) — mô hình KHÔNG học phần này ──
    # Mục đích: đo eval_loss sau mỗi N steps → phát hiện overfit (học vẹt)
    # sớm, load_best_model_at_end giữ checkpoint tốt nhất
    import random
    random.seed(42)
    indices = list(range(len(processed)))
    random.shuffle(indices)
    n_val = max(1, int(len(processed) * VAL_RATIO))
    val_indices = set(indices[:n_val])
    train_processed = [p for i, p in enumerate(processed) if i not in val_indices]
    val_processed = [p for i, p in enumerate(processed) if i in val_indices]
    logger.info(f"Split: {len(train_processed)} train + {len(val_processed)} val "
                f"({VAL_RATIO:.0%} validation)")

    # ── Training arguments ──
    # V2 tối ưu cho dataset MỚI (14000+ mẫu):
    #   - 4 epochs (10 quá nhiều → overfit)
    #   - batch 8 × 4 accum = effective 32 (V100 dư VRAM, gradient ổn định)
    #   - save_steps = steps/epoch (tính động theo dataset thật)
    steps_per_epoch = max(1, len(train_processed) // (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * NUM_EPOCHS
    logger.info(f"Steps/epoch: {steps_per_epoch}, total: {total_steps} "
                f"({NUM_EPOCHS} epochs)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_strategy="steps",
        save_steps=steps_per_epoch,          # save 1 lần/epoch
        save_total_limit=4,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,          # eval 1 lần/epoch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        bf16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id="Teedyyy-rm/omnivoice-ngochuyen-lora-v2",
        hub_strategy="checkpoint",
        hub_token=os.environ.get("HF_TOKEN"),
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_first_step=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
    )

    # ── Trainer ──
    trainer = Trainer(
        model=m,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        data_collator=DataCollatorForOmniVoice(),
        callbacks=[DetailedLogCallback()],
    )

    # ── Train ──
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # ── Save final LoRA (only m.llm adapter) ──
    final_path = os.path.join(output_dir, "final_lora")
    m.llm.save_pretrained(final_path)
    logger.info(f"Final LoRA saved to: {final_path}")

    # ── Also save config ──
    import json
    with open(os.path.join(final_path, "training_config.json"), "w") as f:
        json.dump({
            "base_model": base_model_path,
            "dataset": "Teedyyy-rm/Voice_Ngoc_Huyen",
            "lora_rank": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": lora_config.target_modules,
            "task_type": "FEATURE_EXTRACTION",
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "val_ratio": VAL_RATIO,
            "eval_strategy": "steps",
            "load_best_model_at_end": True,
        }, f, indent=2)


if __name__ == "__main__":
    main()
