"""
Fine-Tune LoRA OmniVoice-Vietnamese (Ngoc Huyen Voice)
------------------------------------------------------
Base model: splendor1811/omnivoice-vietnamese
Dataset:   pnnbao-ump/ngochuyen_voice
Architecture: OmniVoice → LoRA on m.llm (Qwen3Model)
Task:      Audio language modeling (text → audio tokens)
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
import soundfile as sf
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForOmniVoice:
    """Collate samples into batches for OmniVoice."""
    pad_token_id: int = 0
    audio_mask_id: int = 1024

    def __call__(self, features):
        # features is a list of dicts with 'input_ids' tensors
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        batch_input_ids = []
        batch_labels = []
        batch_audio_mask = []
        batch_attention_mask = []

        for ids, labs in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            batch_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            batch_labels.append(
                torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)])
            )
            batch_attention_mask.append(
                torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]).bool()
            )
            # audio_mask: 1 for audio token positions, 0 for text
            batch_audio_mask.append(
                torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]).bool()
            )

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_mask),
            "audio_mask": torch.stack(batch_audio_mask),
        }


def preprocess_dataset(dataset, audio_tokenizer, text_tokenizer, device="cpu"):
    """Convert raw audio + text to OmniVoice input format.

    Dataset features: audio (array), transcription (str), file_name (str)
    """
    processed = []

    for i, sample in enumerate(dataset):
        audio_array = sample["audio"]["array"] if isinstance(sample["audio"], dict) else sample["audio"]
        sr = sample["audio"]["sampling_rate"] if isinstance(sample["audio"], dict) else 24000
        text = sample["transcription"]

        if i % 500 == 0:
            logger.info(f"Processing {i}/{len(dataset)}")

        # Convert audio to tensor
        audio_t = torch.from_numpy(audio_array).float()
        if sr != 24000:
            import torchaudio.functional as F
            audio_t = F.resample(audio_t, sr, 24000)

        # Normalize RMS
        rms = torch.sqrt(torch.mean(audio_t ** 2))
        if rms > 0:
            audio_t = audio_t * (0.13 / rms)

        # Encode audio → tokens using OmniVoice's audio tokenizer
        with torch.no_grad():
            audio_input = audio_t.unsqueeze(0).unsqueeze(0).to(device)
            enc = audio_tokenizer.encode(audio_input)
            audio_tokens = enc.audio_codes[0]  # [codebooks, time]
            # Flatten interleaved: [c0_t0, c1_t0, ..., c0_t1, c1_t1, ...]
            audio_tokens = audio_tokens.transpose(0, 1).reshape(-1)

        # Tokenize text
        text_tokens = text_tokenizer.encode(text, add_special_tokens=False)

        # Build input: text_tokens + audio_tokens
        input_ids = torch.tensor(text_tokens + audio_tokens.tolist(), dtype=torch.long)

        # Labels: predict audio tokens, ignore text tokens
        num_text = len(text_tokens)
        labels = input_ids.clone()
        labels[:num_text] = -100  # Don't compute loss on text

        processed.append({"input_ids": input_ids, "labels": labels})

    return processed


def main():
    # ── Paths (điều chỉnh cho V100 Docker) ──
    base_model_path = "./base_model/omnivoice-vietnamese"
    dataset_path = "./dataset/ngochuyen_voice"
    output_dir = "./omnivoice_ngochuyen_lora"

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
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
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
    # Trainable: ~5M params

    # ── Load & preprocess dataset ──
    logger.info("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    split = "train" if "train" in dataset else list(dataset.keys())[0]
    train_data = dataset[split]
    logger.info(f"Raw dataset: {len(train_data)} samples")

    # Precompute audio tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tokenizer = m.audio_tokenizer.to(device)
    text_tokenizer = m.text_tokenizer if hasattr(m, "text_tokenizer") else None

    if text_tokenizer is None:
        # Use HuggingFace tokenizer from the base model
        from transformers import AutoTokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

    logger.info("Preprocessing dataset (encoding audio → tokens)...")
    processed = preprocess_dataset(train_data, audio_tokenizer, text_tokenizer, device)
    audio_tokenizer.to("cpu")
    logger.info(f"Processed: {len(processed)} samples")

    # ── Training arguments ──
    # 7540 / (4*4) = 471 steps/epoch × 6 epochs = 2826 steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        logging_steps=50,
        save_strategy="steps",
        save_steps=471,
        save_total_limit=4,
        fp16=True,
        bf16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs"),
        gradient_checkpointing=False,
        optim="adamw_torch",
    )

    # ── Trainer ──
    trainer = Trainer(
        model=m,
        args=training_args,
        train_dataset=processed,
        data_collator=DataCollatorForOmniVoice(),
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
            "dataset": "pnnbao-ump/ngochuyen_voice",
            "lora_rank": 16,
            "lora_alpha": 32,
            "target_modules": lora_config.target_modules,
            "task_type": "FEATURE_EXTRACTION",
            "epochs": 6,
            "batch_size": 16,
            "learning_rate": "2e-5",
        }, f, indent=2)


if __name__ == "__main__":
    main()
