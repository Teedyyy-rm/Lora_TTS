#!/usr/bin/env python3
"""STT: Finetune Whisper large-v3 (LoRA) cho giọng Ngọc Huyền.

Mục đích: transcribe CHÍNH XÁC audio truyện giọng Ngọc Huyền (WER <3%)
→ tự động hóa pipeline: audio truyện → text chuẩn → TTS đọc lại.

Dựa trên chuẩn openai/whisper-large-v3 + PEFT LoRA (giữ base, tránh overfit).
Khác hẳn TTS (chữ→giọng) — file này là ASR (giọng→chữ).

Cách chạy (V100):
    source /root/venv_lora/bin/activate
    python3 train_whisper_lora.py --config config_stt.yaml
"""
import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("train_whisper_stt")

# ── INT/FLOAT coerce (bài học PyYAML '2e-5' → str crash — PHẢI khai báo field mới) ──
INT_FIELDS = {"num_epochs", "batch_size", "grad_accum", "warmup_steps", "max_length",
              "logging_steps", "save_steps", "seed", "lora_rank", "lora_alpha", "max_steps"}
FLOAT_FIELDS = {"learning_rate", "warmup_ratio", "lora_dropout", "val_ratio"}


def _coerce_cfg(cfg: dict) -> dict:
    for k, v in list(cfg.items()):
        if v is None or isinstance(v, (list, dict, bool)):
            continue
        if k in INT_FIELDS and isinstance(v, str):
            cfg[k] = int(v)
        elif k in FLOAT_FIELDS and isinstance(v, str):
            cfg[k] = float(v)
    return cfg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config_stt.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    return ap.parse_args()


def resolve_config(args):
    """CLI > yaml > default (đồng bộ logic với TTS)."""
    cfg = {"model": "openai/whisper-large-v3", "num_epochs": 5, "batch_size": 2,
           "grad_accum": 8, "learning_rate": 1e-4, "warmup_ratio": 0.03,
           "lora_rank": 16, "lora_alpha": 32, "val_ratio": 0.05,
           "max_length": 448, "seed": 42, "push_to_hub": False}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    if args.epochs:
        cfg["num_epochs"] = args.epochs
    return _coerce_cfg(cfg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg["seed"])
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    # ── Dataset: 3 repo gộp (cùng giọng Ngọc Huyền — verify sim ≥0.75 trước khi gộp) ──
    from datasets import load_dataset, concatenate_datasets
    from huggingface_hub import snapshot_download

    repos = cfg.get("repo", ["Teedyyy-rm/Voice_Ngoc_Huyen"])
    dataset_path = cfg.get("dataset_path", "./dataset/whisper_ngochuyen")

    os.makedirs(dataset_path, exist_ok=True)
    datasets_list = []
    for repo in repos:
        dest = os.path.join(dataset_path, repo.split("/")[-1])
        if not os.path.isdir(dest):
            logger.info(f"Cloning dataset: {repo} ...")
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest,
                              token=os.environ.get("HF_TOKEN"))
        ds = load_dataset(dest, token=os.environ.get("HF_TOKEN"))
        split = "train" if "train" in ds else list(ds.keys())[0]
        datasets_list.append(ds[split])
        logger.info(f"  + {repo}: {len(ds[split])} samples")
    dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    logger.info(f"✅ STT dataset: {len(dataset)} samples ({len(repos)} repos)")

    # ── Load Whisper + processor ──
    from transformers import (
        WhisperForConditionalGeneration, WhisperProcessor,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    processor = WhisperProcessor.from_pretrained(cfg["model"], language="vi",
                                                 task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.float16)
    model.config.forced_decoder_ids = None  # bỏ ép ngôn ngữ cứng — dùng Việt
    model.config.suppress_tokens = []

    # ⚠️ LoRA CHỈ vào decoder attention (chuẩn finetune Whisper — encoder giữ nguyên)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],  # decoder self-attn — đủ cho speaker-adapted
        bias="none",
    ))
    model.print_trainable_parameters()

    # ── Preprocess: resample 16kHz + feature extractor ──
    def prepare(batch):
        audio = batch["audio"]
        array = audio["array"] if isinstance(audio, dict) else audio
        sr = audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000
        features = processor(audio=array, sampling_rate=sr,
                             return_tensors="pt").input_features[0]
        text = batch.get("transcription") or batch.get("text", "")
        labels = processor.tokenizer(text, padding="max_length",
                                     max_length=cfg["max_length"],
                                     truncation=True).input_ids
        batch["input_features"] = features
        batch["labels"] = labels
        return batch

    dataset = dataset.map(prepare, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=cfg["val_ratio"], seed=cfg["seed"])
    train_ds, val_ds = split["train"], split["test"]
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Training ──
    steps_per_epoch = max(1, len(train_ds) // (cfg["batch_size"] * cfg["grad_accum"]))
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))
    logger.info(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")

    output_dir = cfg.get("output_dir", "./output_whisper_ngochuyen")
    args_tr = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=warmup_steps,
        num_train_epochs=cfg["num_epochs"],
        evaluation_strategy="steps",
        eval_steps=max(1, steps_per_epoch // 2),
        save_steps=max(1, steps_per_epoch // 2),
        logging_steps=cfg["logging_steps"],
        fp16=True,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=cfg["max_length"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args_tr,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

    # ── Save LoRA adapter (STT adapter — RIÊNG, không dùng chung với TTS!) ──
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    processor.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    logger.info(f"✅ STT LoRA saved → {output_dir}/lora_adapter")

    hub_id = cfg.get("hub_model_id")
    if hub_id and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=hub_id, repo_type="model", private=True,
                        exist_ok=True, token=os.environ.get("HF_TOKEN"))
        api.upload_folder(repo_id=hub_id, repo_type="model",
                          folder_path=os.path.join(output_dir, "lora_adapter"),
                          token=os.environ.get("HF_TOKEN"))
        logger.info(f"✅ STT LoRA pushed → {hub_id}")


if __name__ == "__main__":
    main()
