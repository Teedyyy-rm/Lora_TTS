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
    TrainerCallback,
    Qwen3Model,           # Backbone bên trong OmniVoice
)
from peft import get_peft_model, LoraConfig, TaskType
import logging
from callbacks import DetailedLogCallback
import soundfile as sf
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Hyperparameters V3 (tối ưu CHẤT LƯỢNG — Aug 2, V100 mới 16GB) ──
# Ưu tiên chất lượng giọng, không cần nhanh:
#   - 3 epochs (KHÔNG 4 — đã chứng minh epoch 4 overfit: eval_loss thấp nhưng nhiễu 25%)
#   - batch 16 × 2 accum = effective 32 (giữ nhịp học như bản chạy tốt, gradient mượt hơn;
#     8→16 tận dụng VRAM dư; bắt buộc gradient_checkpointing=True)
#   - dropout 0.0 (Unsloth: LoRA dropout không hữu ích cho TTS, 0 = consistency)
NUM_EPOCHS = 3          # 4 → 3: chống overfit (checkpoint-1329 epoch 3 = giọng chuẩn)
VAL_RATIO = 0.05        # 5% validation set (phát hiện overfit, giữ best)
BATCH_SIZE = 16         # 8 → 16: tận dụng VRAM dư 6GB, gradient ổn định
GRAD_ACCUM = 2          # 4 → 2: giữ effective 32 (16×2)


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


class PushAdapterOnSave(TrainerCallback):
    """Sau mỗi lần Trainer save checkpoint (mỗi epoch): trích LoRA adapter chuẩn
    + audio_specific.pt → push lên HF (adapters/checkpoint-{step}/).

    Mục đích (fix: hub_strategy=checkpoint chỉ push FULL MODEL 2.2GB + optimizer 616MB,
    KHÔNG có adapter_config.json → máy cá nhân không test được giữa chừng):
    mỗi epoch, m.llm.save_pretrained() tạo adapter chuẩn (~160MB, có adapter_config.json)
    + audio_specific.pt (~65MB) → upload lên HF. Máy cá nhân tải ~230MB là TEST ĐƯỢC
    NGAY mà không cần đợi train xong, không cần tải full model 2.2GB.
    """

    def __init__(self, model, hub_model_id, hub_token, output_dir):
        self.model = model            # OmniVoice wrapper (m.llm = PeftModel)
        self.hub_model_id = hub_model_id
        self.hub_token = hub_token
        self.output_dir = output_dir

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        step = state.global_step
        if step <= 0:
            return
        try:
            # 1. LoRA adapter chuẩn (adapter_config.json + adapter_model.safetensors)
            adapter_dir = os.path.join(self.output_dir, f"adapter_step{step}")
            os.makedirs(adapter_dir, exist_ok=True)
            self.model.llm.save_pretrained(adapter_dir)

            # 1b. ⚠️ SANITIZE adapter_config.json — peft V100 (0.20) viết 40+ keys mới
            #     (alora_invocation_tokens, arrow_config, corda_config, eva_config,
            #      megatron_core, monteclora_config, velora_config, use_dora, use_rslora...)
            #     mà peft máy cá nhân (0.12) KHÔNG hiểu → LoraConfig crash
            #     ('unexpected keyword argument'). Chỉ giữ whitelist peft 0.12 hiểu.
            import json as _json
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            raw_cfg = _json.load(open(cfg_path))
            whitelist = {
                "peft_type", "r", "lora_alpha", "lora_dropout", "bias",
                "task_type", "target_modules", "base_model_name_or_path",
                "fan_in_fan_out", "init_lora_weights", "use_rslora",
            }
            clean_cfg = {k: raw_cfg[k] for k in whitelist if k in raw_cfg}
            with open(cfg_path, "w") as fp:
                _json.dump(clean_cfg, fp, indent=2)
            print(f"  adapter_config.json sanitized: {len(raw_cfg)} → {len(clean_cfg)} keys",
                  flush=True)

            # 2. audio_specific.pt — CHỈ audio_heads + audio_embeddings (không tokenizer)
            torch.save(
                {n: p.detach().cpu() for n, p in self.model.named_parameters()
                 if ("audio_" in n and "llm" not in n
                     and "audio_tokenizer" not in n)},
                os.path.join(adapter_dir, "audio_specific.pt"),
            )

            # 3. Push lên HF: adapters/checkpoint-{step}/
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=adapter_dir,
                repo_id=self.hub_model_id,
                repo_type="model",
                path_in_repo=f"adapters/checkpoint-{step}",
                token=self.hub_token,
            )
            size_mb = os.path.getsize(
                os.path.join(adapter_dir, "adapter_model.safetensors")) / 1e6
            print(f"✅ [PushAdapterOnSave] step {step}: adapter ({size_mb:.0f}MB) "
                  f"+ audio_specific.pt → adapters/checkpoint-{step} trên HF",
                  flush=True)
        except Exception as e:
            print(f"⚠️ [PushAdapterOnSave] step {step} push lỗi (train vẫn tiếp tục): {e}",
                  flush=True)


def main():
    # ── Paths (điều chỉnh cho V100 Docker) ──
    base_model_path = "./base_model/omnivoice-vietnamese"
    dataset_path = "./dataset/ngochuyen_voice"     # sau khi clone dataset mới
    output_dir = "./omnivoice_ngochuyen_lora_2.0"  # Ngọc Huyền 2.0 — KHÔNG đè bản cũ

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

    # ── Load OmniVoice model (FP32 — Trainer tự cast FP16) ──
    # QUAN TRỌNG: load FP32, KHÔNG cast trước. Trainer fp16=True sẽ tự:
    #   model → FP16 (autocast) + optimizer master weights → FP32
    # Nếu load/cast FP16 thủ công → optimizer giữ FP16 params → gradients FP16
    # → GradScaler lỗi "Attempting to unscale FP16 gradients".
    logger.info("Loading OmniVoice model (FP32)...")
    import omnivoice
    m = omnivoice.OmniVoice.from_pretrained(
        base_model_path,
        device_map="auto",
    )
    logger.info(f"Loaded: OmniVoice (LLM={type(m.llm).__name__})")

    # ── Freeze model: CHỈ train llm (LoRA) + audio_heads + audio_embeddings ──
    # QUAN TRỌNG (học từ bản cũ qlora_train_v100.py chạy tốt):
    #   ✅ train: LoRA trên m.llm + audio_heads + audio_embeddings (audio_specific.pt 65MB)
    #   ❌ KHÔNG train audio_tokenizer.* (quantizer codebook, acoustic encoder/decoder)
    #      — tokenizer là thành phần cố định, train nó → hỏng → audio NaN
    for name, param in m.named_parameters():
        if "audio_tokenizer" in name:
            param.requires_grad = False          # tokenizer GIỮ NGUYÊN
        elif "llm" not in name and "audio_" not in name:
            param.requires_grad = False          # phần khác freeze
        # còn lại (llm.* + audio_heads.* + audio_embeddings.*) → train

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
        lora_dropout=0.0,   # 0.05 → 0.0 (Unsloth: dropout không hữu ích cho TTS)
        bias="none",
        # FEATURE_EXTRACTION — m.llm là Qwen3Model thuần (không có lm_head),
        # CAUSAL_LM cần Qwen3ForCausalLM → crash. Fix nhiễu = KHÔNG freeze audio_*
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    m.llm = get_peft_model(m.llm, lora_config)
    m.llm.print_trainable_parameters()
    # Trainable: ~40M / 605M ≈ 6.6%

    # ── Load dataset ──
    logger.info("Loading dataset...")
    from datasets import load_dataset
    # Dataset local (HF parquet format: data/ + dataset_info.json) → load_dataset.
    # load_from_disk CHỈ dùng khi có dataset_dict.json (DatasetDict lưu disk cũ)
    if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
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
        save_steps=max(1, steps_per_epoch // 2),  # save 2 lần/epoch → test được nhiều điểm
        save_total_limit=6,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,          # eval 1 lần/epoch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        bf16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id="Teedyyy-rm/LoRa_Ngoc_Huyen_2.0",
        hub_strategy="checkpoint",
        hub_token=os.environ.get("HF_TOKEN"),
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_first_step=True,
        gradient_checkpointing=True,   # BẮT BUỘC với batch 16 (không bật = OOM ~22GB)
        optim="adamw_torch",
    )

    # ── Trainer ──
    trainer = Trainer(
        model=m,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        data_collator=DataCollatorForOmniVoice(),
        callbacks=[
            DetailedLogCallback(),
            PushAdapterOnSave(
                model=m,
                hub_model_id="Teedyyy-rm/LoRa_Ngoc_Huyen_2.0",
                hub_token=os.environ.get("HF_TOKEN"),
                output_dir=output_dir,
            ),
        ],
    )

    # ── Train ──
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # ── Save final LoRA (only m.llm adapter) ──
    final_path = os.path.join(output_dir, "final_lora")
    m.llm.save_pretrained(final_path)
    logger.info(f"Final LoRA saved to: {final_path}")

    # ── Save audio_specific.pt (CHỈ audio_heads + audio_embeddings) ──
    # QUAN TRỌNG: giống bản cũ — chỉ ~65MB (heads + embeddings).
    # KHÔNG lưu audio_tokenizer.* (giữ nguyên từ base, không train).
    torch.save(
        {n: p.detach().cpu() for n, p in m.named_parameters()
         if ("audio_" in n and "llm" not in n
             and "audio_tokenizer" not in n)},
        os.path.join(final_path, "audio_specific.pt"),
    )
    logger.info(f"audio_specific.pt saved (chỉ audio_heads + audio_embeddings)")

    # ── Also save config ──
    import json
    with open(os.path.join(final_path, "training_config.json"), "w") as f:
        json.dump({
            "base_model": base_model_path,
            "dataset": "Teedyyy-rm/Voice_Ngoc_Huyen",
            "voice_name": "Ngọc Huyền 2.0",
            "lora_rank": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules),
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
