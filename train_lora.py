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

# ── Chọn dataset: --dataset old|new (Aug 2) ──
#   new = Teedyyy-rm/Voice_Ngoc_Huyen   (14,937 mẫu, ~20.78h)
#   old = pnnbao-ump/ngochuyen_voice    (7,540 mẫu,  ~20h)
# ⚠️ load_best_model_at_end=False (bài học Aug 2: eval_loss chọn NHẦM checkpoint
#    overfit — chọn bản cuối bằng NGHE/phổ, PushAdapterOnSave đã push từng
#    adapter_step lên HF để test tay).
import argparse
import os

DATASET_CFG = {
    # new: dataset mới, 3 epochs (bài học: epoch 4 overfit trên data lớn)
    "new": {
        "repo": "Teedyyy-rm/Voice_Ngoc_Huyen",
        "dataset_path": "./dataset/ngochuyen_voice",
        "output_dir": "./omnivoice_ngochuyen_lora_2.0",
        "hub_model_id": "Teedyyy-rm/LoRa_Ngoc_Huyen_2.0",
        "num_epochs": 3,
        "val_ratio": 0.05,
        "batch_size": 8,
        "grad_accum": 4,        # effective 32
        "warmup_steps": 100,
        "save_per_epoch": 2,    # save + push 2 lần/epoch (test nhiều điểm)
    },
    # old: dataset cũ, cấu hình THEO checkpoint cũ (Teedyyy-rm/omnivoice-ngochuyen-lora
    # last-checkpoint: 471/4720 steps, epoch 1/10, batch 4×4=16, warmup 200, save 1/epoch)
    # để A/B so sánh chất lượng đúng điều kiện "học kĩ, thời gian lâu" như lần trước.
    # ⚠️ Aug 2: hub_model_id đổi sang LoRa_Ngoc_Huyen_2.0 (Teedyy yêu cầu) — sau khi
    #    fix root cause masking, adapter old sẽ push vào repo 2.0 (đã dọn sạch chỉ giữ README)
    "old": {
        "repo": "pnnbao-ump/ngochuyen_voice",
        "dataset_path": "./dataset/ngochuyen_voice_old",
        "output_dir": "./omnivoice_ngochuyen_lora_old",
        "hub_model_id": "Teedyyy-rm/LoRa_Ngoc_Huyen_2.0",
        "num_epochs": 10,
        "val_ratio": 0.05,
        "batch_size": 4,
        "grad_accum": 4,        # effective 16 (đúng checkpoint cũ)
        "warmup_steps": 200,
        "save_per_epoch": 1,    # save + push 1 lần/epoch (đúng 471/epoch cũ)
    },
    # combined: GỘP 2 dataset CÙNG GIỌNG (verify sim 0.873 Aug 2) — ~41h data.
    #   - new (14,937 mẫu truyện) + old (7,540 mẫu tin tức) = 22,477 mẫu
    #   - data LỚN → epochs giảm (5-6 đủ), batch 8×4=32, LR 2e-5 (đã kiểm chứng)
    #   - mask_beta + audio_lr: nghiệm thức chống robot/vang
    "combined": {
        "repo": ["Teedyyy-rm/Voice_Ngoc_Huyen", "pnnbao-ump/ngochuyen_voice", "thangnzt/NgocHuyenViVoice"],
        "dataset_path": "./dataset/ngochuyen_combined",
        "output_dir": "./omnivoice_ngochuyen_lora_combined",
        "hub_model_id": "Teedyyy-rm/Omnivoice_Lora_v2",
        "num_epochs": 6,
        "val_ratio": 0.05,
        "batch_size": 8,
        "grad_accum": 4,        # effective 32
        "warmup_ratio": 0.03,   # 3% total steps
        "save_per_epoch": 2,    # save + push 2 lần/epoch
    },
}


def parse_args():
    ap = argparse.ArgumentParser(description="Finetune LoRA OmniVoice (Ngọc Huyền)")
    ap.add_argument("--dataset", choices=list(DATASET_CFG.keys()), default=None,
                    help="new=Voice_Ngoc_Huyen 14.9k mẫu (3 epochs, batch 8×4) | "
                         "old=ngochuyen_voice 7.5k mẫu (10 epochs, batch 4×4 — theo checkpoint cũ). "
                         "Nếu bỏ trống → lấy từ config.yaml (nếu có), nếu không → 'new'")
    ap.add_argument("--config", default="config.yaml",
                    help="Path config.yaml (mặc định ./config.yaml) — bỏ trống để dùng default code")
    return ap.parse_args()


def load_yaml_config(path="config.yaml"):
    """Đọc config.yaml nếu tồn tại → dict. Không có file → {} (dùng default code).

    Ưu tiên: CLI arg (--dataset) > config.yaml > default trong DATASET_CFG/code.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("⚠️ PyYAML chưa cài — bỏ qua config.yaml, dùng default code")
        return {}
    except Exception as e:
        print(f"⚠️ Lỗi đọc {path}: {e} — dùng default code")
        return {}


def _coerce_cfg(cfg):
    """Ép kiểu an toàn cho config.yaml — PyYAML parse '2e-5' thành str (bug gặp Aug 2:
    TypeError: '<=' not supported between instances of 'float' and 'str' trong AdamW).
    Int fields → int, float fields → float, còn lại giữ nguyên (str/bool/list)."""
    INT_FIELDS = {"num_epochs", "batch_size", "grad_accum", "warmup_steps",
                  "save_per_epoch", "lora_rank", "lora_alpha", "logging_steps",
                  "save_total_limit", "seed"}
    FLOAT_FIELDS = {"val_ratio", "learning_rate", "lora_dropout", "rms_normalize",
                    "drop_cond_ratio", "audio_mask_id", "audio_lr", "warmup_ratio"}
    for k, v in list(cfg.items()):
        if v is None:
            continue
        if k in INT_FIELDS:
            try:
                cfg[k] = int(v)
            except (TypeError, ValueError):
                pass
        elif k in FLOAT_FIELDS:
            try:
                cfg[k] = float(v)
            except (TypeError, ValueError):
                pass
    return cfg


def resolve_config(args):
    """Kết hợp: default DATASET_CFG ← config.yaml ← CLI args."""
    ycfg = load_yaml_config(args.config)

    # 1. Chọn dataset: CLI > config.yaml > 'new'
    dataset = args.dataset or ycfg.get("dataset", "new")
    if dataset not in DATASET_CFG:
        print(f"⚠️ dataset '{dataset}' không hợp lệ — dùng 'new'")
        dataset = "new"

    # 2. Merge config.yaml datasets.<name> lên default
    cfg = dict(DATASET_CFG[dataset])
    yds = ycfg.get("datasets", {}).get(dataset, {})
    cfg.update({k: v for k, v in yds.items() if v is not None})

    # 3. Config chung training.* (ghi đè nếu có)
    ytr = ycfg.get("training", {})
    cfg["training"] = {k: v for k, v in ytr.items() if v is not None}

    # 4. Preprocess config (ghi đè nếu có)
    ypp = ycfg.get("preprocess", {})
    cfg["preprocess"] = {k: v for k, v in ypp.items() if v is not None}

    # 5. Ép kiểu số (fix PyYAML str '2e-5' → float)
    cfg = _coerce_cfg(cfg)
    cfg["training"] = _coerce_cfg(cfg.get("training", {}))
    cfg["preprocess"] = _coerce_cfg(cfg.get("preprocess", {}))

    return dataset, cfg


# ── Hyperparameters V3 (tối ưu CHẤT LƯỢNG — Aug 2, V100 mới 16GB) ──
# Ưu tiên chất lượng giọng, không cần nhanh:
#   - epochs theo dataset (3 cho 14.9k mẫu, 6 cho 7.5k mẫu — data lớn thì ít epochs,
#     epoch 4 từng overfit trên data mới: eval_loss thấp nhưng nhiễu 25%)
#   - batch 8 × 4 accum = effective 32 (⚠️ KHÔNG dùng batch 16: OmniVoice KHÔNG hỗ
#     trợ gradient_checkpointing — batch 16 không checkpointing = ~22GB OOM. Batch 8
#     không checkpointing = 10.9GB/16GB — đã verify trên V100 cũ)
#   - dropout 0.0 (Unsloth: LoRA dropout không hữu ích cho TTS, 0 = consistency)
# ⚠️ BATCH_SIZE/GRAD_ACCUM/WARMUP thực tế lấy từ DATASET_CFG trong main() — 2 dòng
#    dưới chỉ là default cho module-level (không còn dùng trực tiếp).


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


def preprocess_dataset(dataset, audio_tokenizer, text_tokenizer, pcfg=None):
    """Convert raw audio + text to OmniVoice input format (on CPU).

    Dataset features: audio (array), transcription (str), file_name (str)
    pcfg: dict từ config.yaml preprocess.* (drop_cond_ratio, prompt_ratio_range,
          mask_ratio_range, audio_mask_id, language, instruct) — mặc định theo chuẩn.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed = []
    from tqdm import tqdm

    # ── Preprocess config (từ config.yaml hoặc mặc định chuẩn OmniVoice) ──
    pcfg = pcfg or {}
    drop_cond_ratio = float(pcfg.get("drop_cond_ratio", 0.1))
    prompt_range = pcfg.get("prompt_ratio_range", [0.0, 0.3])
    mask_range = pcfg.get("mask_ratio_range", [0.0, 1.0])
    audio_mask_id = int(pcfg.get("audio_mask_id", 1024))
    language = pcfg.get("language", "vi")
    instruct = pcfg.get("instruct", "None")
    rms_norm = float(pcfg.get("rms_normalize", 0.13))

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
        # text: linh hoạt column — 'transcription' (chuẩn) hoặc 'text' (thangnzt)
        text = sample.get("transcription") or sample.get("text", "")

        # Convert audio to tensor
        audio_t = torch.from_numpy(audio_array).float()
        if sr != 24000:
            import torchaudio.functional as F
            audio_t = F.resample(audio_t, sr, 24000)

        # Normalize RMS
        rms = torch.sqrt(torch.mean(audio_t ** 2))
        if rms > 0:
            audio_t = audio_t * (rms_norm / rms)

        # Encode audio → tokens: [1, 8, time] — encode on CPU, avoid VRAM leak
        with torch.no_grad():
            audio_input = audio_t.unsqueeze(0).unsqueeze(0).to(device)
            enc = audio_tokenizer.encode(audio_input)
            audio_tokens = enc.audio_codes[0]  # [8, time]

        import random

        # CFG (Classifier-Free Guidance) Dropout: Rất quan trọng để giảm nhiễu và vấp câu!
        # Dành drop_cond_ratio (mặc định 10%) tỷ lệ bắt mô hình học sinh âm thanh mà
        # không có text/prompt (unconditional). Nhờ vậy, khi test với guidance_scale=2.0,
        # tín hiệu text sẽ được kích mạnh lên chuẩn xác.
        drop_cond = random.random() < drop_cond_ratio

        if drop_cond:
            text_tokens = []
            prompt_ratio = 0.0
        else:
            # 1. Định dạng văn bản đúng chuẩn OmniVoice (kèm các tag đặc biệt)
            full_text = f"<|lang_start|>{language}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|><|text_start|>{text}<|text_end|>"
            text_tokens = text_tokenizer.encode(full_text, add_special_tokens=False)
            # Giữ lại một phần ngẫu nhiên làm prompt (0% - 30%)
            prompt_ratio = random.uniform(*prompt_range)

        num_text = len(text_tokens)
        num_audio = audio_tokens.shape[1]
        seq_len = num_text + num_audio

        # 2. Audio Masking Logic (Flow-Matching)
        # ⚠️ NGHIỆM THỨC (Aug 2 — chống "robot/vấp từ"): inference thật = mask cao
        #    (sinh toàn bộ audio từ text). Nếu mask_ratio_range=[0,1] uniform →
        #    model ít được train ở chế độ khó/đúng inference. Dùng Beta(2,1)
        #    (mean 0.67, thiên cao) khi config mask_beta=true.
        if pcfg.get("mask_beta", False):
            mask_ratio = float(np.random.beta(2, 1))
        else:
            mask_ratio = random.uniform(*mask_range)
        
        prompt_length = int(num_audio * prompt_ratio)
        audio_inputs = audio_tokens.clone()
        audio_labels = audio_tokens.clone()

        # Tạo mask ngẫu nhiên cho vùng maskable (sau phần prompt_length)
        maskable_region = audio_inputs[:, prompt_length:]
        token_mask = torch.rand(maskable_region.shape) < mask_ratio
        
        # Gán token bị che thành audio_mask_id (1024)
        audio_inputs[:, prompt_length:][token_mask] = audio_mask_id
        
        # Labels: Chỉ tính Loss trên các token BỊ CHE, các vùng khác gán -100
        audio_labels[:, prompt_length:][~token_mask] = -100
        audio_labels[:, :prompt_length] = -100

        # 3. Build input_ids: [8, seq_len]
        input_ids = torch.zeros((8, seq_len), dtype=torch.long)
        if num_text > 0:
            input_ids[:, :num_text] = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0).repeat(8, 1)
        input_ids[:, num_text:] = audio_inputs

        # 4. audio_mask: 0=text, 1=audio (để model biết áp dụng embedding nào)
        audio_mask = torch.zeros(seq_len, dtype=torch.bool)
        audio_mask[num_text:] = True

        # 5. labels: text không tính loss (-100), chỉ audio_labels tính loss
        labels = torch.zeros((8, seq_len), dtype=torch.long)
        if num_text > 0:
            labels[:, :num_text] = -100
        labels[:, num_text:] = audio_labels

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

            # 3. ⚠️ XÓA README.md PEFT tự tạo (frontmatter không hợp lệ → upload_folder
            #    fail "Invalid metadata in README.md"). Chỉ upload adapter + config + audio.
            readme = os.path.join(adapter_dir, "README.md")
            if os.path.exists(readme):
                os.remove(readme)

            # 4. Push lên HF: adapters/checkpoint-{step}/
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
            print(f"────────────────────────────────────────────────────────────────",
                  flush=True)
            print(f"🚀 [PushAdapterOnSave] step {step}: adapter ({size_mb:.0f}MB) "
                  f"+ audio_specific.pt → adapters/checkpoint-{step} trên HF",
                  flush=True)
            print(f"────────────────────────────────────────────────────────────────",
                  flush=True)
        except Exception as e:
            print(f"⚠️ [PushAdapterOnSave] step {step} push lỗi (train vẫn tiếp tục): {e}",
                  flush=True)


def main():
    args = parse_args()
    dataset_name, cfg = resolve_config(args)
    NUM_EPOCHS = cfg["num_epochs"]
    VAL_RATIO = cfg["val_ratio"]
    BATCH_SIZE = cfg["batch_size"]
    GRAD_ACCUM = cfg["grad_accum"]
    WARMUP_STEPS = cfg.get("warmup_steps", 100)
    WARMUP_RATIO = cfg.get("warmup_ratio", 0.0)  # nếu >0: warmup = ratio × total_steps (chống méo giọng)
    SAVE_PER_EPOCH = cfg["save_per_epoch"]
    PREPROCESS_CFG = cfg.get("preprocess", {})
    TRAINING_CFG = cfg.get("training", {})
    logger.info(f"Dataset: {dataset_name} → {cfg['repo']} | {NUM_EPOCHS} epochs, "
                f"batch {BATCH_SIZE}×{GRAD_ACCUM} (effective {BATCH_SIZE*GRAD_ACCUM}), "
                f"warmup {WARMUP_STEPS}, save {SAVE_PER_EPOCH} lần/epoch")
    if PREPROCESS_CFG:
        logger.info(f"Preprocess config: {PREPROCESS_CFG}")

    # ── Paths (điều chỉnh cho V100 Docker) ──
    base_model_path = "./base_model/omnivoice-vietnamese"
    dataset_path = cfg["dataset_path"]         # theo --dataset
    output_dir = cfg["output_dir"]             # KHÔNG đè bản đang chạy
    hub_model_id = cfg["hub_model_id"]

    # ── Clone dataset từ HuggingFace (nếu chưa có) ──
    # ⚠️ combined: repo có thể là LIST (gộp 2 dataset) → clone MỖI repo vào
    #    subfolder riêng (<dataset_path>/<repo_name>) — tránh ghi đè lẫn nhau!
    repos = cfg["repo"] if isinstance(cfg["repo"], list) else [cfg["repo"]]
    need_clone = False
    if len(repos) > 1:
        # combined: kiểm tra từng subfolder
        for repo in repos:
            repo_name = repo.split("/")[-1]
            if not os.path.isdir(os.path.join(dataset_path, repo_name)):
                need_clone = True
                break
    else:
        need_clone = not os.path.isdir(dataset_path)

    if need_clone:
        for repo in repos:
            logger.info(f"Cloning dataset từ HuggingFace: {repo} ...")
            from huggingface_hub import snapshot_download
            dest = os.path.join(dataset_path, repo.split("/")[-1]) if len(repos) > 1 else dataset_path
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=dest,
                token=os.environ.get("HF_TOKEN"),
            )
        logger.info(f"✅ Dataset cloned → {dataset_path} ({len(repos)} repo)")

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
        r=TRAINING_CFG.get("lora_rank", 128),
        lora_alpha=TRAINING_CFG.get("lora_alpha", 256),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=TRAINING_CFG.get("lora_dropout", 0.0),  # Unsloth: dropout không hữu ích cho TTS
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
    # ⚠️ combined: load TỪNG repo (mỗi repo 1 subfolder) rồi concatenate
    if len(repos) > 1:
        datasets_list = []
        for repo in repos:
            repo_name = repo.split("/")[-1]
            sub_path = os.path.join(dataset_path, repo_name)
            ds = load_dataset(sub_path, token=os.environ.get("HF_TOKEN"))
            split = "train" if "train" in ds else list(ds.keys())[0]
            datasets_list.append(ds[split])
            logger.info(f"  + {repo}: {len(ds[split])} samples")
        from datasets import concatenate_datasets
        train_data = concatenate_datasets(datasets_list)
        logger.info(f"✅ COMBINED: {len(train_data)} samples ({len(repos)} datasets)")
    else:
        if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
            from datasets import load_from_disk
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
    processed = preprocess_dataset(train_data, audio_tokenizer, text_tokenizer,
                                   pcfg=PREPROCESS_CFG)
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
    # warmup_ratio (chống méo giọng): warmup = ratio × total (vd 0.03 × 4430 = 133 steps)
    if WARMUP_RATIO > 0:
        WARMUP_STEPS = max(1, int(total_steps * WARMUP_RATIO))
    logger.info(f"Steps/epoch: {steps_per_epoch}, total: {total_steps} "
                f"({NUM_EPOCHS} epochs)"
                f"(warmup {WARMUP_STEPS}{f' = {WARMUP_RATIO*100:.0f}%' if WARMUP_RATIO > 0 else ' steps'})")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=TRAINING_CFG.get("learning_rate", 2e-5),
        lr_scheduler_type=TRAINING_CFG.get("lr_scheduler_type", "cosine"),
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // SAVE_PER_EPOCH),  # theo cfg (1-2 lần/epoch)
        save_total_limit=10,
        eval_strategy="steps",
        eval_steps=max(1, steps_per_epoch // SAVE_PER_EPOCH),  # eval cùng tần suất save
        load_best_model_at_end=False,   # ⚠️ bài học Aug 2: eval_loss chọn NHẦM checkpoint overfit
                                        # → chọn bản cuối bằng NGHE/phổ (adapter_step đã push lên HF)
        metric_for_best_model="eval_loss",
        fp16=True,
        bf16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,           # ⚠️ Aug 2: TẮT — hub_strategy="checkpoint" push FULL MODEL
                                     # 3.5GB + optimizer 616MB + runs mỗi epoch → phình HF
                                     # (storage từng 56.6GB). PushAdapterOnSave (callback riêng,
                                     # dùng HfApi().upload_folder trực tiếp) tự push adapter nhẹ
                                     # 390MB lên adapters/checkpoint-N — KHÔNG phụ thuộc push_to_hub.
        hub_model_id=hub_model_id,
        hub_strategy="checkpoint",
        hub_token=os.environ.get("HF_TOKEN"),
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_first_step=True,
        gradient_checkpointing=False,   # OmniVoice KHÔNG hỗ trợ — batch 8 không cần
        optim="adamw_torch",
    )

    # ── Trainer ──
    # ⚠️ TÁCH LR (Aug 2 — nghiệm thức chống "giọng chỉ giống ~50%"): LoRA (llm.*)
    #    và audio_heads/embeddings (audio_*, full-rank) có tốc độ hội tụ KHÁC NHAU.
    #    audio_lr = 5× LoRA (mặc định 1e-4 vs 2e-5) — audio_* cần LR cao hơn để
    #    "áp" giọng đích vào, LoRA giữ LR thấp tránh overfit sớm.
    audio_lr = TRAINING_CFG.get("audio_lr", 0.0)  # 0 = KHÔNG tách (dùng LR chung)
    if audio_lr > 0:
        llm_params = [p for n, p in m.named_parameters()
                      if "llm" in n and p.requires_grad]
        audio_params = [p for n, p in m.named_parameters()
                        if "audio_" in n and "audio_tokenizer" not in n and p.requires_grad]
        lr_main = TRAINING_CFG.get("learning_rate", 2e-5)
        logger.info(f"🔀 TÁCH LR: LoRA(llm)={lr_main} | audio_*={audio_lr} "
                    f"(llm {len(llm_params)} tensors, audio {len(audio_params)} tensors)")
        optimizer = torch.optim.AdamW([
            {"params": llm_params, "lr": lr_main},
            {"params": audio_params, "lr": audio_lr},
        ])
        # Scheduler: Trainer tự tạo nếu truyền optimizer đơn — nhưng với custom
        # optimizer + lr_scheduler_type, dùng optimizers=(opt, None) → Trainer
        # tạo scheduler theo args (warmup + cosine trên LR chính)
        trainer = Trainer(
            model=m,
            args=training_args,
            train_dataset=train_processed,
            eval_dataset=val_processed,
            data_collator=DataCollatorForOmniVoice(),
            optimizers=(optimizer, None),
            callbacks=[
                DetailedLogCallback(),
            PushAdapterOnSave(
                model=m,
                hub_model_id=hub_model_id,
                hub_token=os.environ.get("HF_TOKEN"),
                output_dir=output_dir,
            ),
        ],
    )
    else:
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
                    hub_model_id=hub_model_id,
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
            "dataset": cfg["repo"],
            "voice_name": f"Ngọc Huyền ({args.dataset})",
            "lora_rank": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules),
            "task_type": "FEATURE_EXTRACTION",
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "val_ratio": VAL_RATIO,
            "eval_strategy": "steps",
            "load_best_model_at_end": training_args.load_best_model_at_end,
        }, f, indent=2)


if __name__ == "__main__":
    main()
