#!/usr/bin/env python3
"""extract_adapter_from_checkpoint.py — Trích LoRA adapter chuẩn + audio_specific.pt
từ checkpoint full model của HF Trainer (last-checkpoint/model.safetensors).

Dùng để verify luồng test local khi HF chưa có adapters/ (checkpoint cũ train trước
khi có PushAdapterOnSave). Mô phỏng ĐÚNG những gì PushAdapterOnSave làm trên V100:
  1. Trích 392 LoRA keys (llm.base_model.model.*lora_A/B) → adapter_model.safetensors
  2. Trích audio_heads + audio_embeddings (audio_*, không llm, không tokenizer) → audio_specific.pt
  3. Viết adapter_config.json thủ công (Trainer không tạo)

Usage:
  python3 extract_adapter_from_checkpoint.py <ckpt_dir> <out_dir>
  # ckpt_dir: thư mục chứa model.safetensors (vd .../last-checkpoint)
  # out_dir:  nơi ghi adapter_model.safetensors + adapter_config.json + audio_specific.pt
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LORA_CONFIG = {
    "base_model_name_or_path": "splendor1811/omnivoice-vietnamese",
    "peft_type": "LORA",
    "task_type": "FEATURE_EXTRACTION",
    "r": 128,
    "lora_alpha": 256,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ckpt_dir, out_dir = sys.argv[1], sys.argv[2]
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"❌ Không thấy {safetensors_path}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    lora_sd, audio_sd = {}, {}
    n_base = 0
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            if "lora_A" in k or "lora_B" in k:
                # llm.base_model.model.layers.N...lora_A.default.weight
                # → bỏ prefix "llm." → base_model.model.layers.N... (PEFT format)
                lora_sd[k.replace("llm.", "", 1)] = f.get_tensor(k)
            elif "audio_" in k and "audio_tokenizer" not in k:
                audio_sd[k] = f.get_tensor(k)
            else:
                n_base += 1

    # 1. LoRA adapter
    adapter_path = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(lora_sd, adapter_path)
    # 2. audio_specific.pt
    torch.save(audio_sd, os.path.join(out_dir, "audio_specific.pt"))
    # 3. adapter_config.json
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as fp:
        json.dump(LORA_CONFIG, fp, indent=2)

    print(f"✅ LoRA keys: {len(lora_sd)} → {adapter_path} "
          f"({os.path.getsize(adapter_path)/1e6:.0f}MB)")
    print(f"✅ audio_* keys: {len(audio_sd)} → audio_specific.pt "
          f"({os.path.getsize(os.path.join(out_dir, 'audio_specific.pt'))/1e6:.0f}MB)")
    print(f"✅ adapter_config.json (r=128, alpha=256, FEATURE_EXTRACTION)")
    print(f"   base keys bỏ qua: {n_base}")


if __name__ == "__main__":
    main()
