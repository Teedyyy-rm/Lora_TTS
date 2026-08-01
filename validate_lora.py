"""
Local Validation: Load LoRA adapter + OmniVoice-Vietnamese
---------------------------------------------------------
Usage: python validate_lora.py --lora_path <path_to_lora>

Tests the finetuned LoRA by generating TTS with voice clone prompt.
"""

import os
import sys
import torch
import pickle
import soundfile as sf
import argparse

sys.path.insert(0, "/home/obito/projects/StoryCast")

import omnivoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA adapter (e.g., ./final_lora or HF repo)")
    parser.add_argument("--base_model", type=str,
                        default="/home/obito/projects/Models/OmniVoice-Vietnamese",
                        help="Base OmniVoice-Vietnamese model path")
    parser.add_argument("--voice_prompt", type=str,
                        default="/home/obito/projects/StoryCast/assets/voices/Ngoc_Huyen/vp1.pkl",
                        help="Voice clone prompt (.pkl)")
    parser.add_argument("--text", type=str,
                        default="Xin chào, tôi là Ngọc Huyền. Rất vui được gặp bạn!",
                        help="Text to generate")
    parser.add_argument("--output", type=str, default="output/validation.wav",
                        help="Output WAV path")
    args = parser.parse_args()

    # ── Load base OmniVoice model ──
    print(f"Loading base model from: {args.base_model}")
    m = omnivoice.OmniVoice.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    # CHUYỂN LÊN CUDA TRƯỚC: Đảm bảo PEFT khi nạp vào sẽ map đúng địa chỉ GPU
    m = m.to("cuda")
    if hasattr(m, "audio_tokenizer"):
        m.audio_tokenizer = m.audio_tokenizer.to("cpu")  # Tiết kiệm VRAM nền

    # ── Apply LoRA to m.llm (Qwen3Model) ──
    if os.path.exists(args.lora_path) or "/" in args.lora_path:
        print(f"Loading LoRA from: {args.lora_path}")

        # ⚠️ SANITIZE adapter_config.json trước khi load (peft V100 0.20 viết 40+ keys
        # mới mà peft máy cá nhân 0.12 không hiểu → LoraConfig crash). Chỉ giữ whitelist.
        cfg_path = os.path.join(args.lora_path, "adapter_config.json")
        if os.path.exists(cfg_path):
            import json as _json
            raw_cfg = _json.load(open(cfg_path))
            whitelist = {
                "peft_type", "r", "lora_alpha", "lora_dropout", "bias",
                "task_type", "target_modules", "base_model_name_or_path",
                "fan_in_fan_out", "init_lora_weights", "use_rslora",
            }
            clean_cfg = {k: raw_cfg[k] for k in whitelist if k in raw_cfg}
            if len(clean_cfg) != len(raw_cfg):
                with open(cfg_path, "w") as fp:
                    _json.dump(clean_cfg, fp, indent=2)
                print(f"  ⚠️ adapter_config.json sanitized: {len(raw_cfg)} → "
                      f"{len(clean_cfg)} keys (peft V100 0.20 → local 0.12)")

        # Vá lỗi thiếu hàm sinh từ của lớp backbone gốc
        if not hasattr(m.llm, "prepare_inputs_for_generation"):
            m.llm.prepare_inputs_for_generation = lambda **kwargs: kwargs

        # Nạp trực tiếp vào m.llm đã ở trên CUDA
        m.llm = PeftModel.from_pretrained(m.llm, args.lora_path)
        print(f"✅ LoRA loaded: {args.lora_path}")

    # ── Load audio_specific.pt (audio_heads + audio_embeddings — BẮT BUỘC) ──
    # LoRA finetune v2 train CẢ audio_* cùng (fix nhiễu gốc) → khi test PHẢI load
    # audio_specific.pt nếu có trong thư mục adapter, nếu không giọng sẽ NHIỄU
    # (LoRA đổi hidden_states → audio_heads base cũ không khớp).
    audio_specific_path = os.path.join(args.lora_path, "audio_specific.pt")
    if os.path.exists(audio_specific_path):
        audio_sd = torch.load(audio_specific_path, map_location="cpu", weights_only=True)
        missing, unexpected = m.load_state_dict(audio_sd, strict=False)
        print(f"✅ audio_specific.pt loaded ({len(audio_sd)} keys, "
              f"missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("⚠️ KHÔNG có audio_specific.pt trong adapter — nếu model train audio_* "
              "thì giọng sẽ NHIỄU (thiếu heads/embeddings đã train)!")


    torch.cuda.empty_cache()

    # ── Load voice prompt ──
    with open(args.voice_prompt, "rb") as f:
        vp = pickle.load(f)
    print(f"✅ Voice prompt: {os.path.basename(args.voice_prompt)} ({vp.ref_audio_tokens.shape[1]} frames)")

    # ── Generate ──
    gc = OmniVoiceGenerationConfig(
        num_step=48, guidance_scale=2.0,
        denoise=True, preprocess_prompt=True, postprocess_output=True,
    )

    print(f'Generating: "{args.text}"')
    with torch.inference_mode():
        audio = m.generate(
            text=args.text,
            language="vi",
            speed=1.0,
            generation_config=gc,
            voice_clone_prompt=vp,
        )
    if isinstance(audio, list):
        audio = audio[0]

    # SỬA LỖI THƯ MỤC: Tự động check và tạo folder output nếu chưa có
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sf.write(args.output, audio, 24000, format="WAV")
    print(f"✅ Saved: {args.output} ({len(audio) / 24000:.1f}s)")


if __name__ == "__main__":
    main()
