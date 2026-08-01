#!/usr/bin/env python3
"""sync_check.py — Kiểm tra đồng bộ 4 thành phần giữa V100 và local (qua HF trung gian).

Vì V100 ↔ local không trực tiếp kết nối (test thực tế qua HF), script này verify:
  1. BASE MODEL:  md5 local model.safetensors vs HF splendor1811/omnivoice-vietnamese
                  (V100 train trên base này → local test phải dùng ĐÚNG base này,
                   lệch weights = LoRA load lên bị NHIỄU)
  2. CODE:        git HEAD local vs GitHub remote (Lora_TTS = omnivoice-ngochuyen-lora)
  3. ADAPTER HF:  repo LoRa_Ngoc_Huyen_2.0 có adapters/checkpoint-N không
                  (test từng epoch được ngay) + final_lora có không
  4. VOICE PROMPT: md5 local vp1.pkl (phải cùng file V100 dùng)

Usage:
  python3 sync_check.py                 # chạy toàn bộ
  python3 sync_check.py --base-only     # chỉ check base model
  python3 sync_check.py --no-base       # bỏ check base (khi HF đang chậm)
Exit code: 0 = tất cả OK, 1 = có lệch.
"""
import argparse
import hashlib
import os
import subprocess
import sys

BASE_LOCAL = os.path.expanduser("~/projects/Models/OmniVoice-Vietnamese/model.safetensors")
BASE_HF = "splendor1811/omnivoice-vietnamese"
ADAPTER_REPO = "Teedyyy-rm/LoRa_Ngoc_Huyen_2.0"
CODE_REPO = os.path.expanduser("~/projects/Lora/Lora_TTS")
VOICE_PROMPT = os.path.expanduser("~/projects/StoryCast/assets/voices/Ngoc_Huyen/vp1.pkl")

FAIL = 0


def ok(msg):
    print(f"  ✅ {msg}")


def warn(msg):
    global FAIL
    FAIL = 1
    print(f"  ⚠️  {msg}")


def md5(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_base():
    print("─" * 60)
    print("1. BASE MODEL (local vs HF splendor1811/omnivoice-vietnamese)")
    if not os.path.exists(BASE_LOCAL):
        warn(f"KHÔNG có base local: {BASE_LOCAL}")
        return
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        if not token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
            token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        hf_path = hf_hub_download(BASE_HF, "model.safetensors", token=token)
        print(f"  Local: {os.path.getsize(BASE_LOCAL)/1e9:.2f}GB  "
              f"| HF: {os.path.getsize(hf_path)/1e9:.2f}GB")
        if os.path.getsize(BASE_LOCAL) == os.path.getsize(hf_path):
            ok("Kích thước giống nhau — hash nhanh tensors đầu/cuối")
            # hash nhanh: đọc block đầu + block cuối (không hash cả 2.4GB)
            def partial_hash(path):
                with open(path, "rb") as f:
                    head = f.read(1 << 20)
                    f.seek(-(1 << 20), 2)
                    tail = f.read(1 << 20)
                return hashlib.md5(head + tail).hexdigest()
            h1, h2 = partial_hash(BASE_LOCAL), partial_hash(hf_path)
            if h1 == h2:
                ok("Base model KHỚP (head+tail hash) — dùng chung base, LoRA sẽ load đúng")
            else:
                warn("Kích thước bằng nhưng nội dung khác (head+tail hash lệch) — "
                     "có thể khác version, LoRA có thể NHIỄU!")
        else:
            warn(f"Kích thước LỆCH ({os.path.getsize(BASE_LOCAL)/1e9:.2f}GB vs "
                 f"{os.path.getsize(hf_path)/1e9:.2f}GB) — KHÔNG cùng base, LoRA sẽ NHIỄU!")
    except Exception as e:
        warn(f"Không so được với HF: {e}")


def check_code():
    print("─" * 60)
    print("2. CODE (local HEAD vs GitHub remote)")
    if not os.path.isdir(os.path.join(CODE_REPO, ".git")):
        warn(f"KHÔNG có repo code: {CODE_REPO}")
        return
    r = subprocess.run(["git", "-C", CODE_REPO, "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    local_head = r.stdout.strip()
    r = subprocess.run(
        ["git", "ls-remote", "https://github.com/Teedyyy-rm/Lora_TTS.git", "HEAD"],
        capture_output=True, text=True)
    remote_head = r.stdout.split()[0] if r.stdout.strip() else ""
    print(f"  Local : {local_head[:10]}")
    print(f"  Remote: {remote_head[:10]}")
    if local_head == remote_head:
        ok("Code ĐỒNG BỘ — local = GitHub")
    else:
        warn(f"Code LỆCH — chạy: git -C {CODE_REPO} pull --ff-only")


def check_hf_adapter():
    print("─" * 60)
    print("3. ADAPTER TRÊN HF (LoRa_Ngoc_Huyen_2.0)")
    try:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
            token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        files = HfApi().list_repo_files(ADAPTER_REPO, repo_type="model", token=token)
        adapters = sorted([f for f in files if f.startswith("adapters/checkpoint-")])
        if adapters:
            ok(f"Có {len(adapters)} adapter giữa chừng (test từng epoch OK): "
               f"{[os.path.basename(a) for a in adapters[-3:]]}")
        else:
            warn("CHƯA có adapters/checkpoint-* — lần train tới (code mới) mới có")
        if any("final_lora" in f for f in files):
            ok("Có final_lora/ (adapter sau train xong)")
        else:
            warn("CHƯA có final_lora/ trên HF — chờ V100 push hoặc upload tay")
        full = [f for f in files if f.endswith("model.safetensors")]
        if full:
            print(f"  (full model checkpoint: {len(full)} file — nặng, không cần cho test)")
    except Exception as e:
        warn(f"Không list được HF repo: {e}")


def check_voice():
    print("─" * 60)
    print("4. VOICE PROMPT (vp1.pkl local)")
    if os.path.exists(VOICE_PROMPT):
        ok(f"vp1.pkl OK ({os.path.getsize(VOICE_PROMPT)/1024:.0f}KB, "
           f"md5 {md5(VOICE_PROMPT)[:10]}...)")
        print("  (V100 dùng cùng file này qua scp — test cùng voice prompt mới so được)")
    else:
        warn(f"KHÔNG có voice prompt: {VOICE_PROMPT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--no-base", action="store_true")
    args = ap.parse_args()

    print("═" * 60)
    print("  SYNC CHECK — V100 ↔ local (qua HF)")
    print("═" * 60)
    if not args.no_base:
        check_base()
    if not args.base_only:
        check_code()
        check_hf_adapter()
        check_voice()
    print("─" * 60)
    print("✅ TẤT CẢ ĐỒNG BỘ — sẵn sàng test" if FAIL == 0
          else "⚠️ CÓ LỆCH — xem chi tiết ở trên")
    sys.exit(FAIL)


if __name__ == "__main__":
    main()
