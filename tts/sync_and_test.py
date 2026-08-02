#!/usr/bin/env python3
"""
sync_and_test.py — Tự động tải checkpoint mới nhất từ HF + test trên máy cá nhân.

Luồng:
1. Kiểm tra HF repo `Teedyyy-rm/omnivoice-ngochuyen-lora-v2` có checkpoint mới?
   (so sánh last_modified — mỗi lần train push checkpoint là thay đổi)
2. Nếu có → snapshot_download về máy cá nhân (thư mục cached)
3. Chạy validate_lora.py với checkpoint mới nhất → sinh WAV test
4. In kết quả (để cron gửi về Discord)

Cách dùng (cron mỗi 15 phút):
    python3 sync_and_test.py
    python3 sync_and_test.py --text "Câu test..." --check-every 900
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

# ── Config ──
REPO_ID = "Teedyyy-rm/LoRa_Ngoc_Huyen_2.0"   # PRIVATE — Ngọc Huyền 2.0
CACHE_DIR = os.path.expanduser("~/projects/Models/lora_2.0_checkpoints")
STATE_FILE = os.path.expanduser("~/.cache/lora_2.0_last_ts.txt")
OUTPUT_DIR = os.path.expanduser("~/projects/Models/lora_2.0_tests")
VALIDATE = os.path.expanduser("~/projects/Models/lora_v2_validate.py")

# Câu test mặc định — câu dài + tên riêng (kiểm tra nuốt chữ)
DEFAULT_TEXT = (
    "Thẩm Hữu Vi bước vào Thanh Vân Tông, gặp Lục Vân Giao đang tu luyện. "
    "Ngọc Huyền kể chuyện Phi Thăng Thất Bại, xin mời các bạn cùng nghe."
)


def get_repo_mtime() -> float:
    """Lấy thời điểm sửa đổi gần nhất của repo HF (số giây epoch)."""
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        info = api.model_info(REPO_ID)
        # lastModified của repo
        return info.last_modified.timestamp() if info.last_modified else 0.0
    except Exception as e:
        print(f"ℹ️  Chưa có checkpoint trên HF ({e}) — bỏ qua", file=sys.stderr)
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Câu test")
    parser.add_argument("--force", action="store_true",
                        help="Test lại kể cả không có checkpoint mới")
    parser.add_argument("--check-every", type=float, default=900,
                        help="Khoảng tối thiểu giữa 2 lần test (giây, mặc định 900)")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Kiểm tra có checkpoint mới không
    repo_mtime = get_repo_mtime()
    if repo_mtime == 0.0:
        return  # repo chưa tồn tại (chưa train) — im lặng

    last_ts = 0.0
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            last_ts = float(f.read().strip() or 0)

    # Giới hạn tần suất: không test quá 1 lần/check_every giây
    if not args.force and last_ts > 0 and (time.time() - last_ts) < args.check_every:
        return  # chưa tới lúc — im lặng

    # 2. Checkpoint MỚI hơn lần test trước?
    if not args.force and repo_mtime <= last_ts:
        return  # không có gì mới — im lặng

    # 3. Tải checkpoint mới nhất về máy cá nhân
    print(f"📥 Checkpoint mới trên HF (mtime={repo_mtime:.0f} > last={last_ts:.0f})")
    print(f"   Đang tải về: {CACHE_DIR} ...")
    from huggingface_hub import snapshot_download
    # KHÔNG tải training state (vô dụng cho test) — nhưng PHẢI GIỮ audio_specific.pt!
    # ⚠️ Đừng dùng ignore "*.pt" chung — nó sẽ bỏ cả audio_specific.pt (~65MB,
    #    audio_heads+embeddings đã train, THIẾU nó = giọng nhiễu). Chỉ ignore file cụ thể.
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=CACHE_DIR,
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["optimizer.pt", "scheduler.pt", "scaler.pt",
                         "rng_state.pth", "runs/*", "*.bin", "training_args.bin"],
    )
    print(f"✅ Đã tải về: {local}")

    # 4. Tìm checkpoint: ưu tiên adapters/checkpoint-* (adapter chuẩn + audio_specific.pt
    #    do PushAdapterOnSave push mỗi epoch — chỉ ~230MB, test được NGAY giữa chừng),
    #    rồi final_lora (adapter sau train xong), cuối cùng checkpoint-* (full model 2.2GB)
    checkpoints = []
    adapters_dir = os.path.join(local, "adapters")
    if os.path.isdir(adapters_dir):
        checkpoints += sorted(
            [os.path.join(adapters_dir, d) for d in os.listdir(adapters_dir)
             if d.startswith("checkpoint-")],
            key=lambda d: int(os.path.basename(d).split("-")[-1]),
        )
    final_lora = os.path.join(local, "final_lora")
    if os.path.isdir(final_lora) and os.path.exists(
        os.path.join(final_lora, "adapter_config.json")
    ):
        checkpoints.append(final_lora)
    checkpoints += sorted(
        [os.path.join(local, d) for d in os.listdir(local)
         if d.startswith("checkpoint-")],
        key=lambda d: int(os.path.basename(d).split("-")[-1]),
    )
    if not checkpoints:
        print("ℹ️  Chưa có checkpoint-* hoặc final_lora trong repo — bỏ qua",
              file=sys.stderr)
        return

    # Chọn bản TỐT NHẤT theo thứ tự ưu tiên (tránh tải full model 2.2GB khi có adapter):
    # 1) final_lora (adapter chuẩn sau train xong — best theo eval_loss)
    # 2) adapters/checkpoint-N mới nhất (đang train — test từng epoch, chỉ ~230MB)
    # 3) checkpoint-N full model (fallback cuối — nặng 2.2GB, cần trích LoRA keys tay)
    if final_lora in checkpoints:
        latest = final_lora
    else:
        adapter_ckpts = [c for c in checkpoints if "/adapters/" in c]
        latest = adapter_ckpts[-1] if adapter_ckpts else checkpoints[-1]
    print(f"   Checkpoint dùng: {latest}")

    # 5. Chạy validate → sinh WAV test
    tag = os.path.basename(latest)  # checkpoint-XXX hoặc final_lora
    out_wav = os.path.join(OUTPUT_DIR, f"test_{tag}.wav")
    cmd = [
        sys.executable, VALIDATE,
        "--lora_path", latest,
        "--text", args.text,
        "--output", out_wav,
    ]
    print(f"🎙️  Test: {args.text[:60]}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Test thất bại:\n{result.stderr[-800:]}")
        return
    print(result.stdout[-500:])

    # 6. Ghi state (chỉ khi thành công)
    with open(STATE_FILE, "w") as f:
        f.write(f"{repo_mtime:.0f}")

    print(f"\n🎉 KẾT QUẢ: {out_wav}")
    print(f"   File: {out_wav} ({os.path.getsize(out_wav)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
