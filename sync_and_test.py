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
REPO_ID = "Teedyyy-rm/omnivoice-ngochuyen-lora-v2"
CACHE_DIR = os.path.expanduser("~/projects/Models/lora_v2_checkpoints")
STATE_FILE = os.path.expanduser("~/.cache/lora_v2_last_ts.txt")
OUTPUT_DIR = os.path.expanduser("~/projects/Models/lora_v2_tests")
VALIDATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_lora.py")

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
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=CACHE_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"✅ Đã tải về: {local}")

    # 4. Tìm checkpoint mới nhất (thư mục checkpoint-*)
    checkpoints = sorted(
        [d for d in os.listdir(local) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[-1]),
    )
    if not checkpoints:
        print("ℹ️  Không thấy thư mục checkpoint-* trong repo — bỏ qua", file=sys.stderr)
        return

    latest = os.path.join(local, checkpoints[-1])
    print(f"   Checkpoint mới nhất: {latest}")

    # 5. Chạy validate → sinh WAV test
    out_wav = os.path.join(OUTPUT_DIR, f"test_{checkpoints[-1]}.wav")
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
