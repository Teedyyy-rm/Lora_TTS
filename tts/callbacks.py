"""DetailedLogCallback — log tiến trình huấn luyện đẹp + VRAM.

Nguyên tắc:
- KHÔNG bao giờ sửa `logs` gốc của Trainer (dict đó được gửi lên
  tensorboard/HF hub — mutate nó gây loạn format như {}).
- Chỉ ĐỌC giá trị từ logs, in ra console 1 dòng gọn.
- Xử lý đúng kiểu dữ liệu: float, tensor, numpy, None.
- Giao diện V2 (Aug 2): đường ngăn cách ─── giữa các dòng log + banner
  ═══ khi đổi epoch + đánh dấu EVAL — dễ theo dõi log dài.
"""

import unicodedata

import torch
from transformers import TrainerCallback

SEP = "─" * 64          # ngăn cách giữa các dòng log
LINE = "═" * 64          # banner lớn (epoch / bắt đầu / kết thúc)
W = 58                   # chiều rộng nội dung giữa ║


def _fmt(value, spec):
    """Format an toàn: xử lý tensor/numpy/None, không crash."""
    if value is None:
        return "?"
    # Tensor / numpy scalar → chuyển về float
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return "?"
    if isinstance(value, float):
        try:
            return f"{value:{spec}}"
        except (ValueError, TypeError):
            return f"{value:.4f}"
    return str(value)


def _safe_float(value):
    """Trả về float hoặc None nếu không convert được."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _banner_row(text):
    """Một dòng trong banner ║...║ — padding động theo độ rộng ký tự (CJK=2 cell)."""
    width = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return f"║  {text}" + " " * max(1, W - 2 - width) + "║"


class DetailedLogCallback(TrainerCallback):
    """Log trạng thái VRAM, Loss, LR, Epoch — giao diện ngăn cách dễ theo dõi."""

    def __init__(self):
        self._last_epoch = -1  # để phát hiện đổi epoch (banner ═══)

    # ── BẮT ĐẦU TRAIN ──
    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._last_epoch = -1

        # Số steps/epoch — ưu tiên từ state/dataloader thật
        train_loader = kwargs.get("train_dataloader", None)
        if train_loader is not None:
            try:
                steps_per_epoch = len(train_loader) // max(args.gradient_accumulation_steps, 1)
            except Exception:
                steps_per_epoch = 0
        else:
            steps_per_epoch = 0

        if steps_per_epoch <= 0 and state.max_steps > 0:
            total_steps = state.max_steps
            steps_per_epoch = total_steps // max(args.num_train_epochs, 1)
        else:
            total_steps = steps_per_epoch * args.num_train_epochs

        print(f"╔{LINE}╗", flush=True)
        print(_banner_row("HUẤN LUYỆN OMNIVOICE - BẮT ĐẦU"), flush=True)
        print(_banner_row(f"Steps/Epoch : {steps_per_epoch}"), flush=True)
        print(_banner_row(f"Total Steps : {total_steps}"), flush=True)
        print(_banner_row(f"Batch       : {args.per_device_train_batch_size} x "
                          f"{args.gradient_accumulation_steps} grad-accum"), flush=True)
        print(_banner_row(f"Epochs      : {args.num_train_epochs}"), flush=True)
        print(_banner_row(f"Device      : {args.device}"), flush=True)
        print(f"╚{LINE}╝", flush=True)

    # ── LOG MỖI N STEPS ──
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if not logs:
            return

        step = state.global_step

        # ── Banner khi ĐỔI EPOCH (chỉ in 1 lần khi bước sang epoch mới) ──
        cur_epoch = _safe_float(logs.get("epoch"))
        if cur_epoch is not None and int(cur_epoch) > self._last_epoch:
            self._last_epoch = int(cur_epoch)
            total = getattr(args, "num_train_epochs", 0) or 0
            print(flush=True)
            print(f"╔{LINE}╗", flush=True)
            print(_banner_row(f"EPOCH {self._last_epoch + 1}/{int(total)}"), flush=True)
            print(f"╚{LINE}╝", flush=True)

        # ── Đường ngăn cách giữa các dòng log ──
        print(SEP, flush=True)

        loss = _fmt(logs.get("loss"), ".4f")
        lr = _fmt(logs.get("learning_rate"), ".2e")
        epoch = _fmt(logs.get("epoch"), ".2f")
        grad_norm = _fmt(logs.get("grad_norm"), ".3f")

        # VRAM — đọc trực tiếp, không nhét vào logs
        mem = ""
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            curr = torch.cuda.memory_allocated() / 1024**3
            mem = f"| VRAM {curr:.1f}/{peak:.1f}GB "

        # % hoàn thành
        pct = ""
        if getattr(state, "max_steps", 0) and state.max_steps > 0:
            pct = f"| {100.0 * step / state.max_steps:5.1f}%"

        print(
            f"[{step:>6d}] Loss {loss} | LR {lr} | Epoch {epoch} "
            f"{f'| GradNorm {grad_norm}' if grad_norm != '?' else ''} "
            f"{mem}{pct}",
            flush=True,
        )

    # ── TRƯỚC KHI EVAL ──
    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        print(SEP, flush=True)
        print(f"  ⚖️  EVAL @ step {state.global_step} ...", flush=True)
        print(SEP, flush=True)

    # ── HOÀN THÀNH ──
    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        print(flush=True)
        print(f"╔{LINE}╗", flush=True)
        print(_banner_row(f"HOÀN THÀNH - {state.global_step} steps"), flush=True)
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(_banner_row(f"VRAM Peak   : {peak:.2f} GB"), flush=True)
        print(f"╚{LINE}╝", flush=True)
