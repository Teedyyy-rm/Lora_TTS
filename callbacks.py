"""DetailedLogCallback — log tiến trình huấn luyện đẹp + VRAM.

Nguyên tắc:
- KHÔNG bao giờ sửa `logs` gốc của Trainer (dict đó được gửi lên
  tensorboard/HF hub — mutate nó gây loạn format như {}).
- Chỉ ĐỌC giá trị từ logs, in ra console 1 dòng gọn.
- Xử lý đúng kiểu dữ liệu: float, tensor, numpy, None.
"""

import unicodedata

import torch
from transformers import TrainerCallback


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


class DetailedLogCallback(TrainerCallback):
    """Log trạng thái VRAM, Loss, LR, Epoch — 1 dòng gọn, không phá logs gốc."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if not logs:
            return

        step = state.global_step

        # Chỉ đọc — không mutate logs
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

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

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

        line = "═" * 58
        W = 58  # chiều rộng nội dung giữa ║
        def row(text):
            # Độ dài thật: ký tự CJK/wide = 2 cell, còn lại 1 cell.
            # KHÔNG dùng emoji trong banner — width không nhất quán giữa terminal
            width = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1
                        for c in text)
            return f"║  {text}" + " " * max(1, W - 2 - width) + "║"

        print(f"╔{line}╗", flush=True)
        print(row("HUẤN LUYỆN OMNIVOICE - LOADING"), flush=True)
        print(row(f"Steps/Epoch : {steps_per_epoch}"), flush=True)
        print(row(f"Total Steps : {total_steps}"), flush=True)
        print(row(f"Batch       : {args.per_device_train_batch_size} x "
                  f"{args.gradient_accumulation_steps} grad-accum"), flush=True)
        print(row(f"Device      : {args.device}"), flush=True)
        print(f"╚{line}╝", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        line = "═" * 58
        W = 58
        def row(text):
            width = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1
                        for c in text)
            return f"║  {text}" + " " * max(1, W - 2 - width) + "║"

        print(f"╔{line}╗", flush=True)
        print(row(f"HOÀN THÀNH - {state.global_step} steps"), flush=True)
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(row(f"VRAM Peak   : {peak:.2f} GB"), flush=True)
        print(f"╚{line}╝", flush=True)
