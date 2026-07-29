import torch
from transformers import TrainerCallback

class DetailedLogCallback(TrainerCallback):
    """Log trạng thái VRAM, Loss và Tiến trình huấn luyện trực quan."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if logs is None:
            return

        # Lấy lượng VRAM thực tế đang chiếm dụng trên V100
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024**3
            logs["gpu_mem_gb"] = round(mem, 2)

        # Bóc tách các thông số từ Trainer
        loss = logs.get("loss", "?")
        lr = logs.get("learning_rate", "?")
        epoch = logs.get("epoch", "?")
        mem = logs.get("gpu_mem_gb", "?")
        step = state.global_step

        # Xây dựng dòng log hiển thị mượt mà
        msg = (
            f"║ Step {step:>5d} | Loss: {loss:.4f}" if isinstance(loss, float) else
            f"║ Step {step:>5d} | Loss: {loss}"
        )
        if isinstance(lr, float):
            msg += f" | LR: {lr:.2e}"
        if isinstance(epoch, float):
            msg += f" | Epoch: {epoch:.2f}"
        if isinstance(mem, float):
            msg += f" | VRAM Peak: {mem:.1f}GB"

        print(msg, flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        # Tính toán số step/epoch an toàn từ dataloader thay vì state.max_steps
        train_loader = kwargs.get("train_dataloader", None)
        if train_loader is not None:
            steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
            total_steps = steps_per_epoch * args.num_train_epochs
        else:
            steps_per_epoch = 471  # Fallback: 7540 / (4*4)
            total_steps = steps_per_epoch * args.num_train_epochs

        print("╔══════════════════════════════════════════════════╗", flush=True)
        print("║          HUẤN LUYỆN OMNIVOICE KÍCH HOẠT          ║", flush=True)
        print(f"║  Steps/Epoch: {steps_per_epoch:<34d}║", flush=True)
        print(f"║  Tổng số Steps: {total_steps:<32d}║", flush=True)
        print(f"║  Cấu hình Batch: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} Grad Accum           ║", flush=True)
        print("╚══════════════════════════════════════════════════╝", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        print("╔══════════════════════════════════════════════════╗", flush=True)
        print(f"║  HOÀN THÀNH HUẤN LUYỆN - {state.global_step:>5d} steps             ║", flush=True)
        print("╚══════════════════════════════════════════════════╝", flush=True)
