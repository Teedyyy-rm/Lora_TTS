import torch
from transformers import TrainerCallback

class DetailedLogCallback(TrainerCallback):
    """Log GPU memory + loss mỗi step."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if logs is None:
            return
        
        # Add GPU info
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**3
            logs["gpu_mem_gb"] = round(mem, 2)
            logs["gpu_util_pct"] = round(
                torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0, 1
            )
        
        # Format dòng log rõ ràng
        loss = logs.get("loss", "?")
        lr = logs.get("learning_rate", "?")
        epoch = logs.get("epoch", "?")
        mem = logs.get("gpu_mem_gb", "?")
        step = state.global_step
        
        msg = (
            f"║ Step {step:>5d} | Loss: {loss:.4f}" if isinstance(loss, float) else
            f"║ Step {step:>5d} | Loss: {loss}"
        )
        if isinstance(lr, float):
            msg += f" | LR: {lr:.2e}"
        if isinstance(epoch, float):
            msg += f" | Epoch: {epoch:.2f}"
        if isinstance(mem, float):
            msg += f" | VRAM: {mem:.1f}GB"
        
        print(msg, flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        """Log mỗi step vào file riêng."""
        if state.global_step % args.logging_steps == 0:
            pass  # on_log handles this

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        print("╔══════════════════════════════════════════════════╗", flush=True)
        print("║          TRAINING STARTED                       ║", flush=True)
        print(f"║  Steps/epoch: {state.max_steps // args.num_train_epochs if args.num_train_epochs > 0 else '?'}", flush=True)
        print(f"║  Total steps: {state.max_steps}", flush=True)
        print(f"║  Batch: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} grad_accum", flush=True)
        print("╚══════════════════════════════════════════════════╝", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        print("╔══════════════════════════════════════════════════╗", flush=True)
        print(f"║  TRAINING COMPLETE - {state.global_step} steps        ║", flush=True)
        print("╚══════════════════════════════════════════════════╝", flush=True)
