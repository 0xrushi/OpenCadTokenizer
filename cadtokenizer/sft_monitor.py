"""TensorBoard system metrics for SFT: per-GPU utilization/memory and measured samples per second."""
from __future__ import annotations

from pathlib import Path
import subprocess
import time

from transformers import TrainerCallback


def gpu_samples():
    """[(index, utilization %, memory used MiB, memory total MiB)] from nvidia-smi; empty if unavailable."""
    try:
        text = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True).stdout
        return [tuple(int(float(v)) for v in line.split(',')) for line in text.strip().splitlines()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


class ThroughputMonitor(TrainerCallback):
    """Rank-0 only. Writes to <output>/tensorboard/system so it appears as its own run beside the Trainer's."""
    def __init__(self, output):
        self.directory = Path(output) / 'tensorboard' / 'system'
        self.writer = None
        self.mark = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            from torch.utils.tensorboard import SummaryWriter
            self.directory.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.directory))
            self.mark = (time.monotonic(), state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.writer is None or not logs or 'loss' not in logs:
            return
        now, step = time.monotonic(), state.global_step
        elapsed, steps = now - self.mark[0], step - self.mark[1]
        self.mark = (now, step)
        if elapsed > 0 and steps > 0:
            per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, args.world_size)
            self.writer.add_scalar('throughput/samples_per_second', per_step * steps / elapsed, step)
            self.writer.add_scalar('throughput/optimizer_steps_per_second', steps / elapsed, step)
        for index, utilization, used, total in gpu_samples():
            self.writer.add_scalar(f'gpu{index}/utilization_percent', utilization, step)
            self.writer.add_scalar(f'gpu{index}/memory_used_gib', used / 1024, step)
            self.writer.add_scalar(f'gpu{index}/memory_fraction', used / total, step)
        self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
