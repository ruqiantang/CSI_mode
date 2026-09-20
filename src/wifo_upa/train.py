"""Training loop with sampled or sequential reconstruction tasks."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import nn
from torch.optim import AdamW

from .evaluate import masked_nmse
from .model import UPAMAE

DEFAULT_RATIOS = {
    "random": 0.85,
    "temporal": 0.50,
    "frequency": 0.50,
    "spatial": 0.25,
}


class Trainer:
    def __init__(
        self,
        model: UPAMAE,
        lr: float = 5e-4,
        weight_decay: float = 0.05,
        task_schedule: str = "sample",
        ratios: Mapping[str, float] | None = None,
        grad_clip: float | None = 1.0,
        device: torch.device | str | None = None,
    ) -> None:
        if task_schedule not in ("sample", "sequential"):
            raise ValueError("task_schedule must be 'sample' or 'sequential'")
        self.model = model
        self.task_schedule = task_schedule
        self.ratios = dict(DEFAULT_RATIOS if ratios is None else ratios)
        self.grad_clip = grad_clip
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.base_lr = lr
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._eval_task_index = 0

    def available_tasks(self) -> list[str]:
        tasks = ["random", "temporal", "frequency"]
        if self.model.config.allow_spatial_mask:
            tasks.append("spatial")
        return tasks

    def _choose_task(self) -> str:
        return random.choice(self.available_tasks())

    def _sample_spatial_type(self) -> str:
        return random.choice(self.model.config.spatial_types)

    def _next_eval_task(self) -> str:
        tasks = self.available_tasks()
        task = tasks[self._eval_task_index % len(tasks)]
        self._eval_task_index += 1
        return task

    def _run_one(
        self,
        H: torch.Tensor,
        task: str,
        training: bool,
    ) -> Dict[str, Any]:
        ratio = self.ratios[task]
        spatial_type = self._sample_spatial_type()
        with torch.set_grad_enabled(training):
            output = self.model(
                H,
                mask_type=task,
                ratio=ratio,
                spatial_type=spatial_type,
            )
        return output

    def train_epoch(
        self,
        loader: Iterable[torch.Tensor],
        epoch: int = 0,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            H = batch[0] if isinstance(batch, (tuple, list)) else batch
            if not isinstance(H, torch.Tensor):
                raise TypeError("DataLoader must yield a complex CSI tensor")
            H = H.to(self.device)
            if not torch.is_complex(H):
                raise TypeError("CSI batches must be complex tensors")

            tasks = (
                [self._choose_task()]
                if self.task_schedule == "sample"
                else self.available_tasks()
            )
            losses = []
            self.optimizer.zero_grad(set_to_none=True)
            for task in tasks:
                output = self._run_one(H, task, True)
                loss = output["loss"] / len(tasks)
                loss.backward()
                losses.append(float(output["loss"].detach().cpu()))

            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_loss = sum(losses) / len(losses)
            if not math.isfinite(batch_loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {batch_loss}")
            total_loss += batch_loss
            batches += 1

        if batches == 0:
            raise ValueError("training loader yielded no batches")
        return {"loss": total_loss / batches, "batches": float(batches)}

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[torch.Tensor],
    ) -> Dict[str, float]:
        self.model.eval()
        values = []
        for batch in loader:
            H = batch[0] if isinstance(batch, (tuple, list)) else batch
            H = H.to(self.device)
            task = self._next_eval_task()
            output = self._run_one(H, task, False)
            values.append(
                masked_nmse(
                    H,
                    output["prediction"],
                    output["mask"],
                    self.model.config.pt,
                    self.model.config.pf,
                )
            )
        if not values:
            raise ValueError("evaluation loader yielded no batches")
        return {"nmse": sum(values) / len(values)}

    def make_scheduler(
        self,
        total_steps: int,
        warmup_steps: int = 0,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        if total_steps <= 0 or warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError("invalid scheduler step counts")

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, schedule
        )
        return self.scheduler

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "epoch": epoch,
            "task_schedule": self.task_schedule,
            "ratios": self.ratios,
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> int:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.task_schedule = payload.get("task_schedule", self.task_schedule)
        self.ratios = payload.get("ratios", self.ratios)
        return int(payload["epoch"])
