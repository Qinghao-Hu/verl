import logging
import os
from collections import deque
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import SmoothL1Loss
from torch.nn import functional as F
from transformers import AutoConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class EagleBackgroundTrainer:
    """FSDP2-capable background trainer for Eagle drafter model."""

    def __init__(self, rank: int, config: Optional[dict] = None):
        self.rank = rank
        self.config = config or {}

        # Runtime objects
        self.model = None
        self.optimizer = None
        self.training_device_mesh: Optional[DeviceMesh] = None

        # State
        self._training_initialized = False
        self._training_active = False
        self.training_steps = 0

        # Data buffer
        self.collected_data = deque(maxlen=int(self.config.get("buffer_max_samples", 2000)))
        self.batch_size = int(self.config.get("batch_size_per_gpu", 32))

        # Loss
        self.criterion = SmoothL1Loss(reduction="none")

        # Config/ckpt
        self.model_config = None
        self.eagle_model_path = self.config.get("eagle_model_path", self.config.get("spec_model_path"))
        self.checkpoint_dir = self.config.get("checkpoint_path")
        self._last_ckpt_step = -1

    def _get_model_class(self, model_type: str):
        if model_type.lower() == "llama":
            from verl.workers.drafter.model.llama_eagle import LlamaForCausalLMEagle

            return LlamaForCausalLMEagle
        if model_type.lower() == "qwen2":
            from verl.workers.drafter.model.qwen2_eagle import Qwen2ForCausalLMEagle

            return Qwen2ForCausalLMEagle
        raise ValueError(f"Unsupported model type: {model_type}")

    async def preinitialize_model(self) -> bool:  # API symmetry
        return True

    async def activate_training_model(self, device_mesh: DeviceMesh, training_ranks: list[int]) -> bool:
        if self._training_initialized:
            logger.info("Worker %s eagle already initialized", self.rank)
            return True

        base_model_path = self.config.get("base_model_path")
        if self.model_config is None:
            if not base_model_path:
                logger.error("EagleBackgroundTrainer needs base_model_path to initialize")
                return False
            cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
            cfg.num_hidden_layers = 1
            self.model_config = cfg

        model_cls = self._get_model_class(getattr(self.model_config, "model_type", "llama"))
        if self.eagle_model_path and os.path.exists(self.eagle_model_path):
            model = model_cls(config=self.model_config).cuda()
        else:
            model = model_cls(config=self.model_config).cuda()

        # Freeze shared parts
        for p in model.lm_head.parameters():
            p.requires_grad = False
        for p in model.model.embed_tokens.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if "lm_head" not in name and "embed_tokens" not in name:
                p.requires_grad_(True)

        if len(training_ranks) > 1:
            fully_shard(model, mesh=device_mesh)

        self.model = model
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.95)
        )
        self.training_device_mesh = device_mesh
        self._training_initialized = True
        self._training_active = True

        return True

    def collect_hidden_states(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        if not self._training_initialized:
            return
        # Support batched inputs
        if hidden_states.dim() == 3:
            for i in range(hidden_states.size(0)):
                self.collected_data.append(
                    {
                        "input_ids": input_ids[i].detach().cpu(),
                        "hidden_states": hidden_states[i].detach().cpu(),
                        "loss_mask": loss_mask[i].detach().cpu(),
                    }
                )
        else:
            self.collected_data.append(
                {
                    "input_ids": input_ids.detach().cpu(),
                    "hidden_states": hidden_states.detach().cpu(),
                    "loss_mask": loss_mask.detach().cpu(),
                }
            )

    def _prepare_training_batch(self) -> Optional[dict[str, torch.Tensor]]:
        if len(self.collected_data) < self.batch_size:
            return None
        items = list(self.collected_data)[: self.batch_size]
        pad_id = int(getattr(self.model_config, "pad_token_id", 0) or 0)
        max_len = max(x["input_ids"].numel() for x in items)
        hidden = items[0]["hidden_states"].size(-1)
        dev = next(self.model.parameters()).device
        bsz = len(items)

        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=dev)
        attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=dev)
        loss_mask = torch.zeros((bsz, max_len), dtype=torch.float32, device=dev)
        base_h = torch.zeros((bsz, max_len, hidden), dtype=torch.float32, device=dev)
        for i, it in enumerate(items):
            L = it["input_ids"].numel()
            input_ids[i, :L] = it["input_ids"].to(dev, non_blocking=True)
            attn_mask[i, :L] = 1
            loss_mask[i, :L] = it["loss_mask"].to(dev, non_blocking=True)
            bh = it["hidden_states"].to(dev, non_blocking=True)
            base_h[i, : bh.size(0)] = bh

        target = base_h[:, 1:].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "hidden_states": base_h,
            "target": target,
            "loss_mask": loss_mask,
        }

    async def training_step(self, step: int, num_released_workers: int) -> bool:
        with torch.enable_grad():
            return await self._training_step_impl(step)

    async def _training_step_impl(self, step: int) -> bool:
        if not self.model:
            return False
        batch = self._prepare_training_batch()
        if batch is None:
            return False

        if self.training_device_mesh is not None:
            dist.barrier(self.training_device_mesh.get_group())

        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            base_model_hidden_states=batch["hidden_states"],
            output_hidden_states=True,
        )
        hs = out.hidden_states[-1]
        logits = out.logits

        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            tgt_logits = self.model.lm_head(batch["target"])  # type: ignore[attr-defined]
            tgt_p = F.softmax(tgt_logits, dim=-1)

        m = batch["loss_mask"].unsqueeze(-1)
        ploss = -torch.sum(torch.sum(m * (tgt_p * logp[:, :-1]), dim=2)) / (m.shape[0] * m.shape[1])

        vloss = self.criterion(hs[:, :-1], batch["target"])  # (B, T-1, H)
        vloss = torch.sum(torch.mean(m * vloss, dim=2)) / (m.shape[0] * m.shape[1])

        w_v = float(self.config.get("vloss_weight", 0.5))
        w_p = float(self.config.get("ploss_weight", 0.5))
        loss = w_v * vloss + w_p * ploss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.training_steps += 1

        if self.checkpoint_dir and (step // 100) > self._last_ckpt_step:
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                ckpt = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": step,
                }
                torch.save(ckpt, os.path.join(self.checkpoint_dir, f"eagle_step_{step}.pth"))
                self._last_ckpt_step = step // 100
            except (OSError, RuntimeError) as e:  # noqa: BLE001
                logger.warning("Eagle checkpoint save failed on rank %s: %s", self.rank, e)

        return True

    def get_model_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        if not self.model:
            return None
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items() if v.requires_grad}

    async def cleanup_training(self):
        if self.checkpoint_dir and self.model is not None:
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                ckpt = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": self.training_steps,
                }
                torch.save(ckpt, os.path.join(self.checkpoint_dir, f"eagle_final_step_{self.training_steps}.pth"))
            except (OSError, RuntimeError) as e:  # noqa: BLE001
                logger.warning("Eagle final checkpoint save failed on rank %s: %s", self.rank, e)

        self.collected_data.clear()
        self.model = None
        self.optimizer = None
        self.training_device_mesh = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._training_initialized = False
        self._training_active = False
        self.training_steps = 0

    @property
    def is_training_initialized(self) -> bool:
        return self._training_initialized

    @property
    def is_training_active(self) -> bool:
        return self._training_active

    def set_training_active(self, active: bool):
        self._training_active = active