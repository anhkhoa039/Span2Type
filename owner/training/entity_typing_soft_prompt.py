"""Trainer for soft-prompt Entity Typing with two-stage prompt-initialized fine-tuning.

Stage 1 — Soft Prompt Warmup (epochs 1 .. unfreeze_epoch-1):
  PLM is strictly frozen (requires_grad=False).  The AdamW optimizer tracks
  only the tiny soft_prompt tensor, so no state is allocated for the 110 M PLM
  params.  A high LR (default 3e-3) lets the prompt converge quickly.

Stage 2 — Full Fine-Tuning (epochs unfreeze_epoch .. num_epochs):
  At the *start* of unfreeze_epoch the PLM is unfrozen and added to the
  optimizer via add_param_group() at a drastically lower LR (default 1e-5).
  Using add_param_group() preserves the existing AdamW first/second-moment
  state for the soft_prompt while giving the PLM params fresh zero state —
  exactly the right initialization for each stage.
  The scheduler is reset so the PLM LR decays from its full 1e-5 value
  rather than inheriting a mid-schedule multiplier from Stage 1.

Config keys (all under [entity_typing]):
  unfreeze_epoch      : epoch index at which Stage 2 starts (default: num_epochs+1,
                        i.e. prompt-only for the full run).
  soft_prompt_lr      : Stage 1 LR for the soft prompt (default: 3e-3).
  stage2_plm_lr       : Stage 2 LR for the PLM params   (default: 1e-5).
  stage2_prompt_lr    : Stage 2 LR for the soft prompt  (default: same as stage2_plm_lr).
"""

import logging

import mlflow
from torch import optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..data.datasets.entity_typing import EntityTypingDataset
from ..utils.pytorch import get_num_workers
from ..models.soft_prompt_entity_encoding import SoftPromptEntityEncodingModel
from .base import BaseTrainer
from .entity_typing import BatchTripletMarginLoss, EntityTypingTrainer

logger = logging.getLogger("mlflow")

_DEFAULT_STAGE1_LR: float = 3e-3
_DEFAULT_STAGE2_PLM_LR: float = 1e-5


class SoftPromptEntityTypingTrainer(EntityTypingTrainer):
    """Entity Typing trainer with two-stage prompt-initialized fine-tuning."""

    def __init__(self, config: dict):
        BaseTrainer.__init__(self, config)
        et_config = config["entity_typing"]

        # Always start with PLM frozen (train_plm=False).
        # Stage 2 will flip this at runtime.
        self.model = self.accelerator.prepare(
            SoftPromptEntityEncodingModel(
                et_config["plm_name"],
                num_soft_tokens=et_config.get("num_soft_tokens", 5),
                dropout_prob=et_config.get("dropout", 0.1),
                train_plm=False,
            )
        )

        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler.LambdaLR = None
        self.loss_fn: BatchTripletMarginLoss = None
        self.train_dataset: EntityTypingDataset = None
        self.test_dataset: EntityTypingDataset = None

    # ------------------------------------------------------------------
    # Helper: unwrap Accelerate's DDP/FSDP wrapper if present
    # ------------------------------------------------------------------
    def _base_model(self) -> SoftPromptEntityEncodingModel:
        """Return the unwrapped SoftPromptEntityEncodingModel."""
        return getattr(self.model, "module", self.model)

    # ------------------------------------------------------------------
    # Two-stage training loop
    # ------------------------------------------------------------------
    def train(self):
        """Train with two-stage prompt-initialized fine-tuning."""
        logger.info("Training Entity Typing (two-stage soft prompt)")
        et_config = self.config["entity_typing"]
        batch_size = et_config["batch_size"]
        num_epochs = et_config["num_epochs"]
        num_workers = get_num_workers()

        # Epoch at which the PLM is unfrozen.  Default keeps PLM frozen forever.
        unfreeze_epoch: int = et_config.get("unfreeze_epoch", num_epochs + 1)

        stage1_lr: float = et_config.get("soft_prompt_lr", _DEFAULT_STAGE1_LR)
        stage2_plm_lr: float = et_config.get("stage2_plm_lr", _DEFAULT_STAGE2_PLM_LR)
        # Soft-prompt LR in Stage 2 — typically kept the same as PLM to avoid
        # the prompt dominating gradients once the whole model is unlocked.
        stage2_prompt_lr: float = et_config.get("stage2_prompt_lr", stage2_plm_lr)

        train_dataloader = self.accelerator.prepare(
            DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        )
        test_dataloader = self.accelerator.prepare(
            DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        )
        steps_per_epoch = len(train_dataloader)

        # ── Stage 1 optimizer initialisation ──────────────────────────────
        # Strictly freeze every PLM parameter *before* building the optimizer so
        # AdamW never allocates momentum/variance tensors for 110 M frozen params.
        base_model = self._base_model()
        for p in base_model.plm.parameters():
            p.requires_grad = False

        # Optimizer tracks only the (num_soft_tokens × hidden_size) soft prompt.
        self.optimizer = self.accelerator.prepare(
            optim.AdamW([base_model.soft_prompt], lr=stage1_lr)
        )

        # Linear decay over Stage 1.  The scheduler is replaced at unfreeze_epoch.
        stage1_steps = min(unfreeze_epoch, num_epochs + 1) * steps_per_epoch
        self.scheduler = self.accelerator.prepare(
            get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=stage1_steps,
            )
        )
        self.loss_fn = self.accelerator.prepare(BatchTripletMarginLoss())

        mlflow.log_params({
            "stage1_soft_prompt_lr": stage1_lr,
            "unfreeze_epoch": unfreeze_epoch,
            "stage2_plm_lr": stage2_plm_lr,
            "stage2_prompt_lr": stage2_prompt_lr,
        })

        self.evaluate_dataloader(test_dataloader, "test_epoch", 0)

        for epoch in range(1, num_epochs + 1):

            # ── Stage 2 transition ─────────────────────────────────────────
            if epoch == unfreeze_epoch:
                logger.info(
                    "Epoch %s: Stage 2 — unfreezing PLM (plm_lr=%.1e, prompt_lr=%.1e)",
                    epoch, stage2_plm_lr, stage2_prompt_lr,
                )

                # 1. Unfreeze every PLM parameter.
                for p in base_model.plm.parameters():
                    p.requires_grad = True

                # 2. Allow SoftPromptEntityEncodingModel.train() to put PLM in
                #    train mode (the custom train() override respects this flag).
                base_model.train_plm = True

                # 3. Update the soft-prompt param group LR for Stage 2.
                #    This directly mutates the existing param group dict — no
                #    state is lost and the change takes effect on the next step.
                self.optimizer.param_groups[0]["lr"] = stage2_prompt_lr

                # 4. Add PLM parameters as a NEW param group at the low Stage-2 LR.
                #
                #    WHY add_param_group() instead of reinitialising the optimizer:
                #    - Existing AdamW state (m_t, v_t) for the soft_prompt is
                #      preserved — the prompt has been training for several epochs
                #      and its momentum is meaningful.
                #    - PLM params get *fresh* zero state, which is correct: they
                #      have never been updated by this optimizer.
                #    - No extra VRAM spike from keeping a second optimizer alive.
                #
                #    NOTE: AdamW lazily allocates momentum/variance tensors for
                #    the PLM params on the *first optimizer step* after this call,
                #    not right here.  For fp32 BERT-base that is ~880 MB of new
                #    VRAM; use Accelerate's mixed-precision or gradient
                #    checkpointing if this causes OOM.
                self.optimizer.add_param_group(
                    {"params": list(base_model.plm.parameters()), "lr": stage2_plm_lr}
                )

                # 5. Reset scheduler so both param groups decay from their *full*
                #    Stage-2 LR values rather than inheriting the mid-schedule
                #    multiplier left over from Stage 1.
                remaining_steps = (num_epochs - epoch + 1) * steps_per_epoch
                self.scheduler = self.accelerator.prepare(
                    get_linear_schedule_with_warmup(
                        self.optimizer,
                        num_warmup_steps=0,
                        num_training_steps=max(remaining_steps, 1),
                    )
                )

            logger.info("Epoch %s/%s", epoch, num_epochs)
            self.train_one_epoch(epoch, train_dataloader, test_dataloader)
