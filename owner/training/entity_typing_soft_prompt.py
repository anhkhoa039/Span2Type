"""Trainer for soft-prompt Entity Typing."""

import logging

from torch import optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..data.datasets.entity_typing import EntityTypingDataset
from ..utils.pytorch import get_num_workers
from ..models.soft_prompt_entity_encoding import SoftPromptEntityEncodingModel
from .base import BaseTrainer
from .entity_typing import BatchTripletMarginLoss, EntityTypingTrainer

logger = logging.getLogger("mlflow")


class SoftPromptEntityTypingTrainer(EntityTypingTrainer):
    """Entity Typing trainer using soft prompt tuning."""

    def __init__(self, config: dict):
        # Initialize common trainer state from BaseTrainer and mirror
        # EntityTypingTrainer's fields so inherited methods can run unchanged.
        BaseTrainer.__init__(self, config)
        et_config = config["entity_typing"]

        self.model = self.accelerator.prepare(
            SoftPromptEntityEncodingModel(
                et_config["plm_name"],
                num_soft_tokens=et_config.get("num_soft_tokens", 5),
                dropout_prob=et_config.get("dropout", 0.1),
                train_plm=et_config.get("train_plm", True),
            )
        )

        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler.LambdaLR = None
        self.loss_fn: BatchTripletMarginLoss = None
        self.train_dataset: EntityTypingDataset = None
        self.test_dataset: EntityTypingDataset = None

    def train(self):
        """Train and optimize only trainable parameters (soft prompt)."""
        logger.info("Training Entity Typing (soft prompt)")
        et_config = self.config["entity_typing"]
        batch_size = et_config["batch_size"]
        num_epochs = et_config["num_epochs"]
        num_workers = get_num_workers()

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
        num_training_steps = num_epochs * len(train_dataloader)

        soft_prompt_params = []
        plm_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Soft prompt parameter lives under `soft_prompt`.
            if "soft_prompt" in name:
                soft_prompt_params.append(p)
            else:
                plm_params.append(p)

        trainable_params = soft_prompt_params + plm_params
        if not trainable_params:
            raise ValueError("No trainable parameters found for soft prompt training.")

        soft_prompt_lr = et_config.get("soft_prompt_learning_rate", et_config["learning_rate"])
        plm_lr = et_config.get("plm_learning_rate", et_config["learning_rate"])

        param_groups = []
        if soft_prompt_params:
            param_groups.append({"params": soft_prompt_params, "lr": soft_prompt_lr})
        if plm_params:
            param_groups.append({"params": plm_params, "lr": plm_lr})

        self.optimizer = self.accelerator.prepare(
            optim.AdamW(param_groups)
        )
        self.scheduler = self.accelerator.prepare(
            get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )
        )
        self.loss_fn = self.accelerator.prepare(BatchTripletMarginLoss())

        self.evaluate_dataloader(test_dataloader, "test_epoch", 0)
        for epoch in range(1, num_epochs + 1):
            logger.info("Epoch %s/%s", epoch, num_epochs)
            self.train_one_epoch(epoch, train_dataloader, test_dataloader)

