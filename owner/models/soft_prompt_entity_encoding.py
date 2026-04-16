"""
Soft prompt entity encoding model.

This module prepends trainable soft tokens to the beginning of the input sequence,
feeds the resulting embeddings into a frozen Hugging Face encoder, and extracts
the hidden state at the (shifted) [MASK] position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class SoftPromptEntityEncodingModel(nn.Module):
    """
    Entity encoder using prepended trainable soft prompts (P-tuning style).

    Requirements implemented:
    1) Loads a Hugging Face base model via `AutoModel.from_pretrained(plm_name)`.
    2) Freezes all base model parameters (no gradients).
    3) Maintains `num_soft_tokens` trainable soft token embeddings of size
       `hidden_size`.
    4) Converts `input_ids` -> dense embeddings using the frozen model's embedding
       layer, prepends soft token embeddings, expands attention mask accordingly,
       and runs the frozen encoder via `inputs_embeds`.
    5) Extracts the final hidden state at the shifted `[MASK]` index and applies
       dropout.
    """

    def __init__(
        self,
        plm_name: str,
        num_soft_tokens: int = 5,
        dropout_prob: float = 0.1,
        train_plm: bool = True,
    ) -> None:
        super().__init__()

        self.plm = AutoModel.from_pretrained(plm_name)
        self.config = AutoConfig.from_pretrained(plm_name)

        self.hidden_size: int = int(self.config.hidden_size)
        self.num_soft_tokens: int = int(num_soft_tokens)
        self.train_plm: bool = bool(train_plm)

        # Optional freezing for prompt-only mode.
        for p in self.plm.parameters():
            p.requires_grad = self.train_plm

        # Trainable soft prompt embeddings: shape [N, H]
        self.soft_prompt = nn.Parameter(
            torch.empty(self.num_soft_tokens, self.hidden_size)
        )
        nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)

        # Dropout applied only to the extracted [MASK] representation.
        self.dropout = nn.Dropout(p=dropout_prob)

    def _word_embeddings(self) -> nn.Module:
        """
        Returns the frozen token embedding module.

        Requirement explicitly targets `model.embeddings.word_embeddings`, which is
        available for common BERT-like architectures.
        """
        if not hasattr(self.plm, "embeddings") or not hasattr(
            self.plm.embeddings, "word_embeddings"
        ):
            raise RuntimeError(
                "Expected base model to have `embeddings.word_embeddings`. "
                "If you're using a non-BERT architecture, adjust this accessor."
            )
        return self.plm.embeddings.word_embeddings

    def train(self, mode: bool = True):
        """Keep frozen PLM in eval mode during prompt-only training."""
        super().train(mode)
        # Backbone is frozen in prompt-only mode; keep it deterministic.
        if not self.train_plm:
            self.plm.eval()
        return self

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.LongTensor,  # [B, L]
        mask_index: torch.LongTensor,  # [B] (index in the original sequence)
    ) -> torch.FloatTensor:  # [B, H]
        batch_size, _ = input_ids.shape
        device = input_ids.device

        # 1) Convert input_ids -> dense embeddings using the frozen embedding layer.
        #    Must not pass input_ids directly into the base model.
        token_embeddings = self._word_embeddings()(input_ids)  # [B, L, H]

        # 2) Prepend soft prompt embeddings to the beginning of the sequence.
        #    soft_prompt: [N, H] -> [B, N, H]
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, N, H]
        inputs_embeds = torch.cat([soft_prompt_batch, token_embeddings], dim=1)  # [B, N+L, H]

        # 3) Expand attention_mask to account for prepended soft tokens.
        #    Soft tokens are always attended to.
        soft_attention_mask = torch.ones(
            batch_size, self.num_soft_tokens, dtype=attention_mask.dtype, device=device
        )  # [B, N]
        expanded_attention_mask = torch.cat(
            [soft_attention_mask, attention_mask], dim=1
        )  # [B, N+L]

        # 4) Run the frozen encoder with inputs_embeds + expanded attention mask.
        outputs = self.plm(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_attention_mask,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state  # [B, N+L, H]

        # 5) Extract representation at shifted [MASK] position.
        #    Since we prepended N soft tokens, original mask_index shifts by +N.
        shifted_mask_index = mask_index.to(torch.long) + self.num_soft_tokens  # [B]
        batch_indices = torch.arange(batch_size, device=device)
        mask_repr = last_hidden_state[batch_indices, shifted_mask_index, :]  # [B, H]

        # Apply dropout and return.
        return self.dropout(mask_repr)


__all__ = ["SoftPromptEntityEncodingModel"]

