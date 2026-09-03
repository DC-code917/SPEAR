from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spare.modeling import CsmHead, MsmHead, SpearConfig, mean_pool


class CsmMsmTarget(nn.Module):
    def __init__(
        self,
        config: SpearConfig,
        token_embeddings: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.config = config
        if token_embeddings is None:
            token_embeddings = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
            )
        self.token_embeddings = token_embeddings
        self.msm_head = MsmHead(config, token_embeddings)
        self.csm_head = CsmHead(config)

    def forward(
        self,
        msm_hidden_states: torch.Tensor,
        msm_labels: torch.Tensor,
        csm_hidden_states: torch.Tensor,
        csm_labels: torch.Tensor,
        csm_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        selected = msm_labels.ne(-100)
        if not torch.any(selected):
            raise ValueError("MSM batch contains no selected prediction positions.")
        msm_logits = self.msm_head(msm_hidden_states[selected])
        msm_loss_sum = F.cross_entropy(msm_logits, msm_labels[selected], reduction="sum")
        msm_count = selected.sum()
        msm_loss = msm_loss_sum / msm_count
        csm_pooled = mean_pool(csm_hidden_states, csm_attention_mask)
        csm_logits = self.csm_head(csm_pooled)
        csm_loss = F.cross_entropy(csm_logits, csm_labels.long(), reduction="mean")
        return {
            "loss": csm_loss + self.config.msm_weight * msm_loss,
            "csm_loss": csm_loss,
            "msm_loss": msm_loss,
            "msm_loss_sum": msm_loss_sum,
            "msm_count": msm_count,
            "csm_logits": csm_logits,
            "msm_logits": msm_logits,
        }
