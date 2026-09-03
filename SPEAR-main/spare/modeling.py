from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpearConfig:
    vocab_size: int
    max_seq_length: int = 1024
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    dropout: float = 0.1
    segment_vocab_size: int = 3
    pad_token_id: int = 0
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    csm_classes: int = 5
    msm_weight: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SpearConfig":
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class SpearEncoder(nn.Module):
    def __init__(self, config: SpearConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.segment_vocab_size, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=config.num_hidden_layers)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_seq_length:
            raise ValueError(
                f"Input length {sequence_length} exceeds {self.config.max_seq_length}."
            )
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, sequence_length)
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        hidden_states = hidden_states + self.segment_embeddings(segment_ids)
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        return self.layers(hidden_states, src_key_padding_mask=~attention_mask.bool())


class MsmHead(nn.Module):
    def __init__(self, config: SpearConfig, token_embeddings: nn.Embedding):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = token_embeddings.weight
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.decoder(hidden_states) + self.bias


class CsmHead(nn.Module):
    def __init__(self, config: SpearConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.csm_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.dense(pooled)))


class SpearPretrainingModel(nn.Module):
    def __init__(self, config: SpearConfig):
        super().__init__()
        self.config = config
        self.encoder = SpearEncoder(config)
        self.msm_head = MsmHead(config, self.encoder.token_embeddings)
        self.csm_head = CsmHead(config)
        self.msm_head.apply(self.encoder._initialize)
        self.csm_head.apply(self.encoder._initialize)
        self.msm_head.decoder.weight = self.encoder.token_embeddings.weight
        with torch.no_grad():
            self.encoder.token_embeddings.weight[config.pad_token_id].zero_()

    def forward(
        self,
        msm_input_ids: torch.Tensor,
        msm_labels: torch.Tensor,
        csm_input_ids: torch.Tensor,
        csm_labels: torch.Tensor,
        msm_segment_ids: Optional[torch.Tensor] = None,
        csm_segment_ids: Optional[torch.Tensor] = None,
        msm_attention_mask: Optional[torch.Tensor] = None,
        csm_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if msm_attention_mask is None:
            msm_attention_mask = msm_input_ids.ne(self.config.pad_token_id)
        if csm_attention_mask is None:
            csm_attention_mask = csm_input_ids.ne(self.config.pad_token_id)
        msm_hidden = self.encoder(msm_input_ids, msm_segment_ids, msm_attention_mask)
        selected = msm_labels.ne(-100)
        if not torch.any(selected):
            raise ValueError("MSM batch contains no selected prediction positions.")
        msm_logits = self.msm_head(msm_hidden[selected])
        msm_loss_sum = F.cross_entropy(msm_logits, msm_labels[selected], reduction="sum")
        msm_count = selected.sum()
        msm_loss = msm_loss_sum / msm_count
        csm_hidden = self.encoder(csm_input_ids, csm_segment_ids, csm_attention_mask)
        csm_pooled = mean_pool(csm_hidden, csm_attention_mask)
        csm_logits = self.csm_head(csm_pooled)
        csm_loss = F.cross_entropy(csm_logits, csm_labels, reduction="mean")
        loss = csm_loss + self.config.msm_weight * msm_loss
        return {
            "loss": loss,
            "csm_loss": csm_loss,
            "msm_loss": msm_loss,
            "msm_loss_sum": msm_loss_sum,
            "msm_count": msm_count,
            "csm_logits": csm_logits,
            "msm_logits": msm_logits,
        }


class SpearClassifier(nn.Module):
    def __init__(
        self,
        config: SpearConfig,
        labels_num: int,
        multilabel: bool = False,
        encoder: Optional[SpearEncoder] = None,
    ):
        super().__init__()
        if labels_num < 2:
            raise ValueError("labels_num must be at least 2.")
        self.config = config
        self.labels_num = labels_num
        self.multilabel = multilabel
        self.encoder = encoder if encoder is not None else SpearEncoder(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_size, labels_num)
        self.dense.apply(self.encoder._initialize)
        self.output.apply(self.encoder._initialize)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        hidden_states = self.encoder(input_ids, segment_ids, attention_mask)
        pooled = mean_pool(hidden_states, attention_mask)
        logits = self.output(self.dropout(torch.tanh(self.dense(pooled))))
        result = {"logits": logits}
        if labels is not None:
            if self.multilabel:
                result["loss"] = F.binary_cross_entropy_with_logits(
                    logits,
                    labels.to(logits.dtype),
                    reduction="mean",
                )
            else:
                result["loss"] = F.cross_entropy(logits, labels.long(), reduction="mean")
        return result


def encoder_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    prefix = "module.encoder."
    if not any(key.startswith(prefix) for key in state):
        prefix = "encoder."
    if any(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    return state
