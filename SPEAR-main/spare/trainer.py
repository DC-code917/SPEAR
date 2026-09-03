import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from spare.modeling import SpearConfig, SpearPretrainingModel
from spare.tokenization import SpearTokenizer


class PretrainingBatcher:
    def __init__(
        self,
        payload: Dict[str, Any],
        tokenizer: SpearTokenizer,
        batch_size: int,
        seq_length: int,
        seed: int,
    ):
        if payload.get("format") != "spear-pretraining-v2":
            raise ValueError("Unsupported pretraining dataset format.")
        full_traces = payload.get("full_traces")
        csm_pools = payload.get("csm_pools")
        if not isinstance(full_traces, list) or not full_traces:
            raise ValueError("The dataset contains no full traces.")
        if not isinstance(csm_pools, list) or len(csm_pools) != 5:
            raise ValueError("The dataset must contain five CSM pools.")
        self.full_traces = [str(trace) for trace in full_traces if str(trace).strip()]
        if not self.full_traces:
            raise ValueError("The dataset contains no non-empty full traces.")
        self.csm_pools: List[List[Tuple[str, str]]] = []
        for label, pool in enumerate(csm_pools):
            if not isinstance(pool, list):
                raise ValueError(f"CSM pool {label + 1} is invalid.")
            pairs = [
                (str(pair[0]), str(pair[1]))
                for pair in pool
                if isinstance(pair, (list, tuple))
                and len(pair) == 2
                and str(pair[0]).strip()
                and str(pair[1]).strip()
            ]
            if not pairs:
                raise ValueError(f"CSM pool {label + 1} is empty.")
            self.csm_pools.append(pairs)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if seq_length < 3:
            raise ValueError("seq_length must be at least 3.")
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rng = random.Random(seed)
        special_ids = tokenizer.special_ids
        if isinstance(special_ids, dict):
            special_ids = special_ids.values()
        self.special_ids = {int(token_id) for token_id in special_ids}
        self.lexical_ids = tuple(
            token_id
            for token_id in range(tokenizer.vocab_size)
            if token_id not in self.special_ids
        )
        if not self.lexical_ids:
            raise ValueError("The tokenizer contains no lexical tokens.")

    def state_dict(self) -> Dict[str, Any]:
        return {"rng_state": self.rng.getstate()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])

    def _mask_trace(self, trace: str) -> Optional[Dict[str, List[int]]]:
        encoded = self.tokenizer.encode_full_trace(trace, max_length=self.seq_length)
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        segment_ids = [int(segment_id) for segment_id in encoded["segment_ids"]]
        attention_mask = [int(value) for value in encoded["attention_mask"]]
        lexical_positions = [
            index
            for index, (token_id, visible) in enumerate(zip(input_ids, attention_mask))
            if visible and token_id not in self.special_ids
        ]
        if not lexical_positions:
            return None
        selected_num = max(1, int(len(lexical_positions) * 0.15 + 0.5))
        selected_positions = self.rng.sample(lexical_positions, selected_num)
        labels = [-100] * len(input_ids)
        for position in selected_positions:
            labels[position] = input_ids[position]
            probability = self.rng.random()
            if probability < 0.8:
                input_ids[position] = self.tokenizer.mask_id
            elif probability < 0.9:
                input_ids[position] = self.rng.choice(self.lexical_ids)
        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def msm_batch(self) -> Dict[str, torch.Tensor]:
        instances: List[Dict[str, List[int]]] = []
        attempts = 0
        attempts_limit = max(100, self.batch_size * 20)
        while len(instances) < self.batch_size and attempts < attempts_limit:
            attempts += 1
            instance = self._mask_trace(self.rng.choice(self.full_traces))
            if instance is not None:
                instances.append(instance)
        if len(instances) != self.batch_size:
            raise ValueError("Unable to build an MSM batch with lexical tokens.")
        return {
            "msm_input_ids": torch.tensor(
                [instance["input_ids"] for instance in instances], dtype=torch.long
            ),
            "msm_segment_ids": torch.tensor(
                [instance["segment_ids"] for instance in instances], dtype=torch.long
            ),
            "msm_attention_mask": torch.tensor(
                [instance["attention_mask"] for instance in instances], dtype=torch.bool
            ),
            "msm_labels": torch.tensor(
                [instance["labels"] for instance in instances], dtype=torch.long
            ),
        }

    def _balanced_labels(self) -> List[int]:
        counts = [self.batch_size // 5] * 5
        for label in self.rng.sample(range(5), self.batch_size % 5):
            counts[label] += 1
        labels = [label for label, count in enumerate(counts) for _ in range(count)]
        self.rng.shuffle(labels)
        return labels

    def csm_batch(self) -> Dict[str, torch.Tensor]:
        labels = self._balanced_labels()
        instances = []
        for label in labels:
            left, right = self.rng.choice(self.csm_pools[label])
            instances.append(
                self.tokenizer.encode_csm_pair(
                    left,
                    right,
                    max_length=self.seq_length,
                )
            )
        return {
            "csm_input_ids": torch.tensor(
                [instance["input_ids"] for instance in instances], dtype=torch.long
            ),
            "csm_segment_ids": torch.tensor(
                [instance["segment_ids"] for instance in instances], dtype=torch.long
            ),
            "csm_attention_mask": torch.tensor(
                [instance["attention_mask"] for instance in instances], dtype=torch.bool
            ),
            "csm_labels": torch.tensor(labels, dtype=torch.long),
        }

    def next_batch(self) -> Dict[str, torch.Tensor]:
        batch = self.msm_batch()
        batch.update(self.csm_batch())
        return batch


def _load_payload(path: str) -> Dict[str, Any]:
    with open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("The pretraining dataset must be a dictionary payload.")
    return payload


def _resolve_distributed(args: Any) -> Tuple[bool, int, int, int, torch.device]:
    environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    requested_world_size = getattr(args, "world_size", None)
    if requested_world_size not in (None, 1, environment_world_size):
        raise ValueError("Multi-process training must be launched with torchrun.")
    distributed = environment_world_size > 1
    requested_device = getattr(args, "device", None)
    if requested_device:
        device = torch.device(requested_device)
        if distributed and device.type == "cuda" and device.index is None:
            device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")
    if distributed:
        backend = getattr(args, "backend", None) or (
            "nccl" if device.type == "cuda" else "gloo"
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        environment_world_size = dist.get_world_size()
    return distributed, rank, local_rank, environment_world_size, device


def _seed_model(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_config(args: Any, tokenizer: SpearTokenizer) -> SpearConfig:
    return SpearConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        segment_vocab_size=3,
        pad_token_id=tokenizer.pad_id,
        layer_norm_eps=args.layer_norm_eps,
        initializer_range=args.initializer_range,
        csm_classes=5,
        msm_weight=0.1,
    )


def _linear_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _numbered_checkpoint_path(path: str, step: int) -> Path:
    output = Path(path)
    if output.suffix:
        return output.with_name(f"{output.stem}-{step}{output.suffix}")
    return output.with_name(f"{output.name}-{step}")


def _unwrap(model: torch.nn.Module) -> SpearPretrainingModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _local_rng_state(batcher: PretrainingBatcher, device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "batcher": batcher.state_dict(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _collect_rng_states(
    batcher: PretrainingBatcher,
    distributed: bool,
    world_size: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    local_state = _local_rng_state(batcher, device)
    if not distributed:
        return [local_state]
    states: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(states, local_state)
    return [state for state in states if state is not None]


def _restore_rng_state(
    checkpoint: Dict[str, Any],
    rank: int,
    batcher: PretrainingBatcher,
    device: torch.device,
) -> None:
    states = checkpoint.get("rng_states")
    if not isinstance(states, list) or rank >= len(states):
        return
    state = states[rank]
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)
    if "batcher" in state:
        batcher.load_state_dict(state["batcher"])


def _save_checkpoint(
    path: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batcher: PretrainingBatcher,
    distributed: bool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    rng_states = _collect_rng_states(batcher, distributed, world_size, device)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        module = _unwrap(model)
        torch.save(
            {
                "format": "spear-pretraining-checkpoint-v2",
                "config": module.config.to_dict(),
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "rng_states": rng_states,
            },
            path,
        )
    if distributed:
        dist.barrier()


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _decay_parameter(name: str) -> bool:
    lowered = name.lower()
    if lowered.endswith("bias"):
        return False
    if lowered.endswith("weight") and "norm" in lowered:
        return False
    return True


def _reduce_stats(stats: torch.Tensor, distributed: bool) -> torch.Tensor:
    reduced = stats.clone()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def train_and_validate(args: Any) -> int:
    distributed, rank, _, world_size, device = _resolve_distributed(args)
    try:
        _seed_model(args.seed)
        tokenizer = SpearTokenizer(args.tokenizer_model_path)
        payload = _load_payload(args.dataset_path)
        checkpoint = None
        if args.resume_from:
            checkpoint = torch.load(args.resume_from, map_location="cpu")
            if not isinstance(checkpoint, dict):
                raise ValueError("Invalid pretraining checkpoint.")
        if checkpoint is not None and isinstance(checkpoint.get("config"), dict):
            config = SpearConfig.from_dict(checkpoint["config"])
        else:
            config = _build_config(args, tokenizer)
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match.")
        if config.pad_token_id != tokenizer.pad_id:
            raise ValueError("Checkpoint and tokenizer PAD token IDs do not match.")
        if config.max_seq_length != args.seq_length:
            raise ValueError("Checkpoint and requested sequence lengths do not match.")
        model = SpearPretrainingModel(config).to(device)
        named_parameters = list(model.named_parameters())
        parameter_groups = [
            {
                "params": [
                    parameter
                    for name, parameter in named_parameters
                    if _decay_parameter(name)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in named_parameters
                    if not _decay_parameter(name)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
        warmup_steps = int(args.total_steps * args.warmup)
        scheduler = _linear_schedule(optimizer, warmup_steps, args.total_steps)
        step = 0
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                _move_optimizer_state(optimizer, device)
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            step = int(checkpoint.get("step", 0))
        if step < 0 or step > args.total_steps:
            raise ValueError("Checkpoint step is outside the requested training range.")
        if distributed:
            if device.type == "cuda":
                model = DistributedDataParallel(
                    model,
                    device_ids=[device.index],
                    output_device=device.index,
                )
            else:
                model = DistributedDataParallel(model)
        torch.manual_seed(args.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed(args.seed + rank)
        batcher = PretrainingBatcher(
            payload,
            tokenizer,
            args.batch_size,
            args.seq_length,
            args.seed + rank,
        )
        if checkpoint is not None:
            _restore_rng_state(checkpoint, rank, batcher, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        stats = torch.zeros(7, dtype=torch.float64, device=device)
        window_steps = 0
        window_start = time.time()
        while step < args.total_steps:
            batch = {
                name: tensor.to(device, non_blocking=device.type == "cuda")
                for name, tensor in batcher.next_batch().items()
            }
            outputs = model(**batch)
            if distributed:
                global_msm_count = outputs["msm_count"].detach().clone()
                dist.all_reduce(global_msm_count, op=dist.ReduceOp.SUM)
                backward_loss = outputs["csm_loss"] + (
                    _unwrap(model).config.msm_weight
                    * outputs["msm_loss_sum"]
                    * world_size
                    / global_msm_count.clamp_min(1)
                )
            else:
                backward_loss = outputs["loss"]
            backward_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            selected = batch["msm_labels"].ne(-100)
            stats[0] += outputs["loss"].detach().to(torch.float64)
            stats[1] += outputs["csm_loss"].detach().to(torch.float64)
            stats[2] += outputs["msm_loss"].detach().to(torch.float64)
            stats[3] += outputs["csm_logits"].argmax(dim=-1).eq(
                batch["csm_labels"]
            ).sum().to(torch.float64)
            stats[4] += batch["csm_labels"].numel()
            stats[5] += outputs["msm_logits"].argmax(dim=-1).eq(
                batch["msm_labels"][selected]
            ).sum().to(torch.float64)
            stats[6] += selected.sum().to(torch.float64)
            window_steps += 1
            if step % args.report_steps == 0 or step == args.total_steps:
                reduced = _reduce_stats(stats, distributed)
                if rank == 0:
                    denominator = max(1, window_steps * world_size)
                    elapsed = max(time.time() - window_start, 1e-9)
                    print(
                        f"step={step}/{args.total_steps} "
                        f"loss={reduced[0].item() / denominator:.6f} "
                        f"csm_loss={reduced[1].item() / denominator:.6f} "
                        f"msm_loss={reduced[2].item() / denominator:.6f} "
                        f"csm_acc={reduced[3].item() / max(reduced[4].item(), 1.0):.6f} "
                        f"msm_acc={reduced[5].item() / max(reduced[6].item(), 1.0):.6f} "
                        f"steps_per_second={window_steps / elapsed:.3f}",
                        flush=True,
                    )
                stats.zero_()
                window_steps = 0
                window_start = time.time()
            if step % args.save_checkpoint_steps == 0:
                _save_checkpoint(
                    _numbered_checkpoint_path(args.output_model_path, step),
                    step,
                    model,
                    optimizer,
                    scheduler,
                    batcher,
                    distributed,
                    rank,
                    world_size,
                    device,
                )
        _save_checkpoint(
            Path(args.output_model_path),
            step,
            model,
            optimizer,
            scheduler,
            batcher,
            distributed,
            rank,
            world_size,
            device,
        )
        return step
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
