import argparse
import os
import sys

sys.path.append(os.getcwd())

from spare.trainer import train_and_validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset_path", default="dataset.pt")
    tokenizer_group = parser.add_mutually_exclusive_group()
    tokenizer_group.add_argument("--tokenizer_model_path")
    tokenizer_group.add_argument("--spm_model_path")
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument(
        "--resume_from",
        "--pretrained_model_path",
        dest="resume_from",
        default=None,
    )
    parser.add_argument("--total_steps", type=int, default=120000)
    parser.add_argument("--save_checkpoint_steps", type=int, default=30000)
    parser.add_argument("--report_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", "--max_seq_length", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument(
        "--intermediate_size",
        "--feedforward_size",
        type=int,
        default=3072,
    )
    parser.add_argument(
        "--num_hidden_layers",
        "--layers_num",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--num_attention_heads",
        "--heads_num",
        type=int,
        default=12,
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default=None)
    parser.add_argument("--device", default=None)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.tokenizer_model_path = args.tokenizer_model_path or args.spm_model_path
    if not args.tokenizer_model_path:
        parser.error("--tokenizer_model_path is required")
    if args.total_steps < 1:
        parser.error("--total_steps must be positive")
    if args.save_checkpoint_steps < 1:
        parser.error("--save_checkpoint_steps must be positive")
    if args.report_steps < 1:
        parser.error("--report_steps must be positive")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.seq_length < 3:
        parser.error("--seq_length must be at least 3")
    if args.hidden_size < 1 or args.hidden_size % args.num_attention_heads != 0:
        parser.error("--hidden_size must be divisible by --num_attention_heads")
    if args.intermediate_size < 1 or args.num_hidden_layers < 1:
        parser.error("model dimensions must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if not 0.0 <= args.warmup < 1.0:
        parser.error("--warmup must be in [0, 1)")
    if args.learning_rate <= 0.0:
        parser.error("--learning_rate must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight_decay must be non-negative")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        parser.error("Adam beta values must be in [0, 1)")
    if args.adam_epsilon <= 0.0:
        parser.error("--adam_epsilon must be positive")
    if args.world_size < 1:
        parser.error("--world_size must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    train_and_validate(args)


if __name__ == "__main__":
    main()
