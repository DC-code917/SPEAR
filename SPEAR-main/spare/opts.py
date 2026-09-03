def model_opts(parser):
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--initializer_range", type=float, default=0.02)


def optimization_opts(parser):
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)


def training_opts(parser):
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--total_steps", type=int, default=120000)
    parser.add_argument("--save_checkpoint_steps", type=int, default=30000)
    parser.add_argument("--report_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)


def tokenizer_opts(parser):
    parser.add_argument("--tokenizer_model_path", "--spm_model_path", dest="tokenizer_model_path", required=True)


def finetune_opts(parser):
    parser.add_argument("--pretrained_model_path")
    parser.add_argument("--output_model_path", default="models/finetuned_model.bin")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--dev_path", required=True)
    parser.add_argument("--test_path")
    tokenizer_opts(parser)
    model_opts(parser)
    optimization_opts(parser)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--epochs_num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)


def infer_opts(parser):
    parser.add_argument("--load_model_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--prediction_path", required=True)
    tokenizer_opts(parser)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=1024)
