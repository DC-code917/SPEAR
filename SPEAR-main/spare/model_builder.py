from spare.modeling import SpearConfig, SpearPretrainingModel


def _value(args, name, default):
    return getattr(args, name, default)


def build_model(args):
    tokenizer = getattr(args, "tokenizer", None)
    vocab_size = _value(args, "vocab_size", None)
    if vocab_size is None and tokenizer is not None:
        vocab_size = tokenizer.vocab_size
    if vocab_size is None:
        raise ValueError("vocab_size or tokenizer is required.")
    pad_token_id = _value(args, "pad_token_id", None)
    if pad_token_id is None and tokenizer is not None:
        pad_token_id = tokenizer.pad_id
    config = SpearConfig(
        vocab_size=vocab_size,
        max_seq_length=_value(args, "seq_length", _value(args, "max_seq_length", 1024)),
        hidden_size=_value(args, "hidden_size", 768),
        intermediate_size=_value(args, "intermediate_size", _value(args, "feedforward_size", 3072)),
        num_hidden_layers=_value(args, "num_hidden_layers", _value(args, "layers_num", 12)),
        num_attention_heads=_value(args, "num_attention_heads", _value(args, "heads_num", 12)),
        dropout=_value(args, "dropout", 0.1),
        segment_vocab_size=3,
        pad_token_id=0 if pad_token_id is None else pad_token_id,
        layer_norm_eps=_value(args, "layer_norm_eps", 1e-12),
        initializer_range=_value(args, "initializer_range", 0.02),
        csm_classes=5,
        msm_weight=0.1,
    )
    return SpearPretrainingModel(config)
