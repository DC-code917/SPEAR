import random

from spare.tokenization import SPECIAL_TOKENS


def _tokenizer_ids(tokenizer):
    if hasattr(tokenizer, "vocab_size"):
        vocab_size = int(tokenizer.vocab_size)
    else:
        vocab_size = len(tokenizer.vocab)
    if hasattr(tokenizer, "special_ids"):
        values = tokenizer.special_ids
        if isinstance(values, dict):
            values = values.values()
        special_ids = {int(value) for value in values}
    else:
        special_ids = {
            int(tokenizer.vocab[token])
            for token in SPECIAL_TOKENS
            if token in tokenizer.vocab
        }
    pad_id = int(tokenizer.pad_id) if hasattr(tokenizer, "pad_id") else int(tokenizer.vocab["[PAD]"])
    mask_id = int(tokenizer.mask_id) if hasattr(tokenizer, "mask_id") else int(tokenizer.vocab["[MASK]"])
    return vocab_size, special_ids, pad_id, mask_id


def create_index(src, tokenizer, whole_word_masking=False, span_masking=False, span_geo_prob=0.2, span_max_length=10):
    _, special_ids, pad_id, _ = _tokenizer_ids(tokenizer)
    positions = [
        index
        for index, token_id in enumerate(src)
        if token_id != pad_id and token_id not in special_ids
    ]
    return positions, list(src)


def mask_seq(
    src,
    tokenizer,
    whole_word_masking=False,
    span_masking=False,
    span_geo_prob=0.2,
    span_max_length=10,
    mask_ratio=0.15,
    rng=None,
):
    rng = rng or random
    vocab_size, special_ids, pad_id, mask_id = _tokenizer_ids(tokenizer)
    output = list(src)
    positions = [
        index
        for index, token_id in enumerate(output)
        if token_id != pad_id and token_id not in special_ids
    ]
    if not positions:
        return output, []
    selected_count = max(1, int(len(positions) * mask_ratio + 0.5))
    selected = rng.sample(positions, selected_count)
    lexical_ids = [token_id for token_id in range(vocab_size) if token_id not in special_ids]
    targets = []
    for position in selected:
        original = output[position]
        targets.append((position, original))
        probability = rng.random()
        if probability < 0.8:
            output[position] = mask_id
        elif probability < 0.9:
            output[position] = rng.choice(lexical_ids)
    return output, targets
