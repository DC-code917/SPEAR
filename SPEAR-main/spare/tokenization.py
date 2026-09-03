from pathlib import Path


SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
SPECIAL_TOKEN_IDS = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
PAD_ID = SPECIAL_TOKEN_IDS["[PAD]"]
UNK_ID = SPECIAL_TOKEN_IDS["[UNK]"]
CLS_ID = SPECIAL_TOKEN_IDS["[CLS]"]
SEP_ID = SPECIAL_TOKEN_IDS["[SEP]"]
MASK_ID = SPECIAL_TOKEN_IDS["[MASK]"]
DEFAULT_MAX_LENGTH = 1024
_DELIMITER_TRANSLATION = str.maketrans({
    "[": " ",
    "]": " ",
    "{": " ",
    "}": " ",
    ":": " ",
    "'": " ",
    "’": " ",
    "/": " ",
    "\\": " ",
})


def normalize_api_text(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    placeholders = {token: f"SPEARCONTROLTOKEN{index}" for index, token in enumerate(SPECIAL_TOKENS)}
    protected = text
    for token, placeholder in placeholders.items():
        protected = protected.replace(token, placeholder)
    normalized = " ".join(protected.translate(_DELIMITER_TRANSLATION).split())
    for token, placeholder in placeholders.items():
        normalized = normalized.replace(placeholder, token)
    return normalized


def _load_sentencepiece():
    try:
        import sentencepiece
    except ImportError as error:
        raise ImportError("sentencepiece is required to load or train a SPEAR tokenizer") from error
    return sentencepiece


def _fixed_length_input(input_ids, segment_ids, max_length, pad_id):
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    ids = [int(token_id) for token_id in input_ids]
    segments = [int(segment_id) for segment_id in segment_ids]
    if len(ids) != len(segments):
        raise ValueError("input_ids and segment_ids must have the same length")
    ids = ids[:max_length]
    segments = segments[:max_length]
    attention_mask = [1] * len(ids)
    padding_length = max_length - len(ids)
    if padding_length:
        ids.extend([int(pad_id)] * padding_length)
        segments.extend([0] * padding_length)
        attention_mask.extend([0] * padding_length)
    return {
        "input_ids": ids,
        "segment_ids": segments,
        "attention_mask": attention_mask,
    }


def truncate_and_pad(input_ids, segment_ids, max_length=DEFAULT_MAX_LENGTH, pad_id=PAD_ID):
    return _fixed_length_input(input_ids, segment_ids, max_length, pad_id)


def build_full_trace_input(input_ids, max_length=DEFAULT_MAX_LENGTH, pad_id=PAD_ID):
    ids = list(input_ids)
    return _fixed_length_input(ids, [0] * len(ids), max_length, pad_id)


def build_csm_pair_input(left_ids, right_ids, sep_id=SEP_ID, max_length=DEFAULT_MAX_LENGTH, pad_id=PAD_ID):
    left = list(left_ids)
    right = list(right_ids)
    input_ids = left + [int(sep_id)] + right
    segment_ids = [1] * (len(left) + 1) + [2] * len(right)
    return _fixed_length_input(input_ids, segment_ids, max_length, pad_id)


class SpearTokenizer:
    def __init__(self, model_path):
        self.model_path = str(Path(model_path))
        sentencepiece = _load_sentencepiece()
        self.processor = sentencepiece.SentencePieceProcessor(model_file=self.model_path)
        self.vocab_size = int(self.processor.get_piece_size())
        self.special_ids = {
            token: int(self.processor.piece_to_id(token))
            for token in SPECIAL_TOKENS
        }
        if self.special_ids != SPECIAL_TOKEN_IDS:
            raise ValueError(
                "tokenizer special-token IDs must be "
                + ", ".join(f"{token}={token_id}" for token, token_id in SPECIAL_TOKEN_IDS.items())
            )
        self.pad_id = self.special_ids["[PAD]"]
        self.unk_id = self.special_ids["[UNK]"]
        self.cls_id = self.special_ids["[CLS]"]
        self.sep_id = self.special_ids["[SEP]"]
        self.mask_id = self.special_ids["[MASK]"]

    def encode(self, text):
        normalized = normalize_api_text(text)
        if not normalized:
            return []
        return [int(token_id) for token_id in self.processor.encode(normalized, out_type=int)]

    def encode_full_trace(self, text, max_length=DEFAULT_MAX_LENGTH):
        return build_full_trace_input(self.encode(text), max_length=max_length, pad_id=self.pad_id)

    def encode_csm_pair(self, left, right, max_length=DEFAULT_MAX_LENGTH):
        return build_csm_pair_input(
            self.encode(left),
            self.encode(right),
            sep_id=self.sep_id,
            max_length=max_length,
            pad_id=self.pad_id,
        )
