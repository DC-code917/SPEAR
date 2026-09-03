import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path


try:
    from spare.tokenization import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, _load_sentencepiece, normalize_api_text
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spare.tokenization import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, _load_sentencepiece, normalize_api_text


class BPEVocabularyBuilder:
    def __init__(self, vocab_size=50000, model_prefix="vocab", special_tokens=None, vocab_json_path=None):
        if not isinstance(vocab_size, int) or vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("vocab_size must be larger than the protected-token count")
        if special_tokens is not None and set(special_tokens) != set(SPECIAL_TOKENS):
            raise ValueError("special_tokens must contain exactly the SPEAR protected tokens")
        self.total_vocab_size = vocab_size
        self.model_prefix = str(model_prefix)
        self.special_tokens = list(SPECIAL_TOKENS)
        self.vocab_json_path = str(vocab_json_path) if vocab_json_path is not None else self.model_prefix + ".json"
        self.model_path = None
        self.tokenizer = None
        self.vocab = None
        self.reverse_vocab = None

    def split_text_into_chunks(self, text, chunk_size=4096):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        words = normalize_api_text(text).split()
        chunks = []
        current = []
        current_size = 0
        for word in words:
            added_size = len(word) + (1 if current else 0)
            if current and current_size + added_size > chunk_size:
                chunks.append(" ".join(current))
                current = [word]
                current_size = len(word)
            else:
                current.append(word)
                current_size += added_size
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _new_training_file(self):
        prefix_path = Path(self.model_prefix)
        directory = prefix_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=prefix_path.name + "_pretraining_",
            suffix=".txt",
            dir=str(directory),
            delete=False,
        )
        return handle, Path(handle.name)

    def _train_file(self, train_file, remove_train_files):
        sentencepiece = _load_sentencepiece()
        prefix_path = Path(self.model_prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        sentencepiece.SentencePieceTrainer.train(
            input=str(train_file),
            model_prefix=self.model_prefix,
            vocab_size=self.total_vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            normalization_rule_name="identity",
            pad_id=SPECIAL_TOKEN_IDS["[PAD]"],
            pad_piece="[PAD]",
            unk_id=SPECIAL_TOKEN_IDS["[UNK]"],
            unk_piece="[UNK]",
            bos_id=-1,
            eos_id=-1,
            user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"],
            hard_vocab_limit=False,
            shuffle_input_sentence=False,
        )
        self.model_path = self.model_prefix + ".model"
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.model_path)
        self.load_vocab()
        self.dump_vocab()
        if remove_train_files:
            Path(train_file).unlink(missing_ok=True)
            Path(self.model_prefix + ".vocab").unlink(missing_ok=True)
        return self

    def train(self, text_data, remove_train_files=True):
        if not isinstance(text_data, str):
            raise TypeError("text_data must be a string containing pre-training corpus text")
        handle, train_file = self._new_training_file()
        try:
            chunks = self.split_text_into_chunks(text_data)
            if not chunks:
                raise ValueError("pre-training corpus text is empty after normalization")
            with handle:
                handle.write("\n".join(chunks))
                handle.write("\n")
            return self._train_file(train_file, remove_train_files)
        except BaseException:
            handle.close()
            train_file.unlink(missing_ok=True)
            raise

    def train_from_corpora(self, corpus_paths, remove_train_files=True):
        paths = [Path(path) for path in corpus_paths]
        if not paths:
            raise ValueError("at least one pre-training corpus path is required")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        handle, train_file = self._new_training_file()
        written = 0
        try:
            with handle:
                for path in paths:
                    with path.open("r", encoding="utf-8") as source:
                        for line in source:
                            normalized = normalize_api_text(line)
                            if normalized:
                                handle.write(normalized + "\n")
                                written += 1
            if written == 0:
                raise ValueError("pre-training corpora are empty after normalization")
            return self._train_file(train_file, remove_train_files)
        except BaseException:
            handle.close()
            train_file.unlink(missing_ok=True)
            raise

    def train_from_preprocessed(self, dataset_path, remove_train_files=True):
        with Path(dataset_path).open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("format") != "spear-pretraining-v2":
            raise ValueError("Unsupported pre-training dataset format")
        traces = payload.get("full_traces")
        if not isinstance(traces, list) or not traces:
            raise ValueError("The pre-training dataset contains no full traces")
        handle, train_file = self._new_training_file()
        try:
            written = 0
            with handle:
                for trace in traces:
                    normalized = normalize_api_text(trace)
                    if normalized:
                        handle.write(normalized + "\n")
                        written += 1
            if written == 0:
                raise ValueError("The pre-training dataset is empty after normalization")
            return self._train_file(train_file, remove_train_files)
        except BaseException:
            handle.close()
            train_file.unlink(missing_ok=True)
            raise

    def load_vocab(self):
        if self.tokenizer is None:
            model_path = self.model_path or self.model_prefix + ".model"
            sentencepiece = _load_sentencepiece()
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_path)
            self.model_path = model_path
        self.vocab = {
            self.tokenizer.id_to_piece(token_id): token_id
            for token_id in range(self.tokenizer.get_piece_size())
        }
        self.reverse_vocab = {token_id: token for token, token_id in self.vocab.items()}
        observed = {token: self.vocab.get(token) for token in SPECIAL_TOKENS}
        if observed != SPECIAL_TOKEN_IDS:
            raise ValueError("trained tokenizer does not preserve the required special-token IDs")
        if len(self.vocab) > self.total_vocab_size:
            raise ValueError("trained vocabulary exceeds the configured cap")
        return self.vocab

    def dump_vocab(self, vocab_json_path=None):
        if self.vocab is None:
            self.load_vocab()
        output_path = Path(vocab_json_path or self.vocab_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output:
            json.dump(self.vocab, output, indent=4, ensure_ascii=False)
        self.vocab_json_path = str(output_path)
        return self.vocab_json_path


def _build_parser():
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--corpus", nargs="+")
    sources.add_argument("--dataset")
    parser.add_argument("--model-prefix", default="vocab")
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--vocab-json")
    parser.add_argument("--keep-sentencepiece-vocab", action="store_true")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    builder = BPEVocabularyBuilder(
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        vocab_json_path=args.vocab_json,
    )
    if args.dataset:
        builder.train_from_preprocessed(
            args.dataset,
            remove_train_files=not args.keep_sentencepiece_vocab,
        )
    else:
        builder.train_from_corpora(
            args.corpus,
            remove_train_files=not args.keep_sentencepiece_vocab,
        )
    print(json.dumps({
        "model_path": builder.model_path,
        "vocab_path": builder.vocab_json_path,
        "vocab_size": len(builder.vocab),
        "special_ids": SPECIAL_TOKEN_IDS,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
