"""Custom tokenizer implementations used by discrete diffusion datasets."""

from __future__ import annotations

from typing import Dict, List
import string
import transformers

from .generators.brevo import (
    bos_token_id as BREVO_BOS_TOKEN_ID,
    eos_token_id as BREVO_EOS_TOKEN_ID,
    mask_token_id as BREVO_MASK_TOKEN_ID,
    pad_token_id as BREVO_PAD_TOKEN_ID,
)

__all__ = [
    "BrevoDummyTokenizer",
    "SyntheticTokenizer",
    "Text8Tokenizer",
    "AsciiCharTokenizer",
]


class BrevoDummyTokenizer:
    """Hardcoded tokenizer metadata for pretokenized BREVO."""

    def __init__(self) -> None:
        self.vocab_size = BREVO_MASK_TOKEN_ID + 1
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.cls_token = self.bos_token
        self.sep_token = self.eos_token

        self.bos_token_id = BREVO_BOS_TOKEN_ID
        self.eos_token_id = BREVO_EOS_TOKEN_ID
        self.pad_token_id = BREVO_PAD_TOKEN_ID
        self.mask_token_id = BREVO_MASK_TOKEN_ID

        self.special_tokens_map = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
        }
        self.all_special_tokens = [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.mask_token,
        ]
        self.all_special_ids = [
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.mask_token_id,
        ]
        self._special_id_to_token = {
            self.bos_token_id: self.bos_token,
            self.eos_token_id: self.eos_token,
            self.pad_token_id: self.pad_token,
            self.mask_token_id: self.mask_token,
        }
        self._special_token_to_id = {
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.pad_token: self.pad_token_id,
            self.mask_token: self.mask_token_id,
        }
        # BREVO uses `bos-2` as a query delimiter marker in pretokenized streams.
        self.query_sep_token = "[Q_SEP]"
        self.query_sep_token_id = self.bos_token_id - 2
        self._display_id_to_token = {
            self.query_sep_token_id: self.query_sep_token,
            **self._special_id_to_token,
        }

        self.padding_side = "right"
        self.truncation_side = "right"

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("BrevoDummyTokenizer does not support tokenization")

    def encode(self, *args, **kwargs):
        raise NotImplementedError("BrevoDummyTokenizer does not support tokenization")

    def convert_ids_to_tokens(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, (list, tuple)):
            return [self.convert_ids_to_tokens(tok_id) for tok_id in ids]
        token_id = int(ids)
        return self._display_id_to_token.get(token_id, str(token_id))

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.convert_tokens_to_ids(tok) for tok in tokens]
        if tokens == self.query_sep_token:
            return self.query_sep_token_id
        if tokens in self._special_token_to_id:
            return self._special_token_to_id[tokens]
        return int(tokens)

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        del kwargs
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids = [int(tok) for tok in token_ids]
        if skip_special_tokens:
            special_ids = {
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
                self.mask_token_id,
            }
            ids = [tok for tok in ids if tok not in special_ids]
        return " ".join(str(tok) for tok in ids)

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **kwargs):
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs)
            for seq in sequences
        ]


class SyntheticTokenizer(transformers.PreTrainedTokenizer):
    """Simple synthetic tokenizer for deterministic experiments."""

    def __init__(
        self,
        vocab_size,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token=None,
        cls_token=None,
        pad_token=None,
        mask_token=None,
        unk_token=None,
        **kwargs,
    ):

        self.tokens = []
        for i in range(vocab_size - 2):
            self.tokens.append(str(i) + " ")
        self._vocab_str_to_int = {
            "[BOS]": vocab_size - 2,
            "[EOS]": vocab_size - 1,
            **{ch: i for i, ch in enumerate(self.tokens)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int


class AsciiCharTokenizer(transformers.PreTrainedTokenizer):
    """Character-level tokenizer for generic English text and symbols."""

    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        # Keep one-token-per-char behavior while covering common prompt symbols.
        self.characters = list(
            dict.fromkeys(string.ascii_lowercase + string.digits + string.punctuation + " ")
        )
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int
