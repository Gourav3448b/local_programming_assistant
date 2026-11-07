"""A dependency-free UTF-8 byte tokenizer.

Every possible UTF-8 byte is representable, so unseen words never become UNK.
"""

from __future__ import annotations

EOS_ID = 256
BOS_ID = 257
VOCAB_SIZE = 258


class ByteTokenizer:
    """Maps UTF-8 bytes to IDs 0..255 plus BOS/EOS special tokens."""

    vocab_size = VOCAB_SIZE

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], skip_special: bool = True) -> str:
        byte_values: list[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in (BOS_ID, EOS_ID):
                if skip_special:
                    continue
                continue
            if 0 <= token_id <= 255:
                byte_values.append(token_id)
        return bytes(byte_values).decode("utf-8", errors="replace")
