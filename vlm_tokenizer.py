"""
vlm_tokenizer.py

A tiny character-level tokenizer for MNIST VQA.
- Works on arbitrary user prompts (within the allowed character set).
- Generates short answers like: "odd", "even", "seven", "yes", "no", "7", etc.

We keep it simple so you don't need any external pretrained text model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TokenizerConfig:
    max_len: int = 256


class CharTokenizer:
    def __init__(self, max_len: int = 256):
        self.cfg = TokenizerConfig(max_len=max_len)

        # Special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        # A practical character set for prompts.
        # Lowercased prompts are recommended.
        chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            " .,!?;:/\\-+*=()[]{}<>\"'_|"
            "\n"
        )

        self.itos: List[str] = [self.pad_token, self.bos_token, self.eos_token, self.unk_token] + list(chars)
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        text = text.lower()
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)

        # Truncate (keep EOS if requested)
        max_len = self.cfg.max_len
        if len(ids) > max_len:
            if add_eos and ids[-1] == self.eos_id:
                ids = ids[: max_len - 1] + [self.eos_id]
            else:
                ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else self.unk_token)
        return "".join(out)

    def state_dict(self) -> dict:
        return {"max_len": self.cfg.max_len}

    @classmethod
    def from_state_dict(cls, d: dict) -> "CharTokenizer":
        return cls(max_len=int(d.get("max_len", 256)))
