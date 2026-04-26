"""SCAN dataset loader and tokenizer for compositional generalization.

SCAN (Lake & Baroni 2018) is a sequence-to-sequence task where input is
an English instruction and output is an action sequence:
    "jump twice and walk"  ->  "JUMP JUMP WALK"

We download the canonical splits from the SCAN GitHub repo (no Python
deps beyond stdlib). For autoregressive training we concatenate
input + <SEP> + output + <EOS> and supervise loss only on output tokens.

Splits we use:
  - simple   : random IID split (sanity check)
  - mcd1     : compositional distribution split #1 (hard)

The "hard" splits hold out specific compositional patterns; transformers
are known to perform poorly on MCD splits (sub-50%).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


SCAN_URLS = {
    "simple_train":   "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt",
    "simple_test":    "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt",
    "mcd1_train":     "https://raw.githubusercontent.com/brendenlake/SCAN/master/MCD_splits/tasks_train_mcd1.txt",
    "mcd1_test":      "https://raw.githubusercontent.com/brendenlake/SCAN/master/MCD_splits/tasks_test_mcd1.txt",
    "addjump_train":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt",
    "addjump_test":   "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def fetch_scan(split: str, cache_dir: str = "/tmp/scan_data") -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (train, test) lists of (in_str, out_str) pairs for a SCAN split."""
    cache = Path(cache_dir)
    train_url = SCAN_URLS[f"{split}_train"]
    test_url = SCAN_URLS[f"{split}_test"]
    train_path = cache / f"{split}_train.txt"
    test_path = cache / f"{split}_test.txt"
    _download(train_url, train_path)
    _download(test_url, test_path)
    return _parse(train_path), _parse(test_path)


def _parse(path: Path) -> list[tuple[str, str]]:
    """Parse a SCAN tasks file. Lines look like:
        IN: jump twice OUT: I_JUMP I_JUMP
    """
    pairs = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            if "IN:" not in ln or "OUT:" not in ln:
                continue
            in_part, out_part = ln.split("OUT:")
            in_part = in_part.replace("IN:", "").strip()
            out_part = out_part.strip()
            pairs.append((in_part, out_part))
    return pairs


class ScanVocab:
    """Word-level vocabulary covering input words, action words, and specials."""
    PAD = 0
    SEP = 1
    EOS = 2

    def __init__(self, pairs: list[tuple[str, str]]):
        in_words: set[str] = set()
        out_words: set[str] = set()
        for x, y in pairs:
            in_words.update(x.split())
            out_words.update(y.split())
        self.input_tokens = sorted(in_words)
        self.output_tokens = sorted(out_words)
        # Layout: [PAD, SEP, EOS, *input_words, *output_words]
        self.tok2id: dict[str, int] = {"<pad>": 0, "<sep>": 1, "<eos>": 2}
        for w in self.input_tokens:
            self.tok2id[w] = len(self.tok2id)
        for w in self.output_tokens:
            self.tok2id[w] = len(self.tok2id)
        self.id2tok: dict[int, str] = {v: k for k, v in self.tok2id.items()}
        self.vocab_size = len(self.tok2id)
        self.first_output_id = 3 + len(self.input_tokens)

    def encode_pair(self, x: str, y: str, max_len: int) -> tuple[list[int], list[int], int]:
        """Encode an input/output pair as a single token sequence with markers.

        Returns (token_ids, mask, n_input_tokens) where mask is 1 on tokens that
        should contribute to loss (only output tokens including EOS).
        """
        x_ids = [self.tok2id[w] for w in x.split()]
        y_ids = [self.tok2id[w] for w in y.split()]
        # Sequence: x ... <sep> y ... <eos>
        seq = x_ids + [self.SEP] + y_ids + [self.EOS]
        if len(seq) > max_len:
            # truncate from the right (rare).
            seq = seq[:max_len]
        n_input = len(x_ids) + 1  # input tokens including the SEP
        # Mask: positions where loss is computed = output positions (after SEP through EOS).
        mask = [0] * len(seq)
        for i in range(n_input, len(seq)):
            mask[i] = 1
        return seq, mask, n_input


class ScanDataset(Dataset):
    """Wraps a list of (input, output) pairs into tensors with a ScanVocab.

    For training: input = sequence shifted by 1, target = sequence,
    with mask blocking input tokens from loss.
    """
    def __init__(self, pairs: list[tuple[str, str]], vocab: ScanVocab, max_len: int):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len
        # Pre-encode for speed.
        self.encoded = []
        for x, y in pairs:
            seq, mask, n_input = vocab.encode_pair(x, y, max_len)
            self.encoded.append((seq, mask, n_input))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq, mask, n_input = self.encoded[idx]
        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "n_input": torch.tensor(n_input, dtype=torch.long),
        }


def collate_pad(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Pad sequences in a batch to the max length present."""
    max_len = max(b["seq"].size(0) for b in batch)
    B = len(batch)
    seq = torch.full((B, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.float)
    n_in = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["seq"].size(0)
        seq[i, :L] = b["seq"]
        mask[i, :L] = b["mask"]
        n_in[i] = b["n_input"]
    return {"seq": seq, "mask": mask, "n_input": n_in}


def sequence_exact_match(pred_ids: list[list[int]], target_ids: list[list[int]]) -> float:
    """Fraction of predictions that exactly match the target token sequence."""
    n_correct = 0
    for p, t in zip(pred_ids, target_ids):
        if p == t:
            n_correct += 1
    return n_correct / max(1, len(pred_ids))
