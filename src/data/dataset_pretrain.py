from __future__ import annotations
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MaskedPretrainDataset(Dataset):
    """
    BERT-style masked modeling for POI sequences.
    Each sample is a window of a user's trajectory:
      input_tokens: [max_len] (with CLS at pos0, PAD at tail)
      time features: tod/dow aligned to tokens (CLS/PAD -> 0)
      targets: only masked positions filled, others = -100
    """

    def __init__(
        self,
        trajs: List[Tuple[int, List[int], List[int], List[int]]],
        *,
        token2cat: np.ndarray,
        token2regions: List[np.ndarray],
        max_len: int = 128,
        mask_prob: float = 0.25,
        mask_token: int = 1,
        pad_token: int = 0,
        cls_token: int = 2,
        vocab_size: int | None = None,
        window_stride: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.trajs = trajs
        self.token2cat = token2cat
        self.token2regions = token2regions
        self.max_len = int(max_len)
        self.mask_prob = float(mask_prob)
        self.mask_token = int(mask_token)
        self.pad_token = int(pad_token)
        self.cls_token = int(cls_token)
        self.vocab_size = int(vocab_size) if vocab_size is not None else int(token2cat.shape[0])
        self.window_stride = int(window_stride)
        self.rng = random.Random(seed)

        # precompute windows: list of (traj_idx, start_pos)
        self.windows: List[Tuple[int, int]] = []
        max_poi_len = self.max_len - 1  # reserve 1 for CLS
        for ti, (_, tokens, _, _) in enumerate(self.trajs):
            L = len(tokens)
            if L <= 1:
                continue
            if L <= max_poi_len:
                self.windows.append((ti, 0))
            else:
                for st in range(0, L - max_poi_len + 1, self.window_stride):
                    self.windows.append((ti, st))
                # ensure last window reaches tail
                last = L - max_poi_len
                if self.windows[-1] != (ti, last):
                    self.windows.append((ti, last))

        if len(self.windows) == 0:
            raise ValueError("No windows constructed. Check your data / max_len.")

    def __len__(self):
        return len(self.windows)

    def _mask_tokens(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """
        Apply BERT masking to tokens (expects CLS already included).
        Returns masked_tokens, target_tokens (target only at masked positions else -100).
        """
        L = len(tokens)
        assert L <= self.max_len
        # candidates: positions 1..L-1 (exclude CLS)
        cand = list(range(1, L))
        if len(cand) == 0:
            masked = tokens[:]
            targets = [-100] * self.max_len
            return masked, targets

        n_mask = max(1, int(round(self.mask_prob * len(cand))))
        mask_pos = self.rng.sample(cand, k=min(n_mask, len(cand)))

        masked = tokens[:]
        targets = [-100] * self.max_len
        for p in mask_pos:
            orig = tokens[p]
            targets[p] = orig
            r = self.rng.random()
            if r < 0.80:
                masked[p] = self.mask_token
            elif r < 0.90:
                # random POI token (avoid special tokens)
                masked[p] = self.rng.randint(3, self.vocab_size - 1)
            else:
                masked[p] = orig
        # pad target to max_len (already max_len sized)
        return masked, targets

    def __getitem__(self, idx: int):
        traj_idx, start = self.windows[idx]
        user, tokens, tod, dow = self.trajs[traj_idx]

        max_poi_len = self.max_len - 1
        chunk_tokens = tokens[start:start + max_poi_len]
        chunk_tod = tod[start:start + max_poi_len]
        chunk_dow = dow[start:start + max_poi_len]

        # prepend CLS
        seq_tokens = [self.cls_token] + chunk_tokens
        seq_tod = [0] + [t + 1 if t >= 0 else 0 for t in chunk_tod]  # shift by +1, 0=unknown
        seq_dow = [0] + [d + 1 if d >= 0 else 0 for d in chunk_dow]

        # pad
        attn = [1] * len(seq_tokens)
        pad_len = self.max_len - len(seq_tokens)
        if pad_len > 0:
            seq_tokens = seq_tokens + [self.pad_token] * pad_len
            seq_tod = seq_tod + [0] * pad_len
            seq_dow = seq_dow + [0] * pad_len
            attn = attn + [0] * pad_len

        masked_tokens, target_poi = self._mask_tokens(seq_tokens[:len(seq_tokens) - pad_len])

        # if padded, ensure masked_tokens length == max_len
        if pad_len > 0:
            masked_tokens = masked_tokens + [self.pad_token] * pad_len

        # build auxiliary targets (only for masked positions)
        target_cat = [-100] * self.max_len
        target_regions = [[-100] * self.max_len for _ in range(len(self.token2regions))]
        for i in range(self.max_len):
            if target_poi[i] != -100:
                tok = target_poi[i]
                target_cat[i] = int(self.token2cat[tok])
                for s, t2r in enumerate(self.token2regions):
                    target_regions[s][i] = int(t2r[tok])

        return {
            "user": int(user),
            "input_tokens": torch.tensor(masked_tokens, dtype=torch.long),
            "tod": torch.tensor(seq_tod, dtype=torch.long),
            "dow": torch.tensor(seq_dow, dtype=torch.long),
            "attn_mask": torch.tensor(attn, dtype=torch.bool),
            "target_poi": torch.tensor(target_poi, dtype=torch.long),
            "target_cat": torch.tensor(target_cat, dtype=torch.long),
            "target_regions": torch.tensor(target_regions, dtype=torch.long),  # [S, L]
        }
