from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_finetune import NextPOIDataset
from src.data.collate import collate_finetune
from src.models.model import STBert, STBertConfig
from src.utils.metrics import topk_and_mrr, format_metrics
from src.utils.checkpoint import load_checkpoint


def mask_special_logits(logits: torch.Tensor, num_special: int = 3) -> torch.Tensor:
    logits = logits.clone()
    logits[:, :num_special] = -1e9
    return logits


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics_sum = None
    n = 0
    for batch in tqdm(loader, desc="Eval", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)
        logits, _, _ = model.forward_next(
            batch["input_tokens"], batch["tod"], batch["dow"], batch["attn_mask"],
            batch["user"], batch["query_tod"], batch["query_dow"],
            causal=True,
        )
        logits = mask_special_logits(logits, num_special=3)
        m = topk_and_mrr(logits, batch["target_poi"], topk=(1, 5, 10, 20))
        if metrics_sum is None:
            metrics_sum = {k: 0.0 for k in m.keys()}
        for k in m:
            metrics_sum[k] += m[k]
        n += 1

    if n == 0:
        return {"acc@1": 0.0, "acc@5": 0.0, "acc@10": 0.0, "acc@20": 0.0, "mrr": 0.0}
    return {k: v / n for k, v in metrics_sum.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_path", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache_path, map_location="cpu")

    PAD = cache["special_tokens"]["pad"]
    MASK = cache["special_tokens"]["mask"]
    CLS = cache["special_tokens"]["cls"]

    ckpt = load_checkpoint(args.ckpt_path, map_location="cpu")
    cfg_dict = ckpt.hparams.get("cfg", {})

    # If ckpt doesn't store cfg (older), fall back to cache
    if not cfg_dict:
        cfg = STBertConfig(
            vocab_size=int(cache["vocab_size"]),
            num_users=int(cache["num_users"]),
            num_cats=int(cache["num_cats"]),
            num_tod=int(cache["num_tod_bins"]) + 1,
            num_dow=8,
            region_vocab_sizes=[int(v) for v in cache["region_vocab_sizes"]],
            max_len=int(cache["max_len"]),
        )
    else:
        cfg = STBertConfig(**cfg_dict)

    model = STBert(
        cfg,
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        token2xy=cache["token2xy"],
        pad_token=PAD,
        mask_token=MASK,
        cls_token=CLS,
    ).to(device)
    model.load_state_dict(ckpt.model_state, strict=True)

    augment = True if args.split == "train" else False
    ds = NextPOIDataset(
        cache["splits"][args.split],
        token2cat=cache["token2cat"],
        token2regions=cache["token2regions"],
        max_len=cfg.max_len,
        pad_token=PAD,
        cls_token=CLS,
        augment=augment,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_finetune, pin_memory=True)

    metrics = evaluate(model, loader, device)
    print(f"{args.split.upper()} : {format_metrics(metrics)}")


if __name__ == "__main__":
    main()
