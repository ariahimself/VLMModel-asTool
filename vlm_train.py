"""
vlm_train.py

Train the MNIST VLM (Option A):
- Uses MNIST labels to generate (question, answer) pairs on the fly.
- Loads your pretrained ViT weights if available.
- Freezes the vision encoder by default, and trains:
    - vision_proj (maps ViT CLS embedding -> decoder embedding)
    - text decoder + token embeddings

Outputs a checkpoint: vlm_mnist_vqa.pth
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vlm_tokenizer import CharTokenizer
from vlm_dataset import MNISTVQADataset, collate_batch
from vlm_model import MNISTVLM, VLMConfig


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    logits: (B,T,V)
    labels: (B,T)
    loss_mask: (B,T) float 0/1
    """
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    labels = labels.reshape(B * T)
    mask = loss_mask.reshape(B * T)

    # Compute per-token CE, ignoring PAD
    loss = F.cross_entropy(logits, labels, ignore_index=pad_id, reduction="none")  # (B*T,)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


@torch.no_grad()
def evaluate(model: MNISTVLM, loader: DataLoader, pad_id: int, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        logits = model(images, input_ids)
        loss = masked_cross_entropy(logits, labels, loss_mask, pad_id)

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--vit_weights", type=str, default="vit_mnist_model.pth", help="pretrained ViT weights (from main2.py)")
    ap.add_argument("--out", type=str, default="vlm_mnist_vqa.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--freeze_vision", action="store_true", help="freeze ViT encoder (recommended)")
    ap.add_argument("--unfreeze_last_n", type=int, default=0, help="optional: unfreeze last N ViT blocks after 1 epoch")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharTokenizer(max_len=args.max_len)

    train_ds = MNISTVQADataset(root=args.data_root, train=True, tokenizer=tokenizer, seed=0, download=True)
    val_ds = MNISTVQADataset(root=args.data_root, train=False, tokenizer=tokenizer, seed=123, download=True)

    collate_fn = lambda b: collate_batch(b, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    cfg = VLMConfig(
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dim_feedforward=args.d_model * 2,
        dropout=0.1,
        max_len=args.max_len,
    )
    model = MNISTVLM(vocab_size=tokenizer.vocab_size, cfg=cfg).to(device)

    vit_path = Path(args.vit_weights)
    if vit_path.exists():
        print(f"Loading ViT weights from: {vit_path}")
        vit_sd = torch.load(str(vit_path), map_location="cpu")
        model.load_vit_weights(vit_sd)
    else:
        print(f"ViT weights not found at {vit_path}. You can train ViT first using main2.py.")

    if args.freeze_vision:
        print("Freezing vision encoder.")
        model.freeze_vision()

    # Only optimize trainable parameters
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_val = float("inf")
    out_path = Path(args.out)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        # Optional: after epoch 1, unfreeze last N ViT blocks
        if epoch == 2 and args.unfreeze_last_n > 0:
            print(f"Unfreezing last {args.unfreeze_last_n} ViT blocks for joint fine-tuning.")
            # unfreeze selected blocks + norm
            for p in model.vision.vit.norm.parameters():
                p.requires_grad = True
            for block in list(model.vision.vit.blocks)[-args.unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            # add to optimizer
            optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr * 0.2)

        for batch in train_loader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(images, input_ids)
            loss = masked_cross_entropy(logits, labels, loss_mask, tokenizer.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += float(loss.item())
            steps += 1

        train_loss = running / max(steps, 1)
        val_loss = evaluate(model, val_loader, tokenizer.pad_id, device)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "tokenizer": tokenizer.state_dict(),
                "cfg": cfg.__dict__,
            }
            torch.save(ckpt, str(out_path))
            print(f"Saved best checkpoint to: {out_path} (val_loss={best_val:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()
