"""Train the multi-task U-Net (tumour mask + tumour type) with PyTorch.

Runs on an NVIDIA GPU (CUDA) if available, otherwise on the CPU. Works on Windows, Linux and Colab.

Data (choose one):
  --figshare-dir DIR                         Jun Cheng's Figshare .mat files (subfolders are searched)
  --images DIR --masks DIR --labels-csv CSV  your own slices; CSV columns: filename,label,patient_id

Examples:
  python train_tumor.py --figshare-dir "C:/Users/chand/Downloads/1512427" --epochs 2   # quick test
  python train_tumor.py --figshare-dir "C:/Users/chand/Downloads/1512427"              # full training
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import evaluate, load_figshare, load_folders, patient_split
from model import MultiTaskUNet, predict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(x, m):
    """Batch augmentation on the GPU: left-right flip (image + mask together), brightness and contrast."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    m = torch.where(flip[:, None, None, None], m.flip(-1), m)
    brightness = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.2      # +/- 0.1
    contrast = 0.9 + torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 0.2          # 0.9 - 1.1
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean + brightness).clamp(0, 1)
    return x, m


def soft_dice(logits, target, smooth=1.0):
    p = torch.sigmoid(logits).flatten()
    t = target.flatten()
    return (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)


def compute_loss(mask_logits, type_logits, m, y, type_weight):
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m) + (1 - soft_dice(mask_logits, m))
    type_loss = F.cross_entropy(type_logits, y)
    return mask_loss + type_weight * type_loss, soft_dice(mask_logits, m).item(), \
        (type_logits.argmax(1) == y).float().mean().item()


def run_epoch(model, x, m, y, idx, args, device, optimizer=None, scaler=None):
    """One pass over idx. Trains if an optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    order = np.random.permutation(idx) if train else idx
    totals, n = np.zeros(3), 0
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        xb = torch.from_numpy(x[b]).float().unsqueeze(1).to(device, non_blocking=True)
        mb = torch.from_numpy(m[b]).float().unsqueeze(1).to(device, non_blocking=True)
        yb = torch.from_numpy(y[b]).long().to(device, non_blocking=True)
        if train:
            xb, mb = augment(xb, mb)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=args.amp):
            mask_logits, type_logits = model(xb)
            loss, dice, acc = compute_loss(mask_logits.float(), type_logits.float(), mb, yb, args.type_weight)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        totals += np.array([loss.item(), dice, acc]) * len(b)
        n += len(b)
    return totals / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figshare-dir")
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--labels-csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--type-weight", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.amp = device.type == "cuda"                      # mixed precision on GPU: faster, less memory
    if device.type == "cuda":
        print(f"Device: GPU – {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU (no CUDA GPU found – training will be slow)")

    print("Loading data ...")
    if args.figshare_dir:
        x, m, y, pids, classes = load_figshare(args.figshare_dir)
    elif args.images and args.masks and args.labels_csv:
        x, m, y, pids, classes = load_folders(args.images, args.masks, args.labels_csv)
    else:
        ap.error("Give --figshare-dir, or all of --images, --masks and --labels-csv")

    tr, va, te = patient_split(pids)
    print(f"{len(x)} slices, {len(set(pids))} patients -> train {len(tr)}, val {len(va)}, test {len(te)}")
    print("Class counts (train):", {c: int((y[tr] == i).sum()) for i, c in enumerate(classes)})
    assert not (set(pids[tr]) & set(pids[te])), "Patient leakage between train and test"

    model = MultiTaskUNet(n_classes=len(classes)).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "unet_tumor_multitask_best.pt")
    history_path = os.path.join(args.out, "history_tumor.csv")
    best_val, bad_epochs = float("inf"), 0
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_loss", "dice", "val_dice", "accuracy", "val_accuracy", "lr"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, tr_acc = run_epoch(model, x, m, y, tr, args, device, optimizer, scaler)
        va_loss, va_dice, va_acc = run_epoch(model, x, m, y, va, args, device)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, va_loss, tr_dice, va_dice, tr_acc, va_acc, lr])
        print(f"Epoch {epoch:3d} | loss {tr_loss:.3f}/{va_loss:.3f} | dice {tr_dice:.3f}/{va_dice:.3f} "
              f"| acc {tr_acc:.3f}/{va_acc:.3f} | lr {lr:.1e} | {time.time() - t0:.0f}s   (train/val)")
        if va_loss < best_val:
            best_val, bad_epochs = va_loss, 0
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    # Evaluate the BEST model on the held-out test patients
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    pred_mask, pred_type = predict(model, x[te], device, batch=args.batch)
    result = evaluate(m[te], pred_mask, y[te], pred_type, classes)
    result.update({"n_test_slices": int(len(te)), "n_test_patients": int(len(set(pids[te]))),
                   "test_patients": sorted(set(pids[te].tolist()))})
    with open(os.path.join(args.out, "test_results_tumor.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"TEST  Dice {result['test_dice_mean']:.3f} | accuracy {result['test_accuracy']:.3f} "
          f"| macro-F1 {result['test_macro_f1']:.3f}  (model: {best_path})")


if __name__ == "__main__":
    main()
