"""Train a multi-task tumour model (tumour mask + tumour type) with PyTorch.

Runs on an NVIDIA GPU (CUDA) if available, otherwise on the CPU. Works on Windows, Linux and Colab.

Data (choose one):
  --figshare-dir DIR                         Jun Cheng's Figshare .mat files (subfolders are searched)
  --images DIR --masks DIR --labels-csv CSV  your own slices; CSV columns: filename,label,patient_id

Models (--arch):
  unetplusplus (default)  U-Net++ on an ImageNet-pretrained EfficientNet-B3 encoder (--encoder)
  unet                    the original from-scratch multi-task U-Net (baseline)

Examples:
  python train_tumor.py --figshare-dir DATA --epochs 2                       # quick test
  python train_tumor.py --figshare-dir DATA --out models_v2                  # improved model
  python train_tumor.py --figshare-dir DATA --arch unet --out models_v1      # baseline
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import IMG_SIZE, evaluate, load_figshare, load_folders, patient_split
from model import build_model, predict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(x, m, strong=True):
    """Batch augmentation on the GPU, applied identically to image and mask.

    Always: left-right flip, brightness/contrast.
    Strong: random rotation (+/-15 deg), zoom (0.85-1.15), shift (+/-8 %), gamma (0.7-1.5), slight noise.
    """
    n, dev = x.shape[0], x.device
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    m = torch.where(flip[:, None, None, None], m.flip(-1), m)
    if strong:
        ang = (torch.rand(n, device=dev) - 0.5) * math.radians(30)
        scale = 0.85 + torch.rand(n, device=dev) * 0.30
        shift = (torch.rand(n, 2, device=dev) - 0.5) * 0.16
        cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
        theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        m = (F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=False) > 0.5).float()
        gamma = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.76)       # ~0.68 - 1.46
        x = x.clamp(1e-6, 1) ** gamma
        x = x + torch.randn_like(x) * 0.02 * (torch.rand(n, 1, 1, 1, device=dev) < 0.3)
    brightness = (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.2
    contrast = 0.9 + torch.rand(n, 1, 1, 1, device=dev) * 0.2
    mean = x.mean(dim=(2, 3), keepdim=True)
    return ((x - mean) * contrast + mean + brightness).clamp(0, 1), m


def dice_per_image(logits, target, smooth=1.0):
    """Soft Dice for each slice separately (matches how the test Dice is measured)."""
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    return (2 * (p * t).sum(1) + smooth) / (p.sum(1) + t.sum(1) + smooth)


def compute_loss(mask_logits, type_logits, m, y, args, class_weights):
    dice = dice_per_image(mask_logits, m)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, m) + (1 - dice).mean()
    type_loss = F.cross_entropy(type_logits, y, weight=class_weights, label_smoothing=args.label_smoothing)
    hard_dice = dice_per_image((mask_logits > 0).float() * 20 - 10, m).mean().item()     # thresholded Dice
    acc = (type_logits.argmax(1) == y).float().mean().item()
    return mask_loss + args.type_weight * type_loss, hard_dice, acc


def run_epoch(model, x, m, y, idx, args, device, class_weights, optimizer=None, scaler=None, scheduler=None):
    """One pass over idx. Trains if an optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    order = np.random.permutation(idx) if train else idx
    totals, n = np.zeros(3), 0
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        if train and len(b) < 2:                         # BatchNorm needs at least 2 samples
            continue
        xb = torch.from_numpy(x[b]).float().unsqueeze(1).to(device, non_blocking=True)
        mb = torch.from_numpy(m[b]).float().unsqueeze(1).to(device, non_blocking=True)
        yb = torch.from_numpy(y[b]).long().to(device, non_blocking=True)
        if train:
            xb, mb = augment(xb, mb, strong=not args.light_aug)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=args.amp):
            mask_logits, type_logits = model(xb)
            loss, dice, acc = compute_loss(mask_logits.float(), type_logits.float(), mb, yb, args, class_weights)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
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
    ap.add_argument("--arch", default="unetplusplus",
                    help="unetplusplus (default), unet-pretrained, manet, deeplabv3plus, upernet, or unet (from scratch)")
    ap.add_argument("--encoder", default="efficientnet-b3", help="Pretrained encoder, e.g. efficientnet-b3, resnet34")
    ap.add_argument("--size", type=int, default=IMG_SIZE, help="Image size; 384 helps small tumours (slower)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--type-weight", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--light-aug", action="store_true", help="Only flips and brightness (old behaviour)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.amp = device.type == "cuda"
    print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if args.amp else 'CPU (training will be slow)'}")

    print("Loading data ...")
    if args.figshare_dir:
        x, m, y, pids, classes = load_figshare(args.figshare_dir, args.size)
    elif args.images and args.masks and args.labels_csv:
        x, m, y, pids, classes = load_folders(args.images, args.masks, args.labels_csv, args.size)
    else:
        ap.error("Give --figshare-dir, or all of --images, --masks and --labels-csv")

    tr, va, te = patient_split(pids)
    print(f"{len(x)} slices, {len(set(pids))} patients -> train {len(tr)}, val {len(va)}, test {len(te)}")
    counts = np.bincount(y[tr], minlength=len(classes))
    print("Class counts (train):", dict(zip(classes, counts.tolist())))
    assert not (set(pids[tr]) & set(pids[te])), "Patient leakage between train and test"
    class_weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)

    model = build_model(args.arch, args.encoder, n_classes=len(classes)).to(device)
    print(f"Model: {args.arch}" + ("" if args.arch == "unet" else f" + {args.encoder} (ImageNet)")
          + f" | parameters: {sum(p.numel() for p in model.parameters()):,} | image {args.size}px")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "unet_tumor_multitask_best.pt")
    history_path = os.path.join(args.out, "history_tumor.csv")
    best_score, bad_epochs = -1.0, 0
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_loss", "dice", "val_dice", "accuracy", "val_accuracy", "lr"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, tr_acc = run_epoch(model, x, m, y, tr, args, device, class_weights, optimizer, scaler, scheduler)
        va_loss, va_dice, va_acc = run_epoch(model, x, m, y, va, args, device, class_weights)
        lr = optimizer.param_groups[0]["lr"]
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, va_loss, tr_dice, va_dice, tr_acc, va_acc, lr])
        score = (va_dice + va_acc) / 2                       # keep the model that is best at BOTH tasks
        mark = ""
        if score > best_score:
            best_score, bad_epochs, mark = score, 0, "  * best"
            torch.save({"model": model.state_dict(), "classes": classes, "arch": args.arch,
                        "encoder": args.encoder, "size": args.size}, best_path)
        else:
            bad_epochs += 1
        print(f"Epoch {epoch:3d} | loss {tr_loss:.3f}/{va_loss:.3f} | dice {tr_dice:.3f}/{va_dice:.3f} "
              f"| acc {tr_acc:.3f}/{va_acc:.3f} | lr {lr:.1e} | {time.time() - t0:.0f}s   (train/val){mark}", flush=True)
        if bad_epochs >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    # Evaluate the BEST model on the held-out test patients (with flip TTA + largest-region cleanup)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    pred_mask, pred_type = predict(model, x[te], device, batch=args.batch)
    result = evaluate(m[te], pred_mask, y[te], pred_type, classes)
    result.update({"model": args.arch, "encoder": None if args.arch == "unet" else args.encoder, "image_size": args.size,
                   "n_test_slices": int(len(te)), "n_test_patients": int(len(set(pids[te]))),
                   "test_patients": sorted(set(pids[te].tolist()))})
    with open(os.path.join(args.out, "test_results_tumor.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"TEST  Dice {result['test_dice_mean']:.3f} | accuracy {result['test_accuracy']:.3f} "
          f"| macro-F1 {result['test_macro_f1']:.3f}  (model: {best_path})")
    print("      Dice per type:", {k: round(v, 3) for k, v in result["test_dice_per_class"].items()})


if __name__ == "__main__":
    main()
