"""Data loading, patient-level split and evaluation metrics (NumPy only, no deep-learning framework)."""

import csv
import os

import cv2
import numpy as np

IMG_SIZE = 256
FIGSHARE_CLASSES = ["meningioma", "glioma", "pituitary"]


def _prep_image(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)          # per-image min-max to [0, 1]
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def _prep_mask(mask):
    mask = cv2.resize(mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def load_figshare(mat_dir):
    """Read Jun Cheng's Figshare .mat files (MATLAB v7.3 / HDF5) from mat_dir and its subfolders.

    Each file holds a 'cjdata' struct with image, tumorMask, label (1-3) and PID.
    Files without 'cjdata' (e.g. cvind.mat) are skipped.
    Returns images (N, 256, 256) float32, masks (N, 256, 256) uint8, labels (N,), patient IDs (N,), class names.
    """
    import h5py
    paths = sorted(os.path.join(root, n) for root, _, names in os.walk(mat_dir)
                   for n in names if n.endswith(".mat"))
    images, masks, labels, patients = [], [], [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            if "cjdata" not in f:
                continue
            d = f["cjdata"]
            img = np.array(d["image"]).T                  # MATLAB stores arrays transposed
            mask = np.array(d["tumorMask"]).T
            label = int(np.array(d["label"]).flatten()[0])
            pid = "".join(chr(int(c)) for c in np.array(d["PID"]).flatten())
        images.append(_prep_image(img))
        masks.append(_prep_mask(mask))
        labels.append(label - 1)                          # 1..3 -> 0..2
        patients.append(pid)
    if not images:
        raise FileNotFoundError(f"No Figshare .mat files with 'cjdata' found under {mat_dir}")
    return (np.stack(images), np.stack(masks), np.array(labels), np.array(patients), FIGSHARE_CLASSES)


def load_folders(image_dir, mask_dir, labels_csv):
    """Read PNG/JPG slices + masks; labels and patient IDs come from a CSV (filename,label,patient_id)."""
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    classes = sorted({r["label"] for r in rows})
    images, masks, labels, patients = [], [], [], []
    for r in rows:
        img = cv2.imread(os.path.join(image_dir, r["filename"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, r["filename"]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"Skipping {r['filename']}: missing image or mask")
            continue
        images.append(_prep_image(img))
        masks.append(_prep_mask(mask > 127))
        labels.append(classes.index(r["label"]))
        patients.append(r["patient_id"])
    return (np.stack(images), np.stack(masks), np.array(labels), np.array(patients), classes)


def patient_split(patients, seed=42):
    """70/15/15 split with every patient in exactly one part."""
    from sklearn.model_selection import GroupShuffleSplit
    idx = np.arange(len(patients))
    train, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(idx, groups=patients))
    v, t = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(rest, groups=patients[rest]))
    return train, rest[v], rest[t]


def per_image_dice(true_masks, pred_masks, threshold=0.5):
    scores = []
    for t, p in zip(true_masks, pred_masks):
        t, p = t > 0.5, p > threshold
        denom = t.sum() + p.sum()
        scores.append(1.0 if denom == 0 else 2.0 * np.logical_and(t, p).sum() / denom)
    return np.array(scores)


def evaluate(true_masks, pred_masks, true_labels, pred_probs, classes):
    """Test-set metrics: Dice (overall and per type), accuracy, macro-F1, confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    dice = per_image_dice(true_masks, pred_masks)
    pred_labels = pred_probs.argmax(axis=1)
    return {
        "test_dice_mean": float(dice.mean()), "test_dice_std": float(dice.std()),
        "test_dice_per_class": {c: float(dice[true_labels == i].mean())
                                for i, c in enumerate(classes) if np.any(true_labels == i)},
        "test_accuracy": float(accuracy_score(true_labels, pred_labels)),
        "test_macro_f1": float(f1_score(true_labels, pred_labels, average="macro")),
        "confusion_matrix": {"classes": classes,
                             "matrix": confusion_matrix(true_labels, pred_labels,
                                                        labels=list(range(len(classes)))).tolist()},
    }
