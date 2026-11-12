"""Train ONE model that both segments a brain tumour and classifies its type.

Architecture: a U-Net whose encoder is shared by two heads
  * "mask" head -> tumour segmentation (Dice + BCE loss)
  * "type" head -> tumour type from the bottleneck features (cross-entropy)

Default dataset: Jun Cheng's Figshare brain tumour dataset (CC BY 4.0)
  3,064 T1 contrast-enhanced slices, 233 patients, labels
  1 = meningioma, 2 = glioma, 3 = pituitary, with a tumour mask and patient ID per slice.
  https://doi.org/10.6084/m9.figshare.1512427
  Note: every slice contains a tumour (there is no "no tumour" class).

Two ways to provide data:
  1) Figshare .mat files:   --figshare-dir /path/to/mat_files
  2) Your own folders:      --images DIR --masks DIR --labels-csv labels.csv
     labels.csv columns: filename,label,patient_id   (label = class name)

The split is ALWAYS by patient (70/15/15), so slices from one patient never
appear in both training and test data.

Example (Colab):
  python train_tumor.py --figshare-dir /content/brain_tumor_mat --out models
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np


def dice_coefficient(y_true, y_pred, smooth=1.0):
    import tensorflow as tf
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_bce_loss(y_true, y_pred):
    import tensorflow as tf
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_coefficient(y_true, y_pred))


def per_image_dice(y_true, y_pred, threshold=0.5):
    scores = []
    for t, p in zip(y_true, y_pred):
        t, p = t > 0.5, p > threshold
        denom = t.sum() + p.sum()
        scores.append(1.0 if denom == 0 else 2.0 * np.logical_and(t, p).sum() / denom)
    return np.array(scores)


IMG_SIZE = 256
FIGSHARE_CLASSES = ["meningioma", "glioma", "pituitary"]


def _prep_image(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)          # per-image min-max
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)[..., None]


def _prep_mask(mask):
    mask = cv2.resize(mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)[..., None]


def load_figshare(mat_dir):
    """Read Figshare .mat (MATLAB v7.3 / HDF5) files from mat_dir and its subfolders.

    Works with the official download as-is (four 'brainTumorDataPublic_*' folders);
    files without a 'cjdata' struct, such as cvind.mat, are skipped.
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
    return np.array(images), np.array(masks), np.array(labels), np.array(patients), FIGSHARE_CLASSES


def load_folders(image_dir, mask_dir, labels_csv):
    """Read PNG/JPG slices + masks, with labels and patient IDs from a CSV."""
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
    return np.array(images), np.array(masks), np.array(labels), np.array(patients), classes


def patient_split(patients, seed=42):
    """70/15/15 split with every patient in exactly one part."""
    from sklearn.model_selection import GroupShuffleSplit
    idx = np.arange(len(patients))
    train, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(idx, groups=patients))
    v, t = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(rest, groups=patients[rest]))
    return train, rest[v], rest[t]


def build_model(n_classes, base=32):
    import tensorflow as tf
    L = tf.keras.layers

    def block(x, f):
        for _ in range(2):
            x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = L.BatchNormalization()(x)
            x = L.Activation("relu")(x)
        return x

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 1))
    skips, x = [], inputs
    for i in range(4):
        x = block(x, base * 2 ** i)
        skips.append(x)
        x = L.MaxPooling2D()(x)
    bottleneck = L.Dropout(0.3)(block(x, base * 16))

    # Classification head: tumour type from the shared encoder features
    c = L.GlobalAveragePooling2D()(bottleneck)
    c = L.Dropout(0.4)(L.Dense(128, activation="relu")(c))
    type_out = L.Dense(n_classes, activation="softmax", name="type")(c)

    # Segmentation head: standard U-Net decoder
    x = bottleneck
    for i in reversed(range(4)):
        x = L.Conv2DTranspose(base * 2 ** i, 2, strides=2, padding="same")(x)
        x = block(L.Concatenate()([x, skips[i]]), base * 2 ** i)
    mask_out = L.Conv2D(1, 1, activation="sigmoid", name="mask")(x)

    return tf.keras.Model(inputs, {"mask": mask_out, "type": type_out})


def make_dataset(x, m, y, batch, augment):
    import tensorflow as tf

    def aug(img, t):
        mask = t["mask"]
        if tf.random.uniform(()) > 0.5:  # flip image and mask together; type is unchanged
            img, mask = tf.image.flip_left_right(img), tf.image.flip_left_right(mask)
        img = tf.clip_by_value(tf.image.random_brightness(img, 0.1), 0.0, 1.0)
        img = tf.clip_by_value(tf.image.random_contrast(img, 0.9, 1.1), 0.0, 1.0)
        return img, {"mask": mask, "type": t["type"]}

    ds = tf.data.Dataset.from_tensor_slices((x, {"mask": m, "type": y}))
    if augment:
        ds = ds.shuffle(len(x), seed=42).map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def evaluate(true_masks, pred_masks, true_labels, pred_probs, classes):
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
    ap.add_argument("--type-weight", type=float, default=0.5,
                    help="Weight of the classification loss relative to the mask loss")
    args = ap.parse_args()

    if args.figshare_dir:
        x, m, y, pids, classes = load_figshare(args.figshare_dir)
    elif args.images and args.masks and args.labels_csv:
        x, m, y, pids, classes = load_folders(args.images, args.masks, args.labels_csv)
    else:
        ap.error("Give --figshare-dir, or all of --images, --masks and --labels-csv")

    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)

    tr, va, te = patient_split(pids)
    print(f"{len(x)} slices, {len(set(pids))} patients -> train {len(tr)}, val {len(va)}, test {len(te)}")
    print("Class counts (train):", {c: int((y[tr] == i).sum()) for i, c in enumerate(classes)})

    model = build_model(len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss={"mask": dice_bce_loss, "type": "sparse_categorical_crossentropy"},
                  loss_weights={"mask": 1.0, "type": args.type_weight},
                  metrics={"mask": [dice_coefficient], "type": ["accuracy"]})

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "unet_tumor_multitask_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8),
        tf.keras.callbacks.CSVLogger(os.path.join(args.out, "history_tumor.csv")),
    ]
    model.fit(make_dataset(x[tr], m[tr], y[tr], args.batch, augment=True),
              validation_data=make_dataset(x[va], m[va], y[va], args.batch, augment=False),
              epochs=args.epochs, callbacks=callbacks)

    out = model.predict(x[te], batch_size=args.batch)
    pred_mask, pred_type = (out["mask"], out["type"]) if isinstance(out, dict) else out
    result = evaluate(m[te], pred_mask, y[te], pred_type, classes)
    result.update({"n_test_slices": int(len(te)), "n_test_patients": int(len(set(pids[te]))),
                   "test_patients": sorted(set(pids[te].tolist()))})
    with open(os.path.join(args.out, "test_results_tumor.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"TEST  Dice {result['test_dice_mean']:.3f} | accuracy {result['test_accuracy']:.3f} "
          f"| macro-F1 {result['test_macro_f1']:.3f}  (model: {best_path})")


if __name__ == "__main__":
    main()
