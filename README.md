# Brain Tumour Segmentation and Classification (Multi-task U-Net)

One PyTorch model that **outlines the tumour** and **predicts its type** on 2D brain MRI slices.

> ⚠️ Research and education only. Not validated for clinical use.

## Files

| File | Purpose |
|---|---|
| `model.py` | Multi-task U-Net (shared encoder + segmentation head + classification head) |
| `data_utils.py` | Loading Figshare `.mat` files or your own images, patient-level split, metrics |
| `train_tumor.py` | Training and test evaluation |
| `colab_train_tumor.ipynb` | Run everything in Google Colab |
| `setup_windows.ps1` | One-time Windows setup (virtual environment + PyTorch with GPU) |
| `.vscode/` | Ready-made VS Code run configurations |

## Model

| Part | Details |
|---|---|
| Input | 1 × 256 × 256 MRI slice, scaled to [0, 1] per image |
| Shared encoder | 4 conv blocks (2 × [3×3 conv → BatchNorm → ReLU]) with 2×2 max-pool; 32 → 256 channels |
| Bottleneck | Conv block, 512 channels at 16 × 16, dropout 0.3 |
| Segmentation head | 4 × [2×2 transposed conv → concat skip → conv block] → 1×1 conv → tumour mask; loss **BCE + Dice** |
| Classification head | Global average pool → linear 128 + ReLU → dropout 0.4 → linear 3; loss **cross-entropy** (weight 0.5) |
| Parameters | ≈ 7.83 M |

## Data

**Jun Cheng's Figshare brain tumour dataset** (CC BY 4.0): 3,064 T1-weighted contrast-enhanced slices from 233 patients, each with a radiologist-drawn tumour mask, tumour type and patient ID: meningioma (708), glioma (1,426), pituitary (930).

Download: <https://doi.org/10.6084/m9.figshare.1512427>. Unzip anywhere; the script searches subfolders and skips `cvind.mat`. Every slice contains a tumour, so there is **no "no tumour" class**.

Own data: image and mask folders with matching filenames plus `labels.csv` (`filename,label,patient_id`).

## Run in VS Code (Windows)

1. **Clone:** `Ctrl+Shift+P` → *Git: Clone* → this repo, then open the folder. Install the recommended extensions when VS Code asks.
2. **Set up (once):** open a terminal (`Ctrl+'`) and run
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
   ```
   This creates `.venv`, installs PyTorch with NVIDIA GPU support and the other packages, and prints whether the GPU is found.
3. **Select the environment:** `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.
4. **Run:** open the **Run and Debug** panel (`Ctrl+Shift+D`), pick a configuration, press **F5**:
   - *1. Quick test (2 epochs)*: check everything works first
   - *2. Full training*
   - *3. Full training (batch 4)*: if you get a GPU "out of memory" error

The run configurations read the data from `Downloads\1512427`. If yours is elsewhere, change the path in `.vscode/launch.json`.

Or run from the terminal:

```powershell
python train_tumor.py --figshare-dir "$HOME\Downloads\1512427" --out test_run --epochs 2
python train_tumor.py --figshare-dir "$HOME\Downloads\1512427" --out models
```

## Run in Google Colab

Open `colab_train_tumor.ipynb` in Colab (*File → Open notebook → GitHub*, tick *Include private repos*), choose **Runtime → T4 GPU**, and run the cells in order. While the repo is private, add a read-only GitHub token as a Colab secret named `GITHUB_TOKEN`.

## Training settings

| Setting | Value |
|---|---|
| Split | **70 / 15 / 15 by patient**: no patient appears in more than one split |
| Optimiser | Adam, learning rate 1e-4, halved when validation loss plateaus (patience 8) |
| Epochs | Up to 150, early stopping after 20 epochs without improvement |
| Batch size | 8 |
| Augmentation | Left–right flip (image + mask), brightness ±0.1, contrast 0.9–1.1 |
| Mixed precision | On when a GPU is used |
| Seed | 42 |

## Outputs (in `--out`, default `models/`)

- `unet_tumor_multitask_best.pt`: best model (by validation loss) and class names
- `history_tumor.csv`: loss, Dice and accuracy per epoch
- `test_results_tumor.json`: **test** Dice (overall and per type), accuracy, macro-F1, confusion matrix, test patient IDs

## Results (held-out test set)

Trained on a Colab T4 GPU (about 27 s per epoch). Early stopping fired at epoch 82, and the best model is the one with the lowest validation loss. Test set: **35 patients, 454 slices**, none of them seen in training or validation.

| | Overall | Meningioma | Glioma | Pituitary |
|---|---|---|---|---|
| Tumour mask Dice | **0.740** ± 0.294 | 0.858 | 0.590 | 0.875 |
| Type recall (sensitivity) | | 0.855 (71/83) | 0.933 (196/210) | 0.988 (159/161) |
| Type precision | | 0.826 (71/86) | 0.942 (196/208) | 0.994 (159/160) |

Type accuracy **0.938**, macro-F1 **0.923**.

Confusion matrix (rows = true type, columns = predicted type):

| | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| **Meningioma** | 71 | 12 | 0 |
| **Glioma** | 13 | 196 | 1 |
| **Pituitary** | 2 | 0 | 159 |

What this shows:
- **Pituitary** tumours are almost always recognised. They sit in a fixed place (the sella) and have a compact shape.
- Most errors are **meningioma vs glioma** (25 of 28 misclassified slices).
- **Glioma outlines are the hardest** (Dice 0.59). Gliomas grow into the surrounding brain and have unclear edges on T1-contrast MRI, so the expert outline itself is harder to reproduce.
- The large spread (± 0.29) comes from small tumour areas on edge slices, where a few pixels of error give a very low Dice.
- Training Dice reached 0.94 while test Dice is 0.74, so outline quality is limited by overfitting on about 2,150 training slices. The next step is a pretrained encoder with stronger augmentation, evaluated on the same patient split.

## Note on results

Many papers on this dataset report very high accuracy after splitting **by slice**, so slices from one patient appear in both training and test sets. This project splits **by patient**, so scores are likely lower but give a fairer estimate for new patients.

## Citation

Cheng, J. (2017). *Brain tumor dataset*. Figshare. https://doi.org/10.6084/m9.figshare.1512427

Cheng, J. et al. (2015). Enhanced performance of brain tumor classification via tumor region augmentation and partition. *PLoS ONE* 10(10).
