# Brain Tumour Segmentation and Classification (Multi-task U-Net)

One model that **outlines the tumour** and **predicts its type** on 2D brain MRI slices, built with TensorFlow/Keras.

> ⚠️ Research and education only. Not validated for clinical use.

## Model

A U-Net whose encoder is **shared** by two heads:

| Part | Details |
|---|---|
| Input | 256 × 256 × 1 MRI slice, scaled to [0, 1] per image |
| Shared encoder | 4 blocks (two 3×3 conv + batch norm + ReLU, then max-pool); 32 → 256 filters |
| Bottleneck | 512 filters at 16 × 16, dropout 0.3 |
| Segmentation head | U-Net decoder with skip links → tumour mask (sigmoid); loss **Dice + BCE** |
| Classification head | Global average pooling → dense 128 + dropout → softmax over 3 types; loss **cross-entropy** (weight 0.5) |

Sharing the encoder lets the mask task guide the classifier towards the tumour region.

## Data

**Jun Cheng's Figshare brain tumour dataset** (CC BY 4.0): 3,064 T1-weighted contrast-enhanced slices from 233 patients. Each slice has a radiologist-drawn tumour mask, tumour type and patient ID.

| Type | Slices |
|---|---|
| Meningioma | 708 |
| Glioma | 1,426 |
| Pituitary | 930 |

Download: <https://doi.org/10.6084/m9.figshare.1512427>. Unzip the four zip files anywhere (for example `data/figshare/`); the script searches subfolders and skips `cvind.mat`.

Every slice contains a tumour, so the model has **no "no tumour" class**.

You can also use your own data: image and mask folders with matching filenames plus a `labels.csv` with columns `filename,label,patient_id`.

## Training

```bash
pip install -r requirements.txt
python train_tumor.py --figshare-dir data/figshare --out models
# quick check first:
python train_tumor.py --figshare-dir data/figshare --out models --epochs 2
```

| Setting | Value |
|---|---|
| Split | **70 / 15 / 15 by patient** (no patient in more than one split) |
| Optimiser | Adam, learning rate 1e-4, halved on plateau |
| Epochs | Up to 150, early stopping (patience 20) on validation loss |
| Augmentation | Left–right flip, small brightness/contrast changes |
| Batch size | 8 (use `--batch 4` if GPU memory is low) |

## Outputs

- `models/unet_tumor_multitask_best.keras`: best model
- `models/history_tumor.csv`: training curves
- `models/test_results_tumor.json`: test Dice (overall and per type), accuracy, macro-F1, confusion matrix, and the test patient IDs

## Note on results

Many papers on this dataset report very high accuracy after splitting **by slice**, which lets slices from the same patient appear in both training and test sets. This project splits **by patient**, so scores are likely lower but give a fairer estimate of performance on new patients.

## Citation

Cheng, J. (2017). *Brain tumor dataset*. Figshare. https://doi.org/10.6084/m9.figshare.1512427

Cheng, J. et al. (2015). Enhanced performance of brain tumor classification via tumor region augmentation and partition. *PLoS ONE* 10(10).
