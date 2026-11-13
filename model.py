"""Tumour models (PyTorch): mask + tumour type from one network.

Two options (train_tumor.py --arch):
  unet          MultiTaskUNet below, trained from scratch (the original baseline)
  unetplusplus  PretrainedMultiTask: U-Net++ decoder on an ImageNet-pretrained encoder (default, more accurate)
                other decoders on the same pretrained encoder: unet-pretrained, manet, deeplabv3plus, upernet

MultiTaskUNet: shared encoder, segmentation head and classification head.

Layout (base = 32 filters, input 1 x 256 x 256):
  Encoder      ConvBlock 32 -> 64 -> 128 -> 256, each followed by 2x2 max-pool
  Bottleneck   ConvBlock 512 at 16 x 16, dropout 0.3
  Seg head     4 x [ConvTranspose 2x2 stride 2 -> concat skip -> ConvBlock], then 1x1 conv -> mask logits
  Cls head     global average pool -> Linear 128 + ReLU -> dropout 0.4 -> Linear n_classes (logits)
ConvBlock = [Conv 3x3 (no bias) -> BatchNorm -> ReLU] x 2.  About 7.83 M parameters.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class MultiTaskUNet(nn.Module):
    def __init__(self, n_classes=3, base=32):
        super().__init__()
        chans = [base * 2 ** i for i in range(4)]              # 32, 64, 128, 256
        self.enc = nn.ModuleList()
        cin = 1
        for c in chans:
            self.enc.append(ConvBlock(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chans[-1], base * 16)      # 512
        self.drop = nn.Dropout(0.3)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = base * 16
        for c in reversed(chans):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.dec.append(ConvBlock(c * 2, c))
            cin = c
        self.mask_head = nn.Conv2d(chans[0], 1, 1)

        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base * 16, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.drop(self.bottleneck(x))
        type_logits = self.type_head(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.mask_head(x), type_logits


class PretrainedMultiTask(nn.Module):
    """U-Net++ (or another smp decoder) on an ImageNet-pretrained encoder, plus a tumour-type head.

    The encoder (e.g. EfficientNet-B3) already knows edges, textures and shapes from 1.2 M natural
    images, which matters with only ~2,000 training slices. The type head is smp's auxiliary
    classifier: global average pooling on the deepest encoder features -> dropout -> linear.
    """

    MEAN, STD = 0.449, 0.226          # ImageNet statistics averaged over RGB (grayscale input)

    def __init__(self, n_classes=3, arch="unetplusplus", encoder="efficientnet-b3", pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.create_model(arch, encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                                    in_channels=1, classes=1, aux_params={"classes": n_classes, "dropout": 0.4})

    def forward(self, x):
        mask_logits, type_logits = self.net((x - self.MEAN) / self.STD)
        return mask_logits, type_logits


def build_model(arch="unetplusplus", encoder="efficientnet-b3", n_classes=3, pretrained=True):
    if arch == "unet":                                   # original from-scratch model
        return MultiTaskUNet(n_classes=n_classes)
    return PretrainedMultiTask(n_classes, "unet" if arch == "unet-pretrained" else arch, encoder, pretrained)


def load_checkpoint(path, device):
    """Rebuild whichever model a checkpoint was trained with (old checkpoints = from-scratch U-Net)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck.get("arch", "unet"), ck.get("encoder", ""), len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["classes"]


@torch.no_grad()
def predict(model, images, device, batch=16, tta=True, clean=True):
    """images: NumPy (N, H, W) in [0, 1]. Returns mask probabilities (N, H, W) and type probabilities (N, C).

    tta:   also predict the left-right mirrored slice and average (test-time augmentation)
    clean: keep only the largest connected tumour region in each mask
    """
    import numpy as np
    from data_utils import keep_largest_component
    model.eval()
    masks, types = [], []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).float().unsqueeze(1).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            m, t = model(x)
            if tta:
                mf, tf = model(x.flip(-1))
                m = (torch.sigmoid(m.float()) + torch.sigmoid(mf.float().flip(-1))) / 2
                t = (torch.softmax(t.float(), 1) + torch.softmax(tf.float(), 1)) / 2
            else:
                m, t = torch.sigmoid(m.float()), torch.softmax(t.float(), 1)
        masks.append(m[:, 0].cpu().numpy())
        types.append(t.cpu().numpy())
    masks = np.concatenate(masks)
    if clean:
        masks = np.stack([keep_largest_component(p) for p in masks])
    return masks, np.concatenate(types)
