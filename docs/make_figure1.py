"""Figure 1 for the brain-tumor-multitask paper: pipeline, architecture, training, inference, results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Rectangle
from scipy import ndimage

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.linewidth": 0.6,
                     "mathtext.fontset": "dejavuserif"})
C_ENC, C_DEC, C_HEAD, C_CLS = "#c9b99a", "#a9c3a0", "#9db4cf", "#d8a7a0"
INK = "#222222"

# ---------------------------------------------------------------- synthetic phantom slice
rng = np.random.default_rng(3)
N = 256
yy, xx = np.mgrid[:N, :N]
def ell(cx, cy, rx, ry): return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
head = ell(128, 128, 100, 118) < 1
brain = ell(128, 128, 90, 108) < 1
tex = ndimage.gaussian_filter(rng.normal(size=(N, N)), 3)
img = np.zeros((N, N))
img[head] = 0.85
img[brain] = 0.42 + 0.06 * tex[brain] / tex.std()
vent = (ell(115, 120, 9, 22) < 1) | (ell(141, 120, 9, 22) < 1)
img[vent] = 0.12
d = np.sqrt(ell(165, 95, 1, 1))
tumour = d < 22
img[(d < 22) & (d > 15)] = 0.95                       # enhancing rim
img[d <= 15] = 0.30                                   # necrotic centre
img = ndimage.gaussian_filter(img, 1.0) + rng.normal(0, 0.015, (N, N))
raw = (img * 900 + 40).clip(0) ; raw[100:102, 30:32] = 4000          # a few very bright pixels
gt = tumour.astype(float)

def cuboid(ax, x, y, w, h, depth, color, label=None, sub=None):
    dx, dy = depth * 0.6, depth * 0.45
    ax.add_patch(Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], closed=True, fc=color, ec=INK, lw=0.5))
    ax.add_patch(Polygon([[x, y + h], [x + dx, y + h + dy], [x + w + dx, y + h + dy], [x + w, y + h]], closed=True,
                         fc=matplotlib.colors.to_hex(np.array(matplotlib.colors.to_rgb(color)) * 0.85 + 0.15), ec=INK, lw=0.5))
    ax.add_patch(Polygon([[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]], closed=True,
                         fc=matplotlib.colors.to_hex(np.array(matplotlib.colors.to_rgb(color)) * 0.75), ec=INK, lw=0.5))
    if label:
        ax.text(x + w / 2, y - 0.25, label, ha="center", va="top", fontsize=5.6)
    if sub:
        ax.text(x + w / 2 + dx / 2, y + h + dy + 0.12, sub, ha="center", va="bottom", fontsize=5.6)

def arrow(ax, p, q, style="-|>", ls="-", color=INK, lw=0.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))

def node(ax, x, y, text, color, w=1.05, h=0.62, fs=5.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=INK, lw=0.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)

def panel_title(ax, letter, title, x=0.0, y=1.02):
    import matplotlib.transforms as mt
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.text(x, y, title, transform=mt.offset_copy(ax.transAxes, fig=ax.figure, x=19, y=0, units="points"),
            fontsize=8.2, style="italic", va="bottom")

fig = plt.figure(figsize=(8.27, 11.2))
fig.text(0.04, 0.985, "BRAIN-TUMOR-MULTITASK  ·  MSc PROJECT  ·  RESEARCH PROTOTYPE, NOT A MEDICAL DEVICE",
         fontsize=6.5, color="#666", va="top")
fig.text(0.04, 0.968, "Multi-task brain tumour segmentation and classification on T1-weighted contrast-enhanced MRI",
         fontsize=11.5, va="top")
fig.text(0.04, 0.949, "Chandu Venkat Sai Narne", fontsize=7.5, style="italic", fontweight="bold", va="top")
fig.text(0.245, 0.949, "U-Net++ decoder on ImageNet-pretrained EfficientNet-B3  →  tumour mask + tumour type from one network",
         fontsize=7, color="#444", va="top")
fig.add_artist(matplotlib.lines.Line2D([0.04, 0.96], [0.94, 0.94], color=INK, lw=0.6))

# ---------------------------------------------------------------- (a) preprocessing
ax_a = fig.add_axes([0.04, 0.775, 0.92, 0.135]); ax_a.axis("off")
panel_title(ax_a, "a", "Input and preprocessing")
sub = [fig.add_axes([0.05 + i * 0.235, 0.785, 0.17, 0.115]) for i in range(4)]
sub[0].imshow(raw, cmap="gray", vmin=0, vmax=raw.max()); sub[0].set_title("(i) stored slice, full range", fontsize=6, y=-0.2)
h, b = np.histogram(raw[raw > 5], bins=80)
sub[1].bar(b[:-1], h / h.max(), width=b[1] - b[0], color="#9aa9bd")
lo, hi = np.percentile(raw, [0.5, 99.5])
sub[1].axvspan(lo, hi, color="#e8dcc2", alpha=0.7, zorder=0)
sub[1].plot([0, lo, hi, raw.max()], [0, 0, 1, 1], color=INK, lw=0.8)
sub[1].set_xlim(0, raw.max()); sub[1].set_yticks([0, 1]); sub[1].tick_params(labelsize=5)
sub[1].set_xlabel("stored intensity", fontsize=5.5, labelpad=1)
sub[1].text(hi * 1.05, 0.55, "0.5–99.5 %\nclip, scale to [0, 1]", fontsize=5.2)
sub[1].set_title("(ii) histogram and robust window", fontsize=6, y=-0.36)
norm = np.clip((raw - lo) / (hi - lo), 0, 1)
sub[2].imshow(norm, cmap="gray"); sub[2].set_title("(iii) 256 × 256, [0, 1]", fontsize=6, y=-0.2)
sub[3].imshow(norm, cmap="gray"); sub[3].contour(gt, levels=[0.5], colors="#3ddc5a", linewidths=0.8)
sub[3].set_title("(iv) expert mask + type label", fontsize=6, y=-0.2)
for s in (sub[0], sub[2], sub[3]):
    s.set_xticks([]); s.set_yticks([])
for i in range(3):
    fig.add_artist(FancyArrowPatch((0.225 + i * 0.235, 0.842), (0.278 + i * 0.235, 0.842), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7, lw=0.6, color=INK))

# ---------------------------------------------------------------- (b) architecture
ax_b = fig.add_axes([0.04, 0.46, 0.92, 0.275]); ax_b.set_xlim(0, 20); ax_b.set_ylim(0, 10); ax_b.axis("off")
panel_title(ax_b, "b", "Network: EfficientNet-B3 encoder, U-Net++ nested decoder, two heads", y=1.01)
ax_in = fig.add_axes([0.045, 0.62, 0.065, 0.065]); ax_in.imshow(norm, cmap="gray"); ax_in.axis("off")
ax_b.text(0.62, 6.25, "1×256²", ha="center", fontsize=5.6)
enc = [("Stem\n+MBConv1", "40", "128²"), ("MBConv6\nstage 2", "32", "64²"), ("MBConv6\nstage 3", "48", "32²"),
       ("MBConv6\nst. 4–5", "136", "16²"), ("MBConv6\nst. 6–7", "384", "8²")]
gx = lambda j: 3.0 + j * 2.05
gy = lambda i: 8.7 - i * 1.75
for i, (name, ch, sz) in enumerate(enc):
    node(ax_b, gx(0), gy(i), f"$X^{{{i},0}}$  {name}\n{ch} ch · {sz}", C_ENC, w=1.75, h=0.95, fs=5.2)
    if i:
        arrow(ax_b, (gx(0), gy(i - 1) - 0.48), (gx(0), gy(i) + 0.48))
arrow(ax_b, (1.25, 8.7), (gx(0) - 0.88, gy(0)))
dec_ch = {1: "256", 2: "128", 3: "64", 4: "32"}
for j in range(1, 5):
    for i in range(0, 5 - j):
        x, y = gx(j), gy(i)
        node(ax_b, x, y, f"$X^{{{i},{j}}}$", C_DEC, w=0.95, h=0.6)
        arrow(ax_b, (gx(j - 1) + (0.88 if j == 1 else 0.48), gy(i + 1) + 0.2), (x - 0.4, y - 0.3), color="#5a7d52")
        arrow(ax_b, (gx(j - 1) + (0.88 if j == 1 else 0.48), y), (x - 0.48, y), color="#888", ls=(0, (2, 1.5)))
for j in range(1, 5):
    ax_b.text(gx(j), gy(0) + 0.55, f"{dec_ch[j]} ch" if j < 4 else "32 ch", ha="center", fontsize=5)
# segmentation head
x_last = gx(4)
node(ax_b, x_last + 2.05, gy(0), "Conv 3×3 +\n×2 upsample\n16 ch", C_DEC, w=1.35, h=0.95, fs=5.2)
arrow(ax_b, (x_last + 0.48, gy(0)), (x_last + 1.37, gy(0)))
node(ax_b, x_last + 3.85, gy(0), "Conv 1×1\n→ 1 logit", C_HEAD, w=1.2, h=0.8, fs=5.2)
arrow(ax_b, (x_last + 2.73, gy(0)), (x_last + 3.25, gy(0)))
ax_mask = fig.add_axes([0.83, 0.66, 0.065, 0.065]); ax_mask.imshow(ndimage.gaussian_filter(gt, 2), cmap="magma"); ax_mask.axis("off")
ax_b.text(19.35, 7.0, "P(tumour)\nper pixel", ha="center", fontsize=5.4)
arrow(ax_b, (x_last + 4.45, gy(0)), (18.55, gy(0)))
# classification head
cy = gy(4)
node(ax_b, gx(1) + 0.35, cy, "Global\navg pool\n384", C_CLS, w=1.15, h=0.95, fs=5.2)
node(ax_b, gx(2) + 0.35, cy, "Dropout\n0.4", C_CLS, w=1.0, h=0.7, fs=5.2)
node(ax_b, gx(3) + 0.35, cy, "Linear\n384 → 3", C_CLS, w=1.1, h=0.7, fs=5.2)
node(ax_b, gx(4) + 0.6, cy, "softmax\nmeningioma\nglioma\npituitary", C_CLS, w=1.35, h=1.2, fs=5.0)
arrow(ax_b, (gx(0) + 0.88, cy), (gx(1) - 0.23, cy))
for a_, b_ in ((gx(1) + 0.93, gx(2) - 0.15), (gx(2) + 0.85, gx(3) - 0.2), (gx(3) + 0.9, gx(4) - 0.07)):
    arrow(ax_b, (a_, cy), (b_, cy))
ax_b.text(gx(2) + 0.35, cy - 0.85, "classification head on the deepest encoder features (shared encoder)", ha="center",
          fontsize=5.4, style="italic")
# legend
lx = 13.3
for k, (c, t) in enumerate([(C_ENC, "encoder node (pretrained)"), (C_DEC, "decoder node: concat → 2×[conv3×3, BN, ReLU]"),
                            (C_HEAD, "segmentation output"), (C_CLS, "classification head")]):
    ax_b.add_patch(Rectangle((lx, 6.45 - k * 0.45), 0.35, 0.28, fc=c, ec=INK, lw=0.4))
    ax_b.text(lx + 0.5, 6.59 - k * 0.45, t, fontsize=5.4, va="center")
arrow(ax_b, (lx, 4.65), (lx + 0.4, 4.65), color="#5a7d52"); ax_b.text(lx + 0.5, 4.65, "×2 upsample from the level below", fontsize=5.4, va="center")
arrow(ax_b, (lx, 4.25), (lx + 0.4, 4.25), color="#888", ls=(0, (2, 1.5))); ax_b.text(lx + 0.5, 4.25, "dense skip: all earlier nodes of the same level", fontsize=5.4, va="center")
ax_b.text(lx, 3.75, "13.63 M parameters · input 1 channel (first conv adapted)\nImageNet mean/std applied inside the model",
          fontsize=5.4, va="center", color="#444")
# MBConv inset
ax_mb = fig.add_axes([0.62, 0.468, 0.33, 0.085]); ax_mb.set_xlim(0, 10); ax_mb.set_ylim(0, 3); ax_mb.axis("off")
ax_mb.text(0, 2.75, "MBConv block (EfficientNet)", fontsize=6, style="italic")
for k, (t, c) in enumerate([("1×1 expand\n×6", "#e6dcc6"), ("depthwise\nk×k", "#e6dcc6"), ("SE\nsqueeze-excite", "#b8d4b0"),
                            ("1×1 project", "#e6dcc6")]):
    ax_mb.add_patch(FancyBboxPatch((0.2 + k * 2.4, 0.9), 1.9, 1.1, boxstyle="round,pad=0.02,rounding_size=0.15",
                                   fc=c, ec=INK, lw=0.5))
    ax_mb.text(1.15 + k * 2.4, 1.45, t, ha="center", va="center", fontsize=5.2)
    if k:
        arrow(ax_mb, (2.1 + (k - 1) * 2.4, 1.45), (0.2 + k * 2.4, 1.45))
ax_mb.annotate("", xy=(9.6, 2.15), xytext=(0.2, 2.15), arrowprops=dict(arrowstyle="-|>", lw=0.5, color="#777",
               connectionstyle="arc3,rad=-0.12"))
ax_mb.text(4.9, 2.42, "identity skip (stride 1, same channels)", ha="center", fontsize=5, color="#555")
ax_mb.text(0.2, 0.35, r"SE: $s=\sigma(W_2\,\delta(W_1\,\mathrm{GAP}(F)))$, channels rescaled by $s$; BN + SiLU after convs",
           fontsize=5.2)

# ---------------------------------------------------------------- (c) training: augmentation + loss
ax_c = fig.add_axes([0.04, 0.245, 0.45, 0.17]); ax_c.axis("off")
panel_title(ax_c, "c", "Training: augmentation and multi-task loss")
augs = [("flip ↔", np.fliplr(norm)), ("rotate 12°", ndimage.rotate(norm, 12, reshape=False)),
        ("zoom 1.12", ndimage.zoom(norm, 1.12)[15:271, 15:271]), ("gamma 0.7", norm ** 0.7)]
for k, (t, im) in enumerate(augs):
    a = fig.add_axes([0.045 + k * 0.111, 0.335, 0.1, 0.07]); a.imshow(im, cmap="gray"); a.axis("off")
    a.set_title(t, fontsize=5.6, pad=1.5)
ax_c.text(0.01, 0.38,
          r"$\mathcal{L}=\mathrm{BCE}(\hat m,m)+\frac{1}{B}\sum_b\left(1-\mathrm{Dice}_b\right)\;+\;0.5\,\mathrm{CE}_{w,\epsilon}(\hat y,y)$", fontsize=7.2, transform=ax_c.transAxes)
ax_c.text(0.01, 0.17, r"$\mathrm{Dice}_b=\frac{2\sum p\,t+1}{\sum p+\sum t+1}$ per slice;  class weights $w$ for imbalance",
          fontsize=6, transform=ax_c.transAxes)
ax_c.text(0.01, -0.02, "AdamW, one-cycle LR (max 3e-4), batch 16, AMP, 70/15/15 split by patient,\n"
          "early stop 20 epochs, keep best mean of validation Dice and accuracy", fontsize=5.7, color="#444",
          transform=ax_c.transAxes)

# ---------------------------------------------------------------- (d) inference: TTA + largest component
ax_d = fig.add_axes([0.53, 0.245, 0.43, 0.17]); ax_d.axis("off")
panel_title(ax_d, "d", "Inference: flip TTA and clean-up")
prob = ndimage.gaussian_filter(gt, 3) * 0.95
speck = np.exp(-(((xx - 70) / 4) ** 2 + ((yy - 190) / 4) ** 2)) * 0.7
p1, p2 = np.clip(prob + speck + rng.normal(0, 0.02, (N, N)), 0, 1), np.clip(prob + rng.normal(0, 0.02, (N, N)), 0, 1)
pav = (p1 + p2) / 2
lab, n = ndimage.label(pav > 0.5)
sizes = ndimage.sum(pav > 0.5, lab, range(1, n + 1))
clean = lab == (1 + int(np.argmax(sizes)))
for k, (t, im, cm) in enumerate([("f(x)", p1, "magma"), ("flip f(flip x)", p2, "magma"),
                                 ("average", pav, "magma"), ("largest region", clean, "gray")]):
    a = fig.add_axes([0.535 + k * 0.105, 0.335, 0.095, 0.07]); a.imshow(im, cmap=cm, vmin=0, vmax=1); a.axis("off")
    a.set_title(t, fontsize=5.6, pad=1.5)
    if k == 3:
        a.contour(gt, levels=[0.5], colors="#3ddc5a", linewidths=0.6)
ax_d.text(0.01, 0.38, r"$\bar P=\frac{1}{2}\left[f(x)+T^{-1}f(Tx)\right]$,  $T$ = left–right flip", fontsize=7,
          transform=ax_d.transAxes)
ax_d.text(0.01, 0.17, r"mask $=$ largest 8-connected region of $\bar P>0.5$;  type $=\arg\max$ of averaged softmax",
          fontsize=6, transform=ax_d.transAxes)
ax_d.text(0.01, -0.02, "removes isolated false-positive specks (one tumour per slice in this dataset)",
          fontsize=5.7, color="#444", transform=ax_d.transAxes)

# ---------------------------------------------------------------- (e) results: Dice per type, v1 vs v2
ax_e = fig.add_axes([0.08, 0.07, 0.38, 0.13])
panel_title(ax_e, "e", "Test results: Dice per tumour type", x=-0.11, y=1.08)
types = ["meningioma", "glioma", "pituitary", "all"]
v1 = [0.858, 0.590, 0.875, 0.740]; v2 = [0.913, 0.707, 0.895, 0.811]
xs = np.arange(4)
ax_e.bar(xs - 0.19, v1, 0.36, color="#c9c9c9", ec=INK, lw=0.4, label="v1 U-Net (scratch)")
ax_e.bar(xs + 0.19, v2, 0.36, color="#7b9cc4", ec=INK, lw=0.4, label="v2 U-Net++ / EffNet-B3")
for x, a_, b_ in zip(xs, v1, v2):
    ax_e.text(x - 0.19, a_ + 0.015, f"{a_:.2f}", ha="center", fontsize=5.2)
    ax_e.text(x + 0.19, b_ + 0.015, f"{b_:.2f}", ha="center", fontsize=5.2)
ax_e.set_xticks(xs, types, fontsize=6); ax_e.set_ylim(0, 1.05); ax_e.set_ylabel("Dice", fontsize=6)
ax_e.tick_params(labelsize=5.5); ax_e.spines[["top", "right"]].set_visible(False)
ax_e.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
ax_e.text(1.0, -0.22, "35 held-out patients, 454 slices", transform=ax_e.transAxes, ha="right", fontsize=5.4, color="#555")

# ---------------------------------------------------------------- (f) confusion matrix v2
ax_f = fig.add_axes([0.62, 0.07, 0.17, 0.13])
panel_title(ax_f, "f", "Tumour type, v2 confusion matrix", x=-0.55, y=1.08)
cm = np.array([[71, 10, 2], [9, 200, 1], [0, 0, 161]])
rec = cm / cm.sum(1, keepdims=True)
ax_f.imshow(rec, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax_f.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6.5, color="white" if rec[i, j] > 0.6 else INK)
lab3 = ["mening.", "glioma", "pituit."]
ax_f.set_xticks(range(3), lab3, fontsize=5.5); ax_f.set_yticks(range(3), lab3, fontsize=5.5)
ax_f.set_xlabel("predicted", fontsize=6); ax_f.set_ylabel("true", fontsize=6)
ax_f.text(3.0, 0.0, "recall 0.855\n\nrecall 0.952\n\nrecall 1.000", fontsize=5.4, va="top", ha="left",
          transform=ax_f.transData)
ax_f.text(3.0, 2.6, "accuracy 0.952\nmacro-F1 0.938", fontsize=5.8, va="top", ha="left", fontweight="bold")

fig.add_artist(matplotlib.lines.Line2D([0.04, 0.96], [0.025, 0.025], color="#999", lw=0.4))
fig.text(0.04, 0.012, "github.com/ChanduVenkatSaiNarne/brain-tumor-multitask", fontsize=5.6, color="#666")
fig.text(0.96, 0.012, "PyTorch · segmentation-models-pytorch · Figshare (Cheng 2017) · Google Colab T4",
         fontsize=5.6, color="#666", ha="right")
fig.savefig("figure1.png", dpi=300)
fig.savefig("figure1_preview.png", dpi=110)
print("saved")
