"""
Generates fig_prob_deltavariance.png and .pdf
Proportion of bǎ as a function of Δvariance: binned empirical points
(40 observations per bin) + GAM smooth with 95% CI fitted on unbinned data.

Dependencies: pandas, numpy, matplotlib, pygam
    pip install pandas numpy matplotlib pygam

Input: model_data_surprisal.csv (same folder), columns:
    delta_variance : UID metric per minimal pair (raw, unstandardized)
    y              : 1 if the attested sentence is bǎ, 0 if SVO
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (no Type 3)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 14      # ≈ 9.7 pt after scaling to one column
import matplotlib.pyplot as plt
from pygam import LogisticGAM, s

# ─── 0. Parameters ────────────────────────────────────────────────
INPUT_CSV = "model_data_surprisal.csv"
OUTPUT_BASENAME = "figures/fig_prob_deltavariance"   # no extension
BIN_SIZE = 40
XLIM = (-20, 18)

os.makedirs("figures", exist_ok=True)

# ─── 1. Load data ─────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
x = df["delta_variance"].values
y = df["y"].values

# ─── 2. GAM on unbinned data ──────────────────────────────────────
gam = LogisticGAM(s(0)).fit(x.reshape(-1, 1), y)

# ─── 3. Binned empirical points ───────────────────────────────────
order = np.argsort(x)
x_sorted, y_sorted = x[order], y[order]
n_bins = len(x) // BIN_SIZE
bin_x, bin_y = [], []
for i in range(n_bins + 1):
    xs = x_sorted[i * BIN_SIZE:(i + 1) * BIN_SIZE]
    ys = y_sorted[i * BIN_SIZE:(i + 1) * BIN_SIZE]
    if len(xs) == 0:
        continue
    bin_x.append(xs.mean())
    bin_y.append(ys.mean())

# ─── 4. Smooth curve + 95% CI ─────────────────────────────────────
xx = np.linspace(XLIM[0], XLIM[1], 200).reshape(-1, 1)
pred = gam.predict_mu(xx)
conf = gam.confidence_intervals(xx, width=0.95)

# ─── 5. Plot (original size and look; scaled to one column in LaTeX) ──
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.scatter(bin_x, bin_y, s=14, color="black", zorder=3)
ax.plot(xx[:, 0], pred, color="#1a56db", linewidth=1.8, zorder=2)
ax.fill_between(xx[:, 0], conf[:, 0], conf[:, 1],
                color="#1a56db", alpha=0.15, linewidth=0, zorder=1)
ax.set_xlim(*XLIM)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks(range(-20, 19, 5))                   # x ticks: every 5
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])        # y ticks: every 0.2
ax.tick_params(axis="both", labelsize=11.5)        # tick labels
ax.set_xlabel("Δvariance = var(bǎ) − var(SVO)")
ax.set_ylabel("Proportion of bǎ")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

# ─── 6. Save ──────────────────────────────────────────────────────
plt.savefig(OUTPUT_BASENAME + ".png", dpi=300)
plt.savefig(OUTPUT_BASENAME + ".pdf")
print(f"Saved {OUTPUT_BASENAME}.png / .pdf ({len(bin_x)} points)")
