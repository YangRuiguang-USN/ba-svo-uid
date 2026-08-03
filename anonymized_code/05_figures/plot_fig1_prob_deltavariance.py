"""
Génère fig_prob_deltavariance.png ET .pdf :
Proportion de bǎ en fonction de Δvariance, avec points groupés (bins de 40
observations, suivant l'approche de binning utilisée dans des études
antérieures sur l'UID et l'ordre des mots) et une courbe lissée par GAM
(Generalized Additive Model) ajustée sur les données brutes (1991 lignes).

Dépendances : pandas, numpy, matplotlib, pygam
    pip install pandas numpy matplotlib pygam --break-system-packages

Entrée attendue : model_data_surprisal.csv, avec au moins les colonnes
    - delta_variance : métrique UID par paire minimale
    - y              : 1 si la phrase attestée est en bǎ, 0 si SVO
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygam import LogisticGAM, s

# ─── 0. Paramètres ────────────────────────────────────────────────────────
INPUT_CSV = "model_data_surprisal.csv"
OUTPUT_BASENAME = "figures/fig_prob_deltavariance"   # sans extension
BIN_SIZE = 40          # nombre d'observations par point
XLIM = (-20, 18)        # bornes d'affichage : ~99.7% des données

os.makedirs("figures", exist_ok=True)

# ─── 1. Chargement des données ────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
x = df["delta_variance"].values
y = df["y"].values

# ─── 2. Ajustement du GAM sur les données brutes (non groupées) ──────────
gam = LogisticGAM(s(0)).fit(x.reshape(-1, 1), y)

# ─── 3. Points empiriques groupés par bins de taille fixe ────────────────
order = np.argsort(x)
x_sorted, y_sorted = x[order], y[order]
n_bins = len(x) // BIN_SIZE
bin_x, bin_y = [], []
for i in range(n_bins + 1):
    xs = x_sorted[i * BIN_SIZE:(i + 1) * BIN_SIZE]
    ys = y_sorted[i * BIN_SIZE:(i + 1) * BIN_SIZE]
    if len(xs) == 0:
        continue
    bin_x.append(xs.mean())      # moyenne de Δvariance dans le bin
    bin_y.append(ys.mean())      # proportion empirique de bǎ dans le bin

# ─── 4. Courbe lissée + intervalle de confiance à 95% ────────────────────
xx = np.linspace(XLIM[0], XLIM[1], 200).reshape(-1, 1)
pred = gam.predict_mu(xx)
conf = gam.confidence_intervals(xx, width=0.95)

# ─── 5. Tracé ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.scatter(bin_x, bin_y, s=14, color="black", zorder=3)
ax.plot(xx[:, 0], pred, color="#1a56db", linewidth=1.8, zorder=2)
ax.fill_between(xx[:, 0], conf[:, 0], conf[:, 1],
                 color="#1a56db", alpha=0.15, linewidth=0, zorder=1)
ax.set_xlim(*XLIM)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel(r"$\Delta$variance")
ax.set_ylabel(r"Proportion of $b\check{a}$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

# ─── 6. Sauvegarde en PNG (aperçu) et PDF (vectoriel, pour LaTeX) ────────
plt.savefig(OUTPUT_BASENAME + ".png", dpi=300)
plt.savefig(OUTPUT_BASENAME + ".pdf")
print(f"Figures sauvegardées -> {OUTPUT_BASENAME}.png / .pdf ({len(bin_x)} points affichés)")
