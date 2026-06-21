"""
Genera gráficas individuales para cada sección del estudio web.
Salida: WebP en src/content/analytics/images/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os

# ─── Destino ───────────────────────────────────────────────────────────────
DEST = "/home/javilazaro/Documents/web-javilazaro/src/content/analytics/images"

# ─── Paleta ────────────────────────────────────────────────────────────────
BG_FIG   = "#0F0F1A"
BG_AX    = "#16162A"
GRID_COL = "#2A2A3E"
TITLE_C  = "#E8E8F0"
TEXT_C   = "#B0B0C8"
TF_COL   = "#4285F4"
JAX_COL  = "#EA4335"

FASES_CORTAS = [
    "InputData",
    "Instanciación",
    "Sample prior",
    "Analyzer",
    "BudgetOpt.",
]

PERFILES = ["Pequeño\n(2g/3ch)", "Mediano\n(10g/8ch)", "Grande\n(50g/15ch)"]
PERFILES_SHORT = ["Pequeño", "Mediano", "Grande"]

TIEMPOS = {
    "TF":  [[0.186, 0.655, 0.689, 7.280, 1.748],
            [0.311, 0.667, 0.668, 7.229, 1.937],
            [0.573, 0.666, 0.680, 7.216, 3.705]],
    "JAX": [[0.187, 2.145, 4.068, 0.249, 1.580],
            [0.317, 2.264, 4.501, 0.275, 1.758],
            [0.579, 2.372, 4.656, 0.281, 3.391]],
}

MEMORIAS = {
    "TF":  [[0.35, 3.12, 0.81, 2.93, 1.02],
            [0.66, 3.14, 0.85, 2.94, 1.03],
            [4.90, 3.95, 0.98, 3.78, 1.02]],
    "JAX": [[0.35, 6.86, 6.29, 0.76, 1.89],
            [0.66, 6.89, 6.76, 0.76, 1.87],
            [4.90, 6.89, 6.91, 0.77, 1.87]],
}

TOTALES_T = {"TF": [10.558, 10.812, 12.841], "JAX": [8.229, 9.114, 11.280]}
TOTALES_M = {"TF": [8.23, 8.62, 14.63],      "JAX": [16.16, 16.95, 21.33]}


def estilo(ax, titulo=""):
    ax.set_facecolor(BG_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    if titulo:
        ax.set_title(titulo, color=TITLE_C, fontsize=12, fontweight="bold", pad=10)


def guardar(fig, nombre, quality=88):
    path = os.path.join(DEST, nombre)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf)
    img.save(path, "WEBP", quality=quality, method=6)
    print(f"  Guardado: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Sección 4: Pipeline completo TF vs JAX (tiempo + memoria)
# Muestra los 3 perfiles lado a lado con barras agrupadas
# ═══════════════════════════════════════════════════════════════════════════

print("Generando grafica-pipeline-tiempo-memoria.webp ...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG_FIG)
fig.suptitle("Pipeline completo — Tiempo y memoria por fase (TF vs JAX)",
             color=TITLE_C, fontsize=14, fontweight="bold", y=0.98)

x = np.arange(len(FASES_CORTAS))
w = 0.38

for i, perfil in enumerate(PERFILES_SHORT):
    # — Fila 0: tiempo —
    ax = axes[0, i]
    b_tf  = ax.bar(x - w/2, TIEMPOS["TF"][i],  w, label="TF",  color=TF_COL,  alpha=0.9)
    b_jax = ax.bar(x + w/2, TIEMPOS["JAX"][i], w, label="JAX", color=JAX_COL, alpha=0.9)
    for b in list(b_tf) + list(b_jax):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.06,
                f"{h:.2f}", ha="center", va="bottom", color=TEXT_C, fontsize=7.5)
    estilo(ax, f"Tiempo — {perfil}")
    ax.set_xticks(x)
    ax.set_xticklabels(FASES_CORTAS, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Tiempo (s)")
    ax.legend(fontsize=8, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C)

    # — Fila 1: memoria —
    ax = axes[1, i]
    b_tf  = ax.bar(x - w/2, MEMORIAS["TF"][i],  w, label="TF",  color=TF_COL,  alpha=0.9)
    b_jax = ax.bar(x + w/2, MEMORIAS["JAX"][i], w, label="JAX", color=JAX_COL, alpha=0.9)
    for b in list(b_tf) + list(b_jax):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", color=TEXT_C, fontsize=7.5)
    estilo(ax, f"Memoria delta — {perfil}")
    ax.set_xticks(x)
    ax.set_xticklabels(FASES_CORTAS, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Memoria delta (MB)")
    ax.legend(fontsize=8, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C)

fig.tight_layout(rect=[0, 0, 1, 0.96])
guardar(fig, "grafica-pipeline-tiempo-memoria.webp")


# ═══════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Sección 5: MCMC producción — TF vs JAX (tiempo y memoria)
# ═══════════════════════════════════════════════════════════════════════════

print("Generando grafica-mcmc-produccion.webp ...")

MCMC_PERFILES = ["Pequeño\n(2g/3ch)", "Mediano\n(10g/8ch)"]
MCMC_T  = {"TF": [234, 391], "JAX": [146, 356]}
MCMC_MS = {"TF": [39, 65],   "JAX": [24, 59]}
MCMC_M  = {"TF": [263, 266], "JAX": [91, 95]}

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(BG_FIG)
fig.suptitle("MCMC producción (4 cadenas × 1.500 steps) — TF vs JAX",
             color=TITLE_C, fontsize=14, fontweight="bold", y=1.01)

x = np.arange(len(MCMC_PERFILES))
w = 0.35

# Tiempo total
ax = axes[0]
b_tf  = ax.bar(x - w/2, MCMC_T["TF"],  w, label="TF",  color=TF_COL,  alpha=0.9)
b_jax = ax.bar(x + w/2, MCMC_T["JAX"], w, label="JAX", color=JAX_COL, alpha=0.9)
for bt, bj, pt, pj in zip(b_tf, b_jax, MCMC_T["TF"], MCMC_T["JAX"]):
    mejora = round((pt - pj) / pt * 100)
    ax.text(bj.get_x() + bj.get_width()/2, pj + 3,
            f"−{mejora}%", ha="center", va="bottom", color="#34A853",
            fontsize=10, fontweight="bold")
    ax.text(bt.get_x() + bt.get_width()/2, pt + 3,
            f"{pt}s", ha="center", va="bottom", color=TEXT_C, fontsize=9)
    ax.text(bj.get_x() + bj.get_width()/2, -18,
            f"{pj}s", ha="center", va="top", color=TEXT_C, fontsize=9,
            clip_on=False)
estilo(ax, "Tiempo total (s)")
ax.set_xticks(x); ax.set_xticklabels(MCMC_PERFILES, fontsize=9)
ax.set_ylabel("Segundos")
ax.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C)

# ms/step
ax = axes[1]
b_tf  = ax.bar(x - w/2, MCMC_MS["TF"],  w, label="TF",  color=TF_COL,  alpha=0.9)
b_jax = ax.bar(x + w/2, MCMC_MS["JAX"], w, label="JAX", color=JAX_COL, alpha=0.9)
for bt, bj, pt, pj in zip(b_tf, b_jax, MCMC_MS["TF"], MCMC_MS["JAX"]):
    for b, v in [(bt, pt), (bj, pj)]:
        ax.text(b.get_x() + b.get_width()/2, v + 0.5,
                f"{v} ms", ha="center", va="bottom", color=TEXT_C, fontsize=9)
estilo(ax, "Velocidad por step (ms/step)")
ax.set_xticks(x); ax.set_xticklabels(MCMC_PERFILES, fontsize=9)
ax.set_ylabel("ms / step")
ax.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C)

# Memoria
ax = axes[2]
b_tf  = ax.bar(x - w/2, MCMC_M["TF"],  w, label="TF",  color=TF_COL,  alpha=0.9)
b_jax = ax.bar(x + w/2, MCMC_M["JAX"], w, label="JAX", color=JAX_COL, alpha=0.9)
for bt, bj, pt, pj in zip(b_tf, b_jax, MCMC_M["TF"], MCMC_M["JAX"]):
    mejora = round((pt - pj) / pt * 100)
    ax.text(bj.get_x() + bj.get_width()/2, pj + 2,
            f"−{mejora}%", ha="center", va="bottom", color="#34A853",
            fontsize=10, fontweight="bold")
    ax.text(bt.get_x() + bt.get_width()/2, pt + 2,
            f"{pt} MB", ha="center", va="bottom", color=TEXT_C, fontsize=9)
estilo(ax, "Memoria pico (MB)")
ax.set_xticks(x); ax.set_xticklabels(MCMC_PERFILES, fontsize=9)
ax.set_ylabel("MB")
ax.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C)

fig.tight_layout()
guardar(fig, "grafica-mcmc-produccion.webp")


# ═══════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Sección 6.3: Spotlight Analyzer — TF 7.2s vs JAX 0.25s
# ═══════════════════════════════════════════════════════════════════════════

print("Generando grafica-analyzer-spotlight.webp ...")

T_ANALYZER_TF  = [7.280, 7.229, 7.216]
T_ANALYZER_JAX = [0.249, 0.275, 0.281]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG_FIG)
fig.suptitle("Analyzer (incremental_outcome) — TF vs JAX por perfil",
             color=TITLE_C, fontsize=14, fontweight="bold", y=1.01)

x = np.arange(3)
w = 0.35

# Barras comparativas
ax = axes[0]
b_tf  = ax.bar(x - w/2, T_ANALYZER_TF,  w, label="TF",  color=TF_COL,  alpha=0.9)
b_jax = ax.bar(x + w/2, T_ANALYZER_JAX, w, label="JAX", color=JAX_COL, alpha=0.9)
for bt, bj, vt, vj in zip(b_tf, b_jax, T_ANALYZER_TF, T_ANALYZER_JAX):
    ax.text(bt.get_x() + bt.get_width()/2, vt + 0.05,
            f"{vt:.2f}s", ha="center", va="bottom", color=TEXT_C, fontsize=9)
    ax.text(bj.get_x() + bj.get_width()/2, vj + 0.05,
            f"{vj:.2f}s", ha="center", va="bottom", color=TEXT_C, fontsize=9)
estilo(ax, "Tiempo Analyzer por perfil")
ax.set_xticks(x)
ax.set_xticklabels(PERFILES_SHORT, fontsize=10)
ax.set_ylabel("Tiempo (s)")
ax.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C)

# Ratio de mejora
ax = axes[1]
ratios = [vt / vj for vt, vj in zip(T_ANALYZER_TF, T_ANALYZER_JAX)]
bars = ax.bar(x, ratios, color=["#34A853"] * 3, alpha=0.9, width=0.5)
for b, r in zip(bars, ratios):
    ax.text(b.get_x() + b.get_width()/2, r + 0.3,
            f"{r:.0f}×", ha="center", va="bottom",
            color="#34A853", fontsize=14, fontweight="bold")
estilo(ax, "Factor de mejora JAX / TF")
ax.set_xticks(x)
ax.set_xticklabels(PERFILES_SHORT, fontsize=10)
ax.set_ylabel("Veces más rápido con JAX")
ax.axhline(1, color=GRID_COL, linewidth=1.2, linestyle="--")
ax.text(2.4, 1.4, "baseline\n(sin mejora)", color=TEXT_C, fontsize=8)

# Anotación central
fig.text(0.5, -0.02,
         "JAX es ~29× más rápido en el Analyzer — la fase que más se repite durante el análisis",
         ha="center", color="#34A853", fontsize=11, fontweight="bold")

fig.tight_layout()
guardar(fig, "grafica-analyzer-spotlight.webp")


# ═══════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Sección 8 / Conclusión: Escalabilidad total TF vs JAX
# Líneas tiempo + stacked bars composición
# ═══════════════════════════════════════════════════════════════════════════

print("Generando grafica-escalabilidad-total.webp ...")

COLORES_FASES = ["#9B59B6", "#3498DB", "#2ECC71", "#E74C3C", "#F39C12"]
FASES_LEGEND  = ["InputData", "Instanciación", "Sample prior", "Analyzer", "BudgetOpt."]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG_FIG)
fig.suptitle("Escalabilidad total del pipeline — TF vs JAX",
             color=TITLE_C, fontsize=14, fontweight="bold", y=1.01)

# Líneas tiempo total
ax = axes[0]
ax.plot(PERFILES_SHORT, TOTALES_T["TF"],  "o-",  color=TF_COL,  linewidth=2.5, markersize=9, label="TF")
ax.plot(PERFILES_SHORT, TOTALES_T["JAX"], "s--", color=JAX_COL, linewidth=2.5, markersize=9, label="JAX")
for i, (vt, vj) in enumerate(zip(TOTALES_T["TF"], TOTALES_T["JAX"])):
    ax.annotate(f"{vt:.1f}s", (PERFILES_SHORT[i], vt),
                textcoords="offset points", xytext=(6, 4), color=TF_COL, fontsize=9, fontweight="bold")
    ax.annotate(f"{vj:.1f}s", (PERFILES_SHORT[i], vj),
                textcoords="offset points", xytext=(6, -14), color=JAX_COL, fontsize=9, fontweight="bold")
estilo(ax, "Tiempo total (s)")
ax.set_ylabel("Tiempo (s)")
ax.set_ylim(0, 16)
ax.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C)

# Stacked TF
ax = axes[1]
bottom = np.zeros(3)
x_pos = np.arange(3)
for f_idx in range(5):
    vals = [TIEMPOS["TF"][p][f_idx] for p in range(3)]
    bars = ax.bar(x_pos, vals, bottom=bottom, color=COLORES_FASES[f_idx],
                  alpha=0.92, label=FASES_LEGEND[f_idx])
    for b, v in zip(bars, vals):
        if v > 0.4:
            ax.text(b.get_x() + b.get_width()/2, b.get_y() + v/2,
                    f"{v:.1f}", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    bottom += vals
estilo(ax, "Composición — TF")
ax.set_xticks(x_pos); ax.set_xticklabels(PERFILES_SHORT)
ax.set_ylabel("Tiempo (s)")
ax.legend(fontsize=8, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C,
          loc="upper left", framealpha=0.8)

# Stacked JAX
ax = axes[2]
bottom = np.zeros(3)
for f_idx in range(5):
    vals = [TIEMPOS["JAX"][p][f_idx] for p in range(3)]
    bars = ax.bar(x_pos, vals, bottom=bottom, color=COLORES_FASES[f_idx],
                  alpha=0.92, label=FASES_LEGEND[f_idx])
    for b, v in zip(bars, vals):
        if v > 0.4:
            ax.text(b.get_x() + b.get_width()/2, b.get_y() + v/2,
                    f"{v:.1f}", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    bottom += vals
estilo(ax, "Composición — JAX")
ax.set_xticks(x_pos); ax.set_xticklabels(PERFILES_SHORT)
ax.set_ylabel("Tiempo (s)")
ax.legend(fontsize=8, facecolor=BG_AX, edgecolor=GRID_COL, labelcolor=TEXT_C,
          loc="upper left", framealpha=0.8)

fig.tight_layout()
guardar(fig, "grafica-escalabilidad-total.webp")

print("\nTodas las gráficas generadas correctamente.")
