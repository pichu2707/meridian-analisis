"""
Genera gráficas de tiempos y memoria del profiling de Meridian.
Produce un PNG con múltiples subgráficas para el análisis comparativo.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DATOS extraídos de los logs
# ─────────────────────────────────────────────────────────────────────────────

FASES = [
    "InputData\n(xarray)",
    "Meridian\n(instanc.)",
    "Sample prior\n(10 draws)",
    "Analyzer\n(incr. outcome)",
    "BudgetOpt.\n(optimize)",
]

PERFILES = ["Pequeño", "Mediano", "Grande"]

# Tiempo (s) por fase — shape: [perfil][fase]
TIEMPOS = {
    "TF": [
        [0.186, 0.655, 0.689, 7.280, 1.748],   # pequeño
        [0.311, 0.667, 0.668, 7.229, 1.937],   # mediano
        [0.573, 0.666, 0.680, 7.216, 3.705],   # grande
    ],
    "JAX": [
        [0.187, 2.145, 4.068, 0.249, 1.580],   # pequeño
        [0.317, 2.264, 4.501, 0.275, 1.758],   # mediano
        [0.579, 2.372, 4.656, 0.281, 3.391],   # grande
    ],
}

# Memoria delta (MB) por fase — shape: [perfil][fase]
MEMORIAS = {
    "TF": [
        [0.35, 3.12, 0.81, 2.93, 1.02],
        [0.66, 3.14, 0.85, 2.94, 1.03],
        [4.90, 3.95, 0.98, 3.78, 1.02],
    ],
    "JAX": [
        [0.35, 6.86, 6.29, 0.76, 1.89],
        [0.66, 6.89, 6.76, 0.76, 1.87],
        [4.90, 6.89, 6.91, 0.77, 1.87],
    ],
}

TOTALES_TIEMPO = {
    "TF":  [10.558, 10.812, 12.841],
    "JAX": [ 8.229,  9.114, 11.280],
}

TOTALES_MEM = {
    "TF":  [ 8.23,  8.62, 14.63],
    "JAX": [16.16, 16.95, 21.33],
}

COLORES = {
    "TF":  "#4285F4",   # azul Google
    "JAX": "#EA4335",   # rojo Google
}

COLORES_PERFIL = ["#5C85D6", "#3A6BC9", "#1A4FAD"]  # azules para pequeño/mediano/grande

# ─────────────────────────────────────────────────────────────────────────────
# FIGURA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#0F0F1A")

TITLE_COLOR  = "#E8E8F0"
TEXT_COLOR   = "#B0B0C8"
GRID_COLOR   = "#2A2A3E"
BG_AX        = "#16162A"

def estilo_ax(ax, titulo=""):
    ax.set_facecolor(BG_AX)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    if titulo:
        ax.set_title(titulo, color=TITLE_COLOR, fontsize=11, fontweight="bold", pad=10)


fig.suptitle(
    "Meridian Profiling — Análisis comparativo TF vs JAX",
    color=TITLE_COLOR, fontsize=16, fontweight="bold", y=0.99
)

gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35,
                       top=0.96, bottom=0.04, left=0.07, right=0.97)

# ─────────────────────────────────────────────────────────────────────────────
# FILA 1 — Tiempo por fase (barras agrupadas TF vs JAX) para cada perfil
# ─────────────────────────────────────────────────────────────────────────────

x = np.arange(len(FASES))
ancho = 0.35

for i, perfil in enumerate(PERFILES):
    ax = fig.add_subplot(gs[0, i])
    bars_tf  = ax.bar(x - ancho/2, TIEMPOS["TF"][i],  ancho, label="TF",  color=COLORES["TF"],  alpha=0.9)
    bars_jax = ax.bar(x + ancho/2, TIEMPOS["JAX"][i], ancho, label="JAX", color=COLORES["JAX"], alpha=0.9)

    # etiquetas encima de cada barra
    for bar in list(bars_tf) + list(bars_jax):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=7.5)

    estilo_ax(ax, f"Tiempo por fase — {perfil}")
    ax.set_xticks(x)
    ax.set_xticklabels(FASES, fontsize=7.5)
    ax.set_ylabel("Tiempo (s)")
    ax.legend(fontsize=8, facecolor=BG_AX, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc="upper left")

# ─────────────────────────────────────────────────────────────────────────────
# FILA 2 — Memoria por fase (barras agrupadas TF vs JAX) para cada perfil
# ─────────────────────────────────────────────────────────────────────────────

for i, perfil in enumerate(PERFILES):
    ax = fig.add_subplot(gs[1, i])
    bars_tf  = ax.bar(x - ancho/2, MEMORIAS["TF"][i],  ancho, label="TF",  color=COLORES["TF"],  alpha=0.9)
    bars_jax = ax.bar(x + ancho/2, MEMORIAS["JAX"][i], ancho, label="JAX", color=COLORES["JAX"], alpha=0.9)

    for bar in list(bars_tf) + list(bars_jax):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.1f}", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=7.5)

    estilo_ax(ax, f"Memoria por fase — {perfil}")
    ax.set_xticks(x)
    ax.set_xticklabels(FASES, fontsize=7.5)
    ax.set_ylabel("Memoria delta (MB)")
    ax.legend(fontsize=8, facecolor=BG_AX, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc="upper left")

# ─────────────────────────────────────────────────────────────────────────────
# FILA 3 — Tiempo TOTAL por perfil (líneas TF vs JAX) + tabla resumen
# ─────────────────────────────────────────────────────────────────────────────

# Gráfica de líneas — total tiempo
ax_lt = fig.add_subplot(gs[2, :2])
ax_lt.plot(PERFILES, TOTALES_TIEMPO["TF"],  "o-", color=COLORES["TF"],  linewidth=2.2,
           markersize=8, label="TF", zorder=3)
ax_lt.plot(PERFILES, TOTALES_TIEMPO["JAX"], "s--", color=COLORES["JAX"], linewidth=2.2,
           markersize=8, label="JAX", zorder=3)

for i, (v_tf, v_jax) in enumerate(zip(TOTALES_TIEMPO["TF"], TOTALES_TIEMPO["JAX"])):
    ax_lt.annotate(f"{v_tf:.2f}s", (PERFILES[i], v_tf),
                   textcoords="offset points", xytext=(6, 4),
                   color=COLORES["TF"], fontsize=9, fontweight="bold")
    ax_lt.annotate(f"{v_jax:.2f}s", (PERFILES[i], v_jax),
                   textcoords="offset points", xytext=(6, -14),
                   color=COLORES["JAX"], fontsize=9, fontweight="bold")

estilo_ax(ax_lt, "Tiempo total del pipeline — escalabilidad")
ax_lt.set_ylabel("Tiempo total (s)")
ax_lt.set_ylim(0, max(max(TOTALES_TIEMPO["TF"]), max(TOTALES_TIEMPO["JAX"])) * 1.3)
ax_lt.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

# Tabla resumen numérico
ax_tab = fig.add_subplot(gs[2, 2])
ax_tab.axis("off")
ax_tab.set_facecolor(BG_AX)

col_labels = ["Perfil", "TF total (s)", "JAX total (s)", "Ganador"]
filas = []
for i, perfil in enumerate(PERFILES):
    t_tf  = TOTALES_TIEMPO["TF"][i]
    t_jax = TOTALES_TIEMPO["JAX"][i]
    ganador = "JAX" if t_jax < t_tf else "TF"
    diff = abs(t_tf - t_jax)
    filas.append([perfil, f"{t_tf:.2f}", f"{t_jax:.2f}", f"{ganador} (-{diff:.2f}s)"])

tabla = ax_tab.table(
    cellText=filas,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1.1, 1.8)

# Estilos de la tabla
for (row, col), cell in tabla.get_celld().items():
    cell.set_facecolor("#1E1E32" if row == 0 else BG_AX)
    cell.set_edgecolor(GRID_COLOR)
    cell.set_text_props(color=TITLE_COLOR if row == 0 else TEXT_COLOR)

ax_tab.set_title("Resumen totales", color=TITLE_COLOR, fontsize=11, fontweight="bold", pad=10)

# ─────────────────────────────────────────────────────────────────────────────
# FILA 4 — Stacked bars para desglosar la composición del tiempo total
# ─────────────────────────────────────────────────────────────────────────────

COLORES_FASES = ["#9B59B6", "#3498DB", "#2ECC71", "#E74C3C", "#F39C12"]

for backend_idx, backend in enumerate(["TF", "JAX"]):
    ax = fig.add_subplot(gs[3, backend_idx])

    bottom = np.zeros(len(PERFILES))
    x_pos  = np.arange(len(PERFILES))

    for f_idx, fase in enumerate(FASES):
        valores = [TIEMPOS[backend][p][f_idx] for p in range(len(PERFILES))]
        bars = ax.bar(x_pos, valores, bottom=bottom,
                      color=COLORES_FASES[f_idx], alpha=0.92,
                      label=fase.replace("\n", " "))
        # etiqueta solo si la sección es suficientemente alta
        for b, v in zip(bars, valores):
            if v > 0.3:
                ax.text(b.get_x() + b.get_width()/2,
                        b.get_y() + v/2,
                        f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=7.5, fontweight="bold")
        bottom += valores

    estilo_ax(ax, f"Composición tiempo total — {backend}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PERFILES)
    ax.set_ylabel("Tiempo (s)")
    ax.legend(fontsize=7, facecolor=BG_AX, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc="upper left",
              ncol=1, framealpha=0.8)

# Gráfica de memoria total (líneas) en la última celda
ax_lm = fig.add_subplot(gs[3, 2])
ax_lm.plot(PERFILES, TOTALES_MEM["TF"],  "o-",  color=COLORES["TF"],  linewidth=2.2,
           markersize=8, label="TF")
ax_lm.plot(PERFILES, TOTALES_MEM["JAX"], "s--", color=COLORES["JAX"], linewidth=2.2,
           markersize=8, label="JAX")

for i, (v_tf, v_jax) in enumerate(zip(TOTALES_MEM["TF"], TOTALES_MEM["JAX"])):
    ax_lm.annotate(f"{v_tf:.1f}MB", (PERFILES[i], v_tf),
                   textcoords="offset points", xytext=(6, 4),
                   color=COLORES["TF"], fontsize=9, fontweight="bold")
    ax_lm.annotate(f"{v_jax:.1f}MB", (PERFILES[i], v_jax),
                   textcoords="offset points", xytext=(6, -14),
                   color=COLORES["JAX"], fontsize=9, fontweight="bold")

estilo_ax(ax_lm, "Memoria total acumulada")
ax_lm.set_ylabel("Memoria delta total (MB)")
ax_lm.set_ylim(0, max(max(TOTALES_MEM["TF"]), max(TOTALES_MEM["JAX"])) * 1.3)
ax_lm.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT = "graficas_profiling.png"
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Gráfica guardada en: {OUTPUT}")
