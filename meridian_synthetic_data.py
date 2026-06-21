"""
Script de profiling para Google Meridian MMM.

MODOS DE EJECUCIÓN:
  python meridian_synthetic_data.py <perfil>              → pipeline completo (prior)
  python meridian_synthetic_data.py <perfil> --mcmc       → MCMC de producción

PERFILES: pequeno | mediano | grande

BACKEND: variable de entorno MERIDIAN_BACKEND=tf (defecto) | jax

Ejemplos:
  python meridian_synthetic_data.py mediano
  python meridian_synthetic_data.py mediano --mcmc
  MERIDIAN_BACKEND=jax python meridian_synthetic_data.py mediano
  MERIDIAN_BACKEND=jax python meridian_synthetic_data.py mediano --mcmc
"""

import cProfile
import io
import os
import pstats
import sys
import time
import tracemalloc
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/javilazaro/Documents/meridian")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

BACKEND = os.environ.get("MERIDIAN_BACKEND", "tf").lower()
MODO_MCMC = "--mcmc" in sys.argv

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# PERFILES DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
PERFILES = {
    "pequeno": dict(n_geos=2,  n_media_ch=3,  n_rf_ch=2, n_controls=2, n_time=52,  n_media_time=55),
    "mediano": dict(n_geos=10, n_media_ch=8,  n_rf_ch=4, n_controls=4, n_time=104, n_media_time=107),
    "grande":  dict(n_geos=50, n_media_ch=15, n_rf_ch=6, n_controls=6, n_time=208, n_media_time=211),
}

args_sin_flags = [a for a in sys.argv[1:] if not a.startswith("--")]
PERFIL = args_sin_flags[0] if args_sin_flags else "mediano"
if PERFIL not in PERFILES:
    print(f"ERROR: Perfil '{PERFIL}' no válido. Opciones: {list(PERFILES.keys())}")
    sys.exit(1)
CONFIG = PERFILES[PERFIL]

# Parámetros MCMC de producción (típicos en literatura Meridian/MMM):
#   n_chains=4  — estándar para diagnóstico R-hat
#   n_adapt=500 — adaptación suficiente para convergencia del step size
#   n_burnin=500 — descarte de la fase transitoria
#   n_keep=500  — muestras finales para inferencia
# Total steps por cadena: 1500. Total steps: 6000.
MCMC_PRODUCCION = dict(n_chains=4, n_adapt=500, n_burnin=500, n_keep=500)

def estimar_mb(cfg):
    n = (cfg["n_geos"] * cfg["n_time"]
         + cfg["n_geos"] * cfg["n_time"] * cfg["n_controls"]
         + cfg["n_geos"] * cfg["n_media_time"] * cfg["n_media_ch"]
         + cfg["n_geos"] * cfg["n_time"] * cfg["n_media_ch"]
         + cfg["n_geos"] * cfg["n_media_time"] * cfg["n_rf_ch"] * 2
         + cfg["n_geos"] * cfg["n_time"] * cfg["n_rf_ch"]
         + cfg["n_geos"])
    return n * 4 / 1024 / 1024

modo_str = "MCMC PRODUCCIÓN" if MODO_MCMC else "PIPELINE COMPLETO"
print(f"\n{'='*65}")
print(f"  MERIDIAN PROFILING — {modo_str}")
print(f"  Perfil: {PERFIL.upper()} | Backend: {BACKEND.upper()}")
print(f"  Geos: {CONFIG['n_geos']} | Media ch: {CONFIG['n_media_ch']} | "
      f"RF ch: {CONFIG['n_rf_ch']} | Tiempo: {CONFIG['n_time']} semanas")
if MODO_MCMC:
    p = MCMC_PRODUCCION
    total = (p['n_adapt'] + p['n_burnin'] + p['n_keep']) * p['n_chains']
    print(f"  MCMC: {p['n_chains']} cadenas × "
          f"({p['n_adapt']} adapt + {p['n_burnin']} burnin + {p['n_keep']} keep)"
          f" = {total} steps totales")
print(f"  Datos estimados: ~{estimar_mb(CONFIG):.1f} MB (arrays float32)")
print(f"{'='*65}\n")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES DEL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def construir_input_data(cfg):
    from meridian.data.test_utils import sample_input_data_revenue
    return sample_input_data_revenue(
        n_geos=cfg["n_geos"], n_times=cfg["n_time"],
        n_media_times=cfg["n_media_time"], n_controls=cfg["n_controls"],
        n_media_channels=cfg["n_media_ch"], n_rf_channels=cfg["n_rf_ch"],
        seed=42,
    )

def instanciar_modelo(input_data):
    from meridian.model.model import Meridian
    from meridian.model.spec import ModelSpec
    return Meridian(input_data=input_data, model_spec=ModelSpec())

def ejecutar_sample_prior(mmm):
    mmm.sample_prior(n_draws=10, seed=42)

def ejecutar_analyzer(mmm):
    from meridian.analysis.analyzer import Analyzer
    analyzer = Analyzer(meridian=mmm)
    _ = analyzer.incremental_outcome(use_posterior=False, by_reach=False)
    return analyzer

def ejecutar_budget_optimizer(mmm, analyzer):
    from meridian.analysis.optimizer import BudgetOptimizer
    optimizer = BudgetOptimizer(meridian=mmm, analyzer=analyzer)
    _ = optimizer.optimize(
        use_posterior=False,
        fixed_budget=True,
        use_optimal_frequency=False,
    )
    return optimizer

def ejecutar_mcmc_produccion(mmm):
    """MCMC con parámetros de producción reales."""
    p = MCMC_PRODUCCION
    mmm.sample_posterior(
        n_chains=p["n_chains"],
        n_adapt=p["n_adapt"],
        n_burnin=p["n_burnin"],
        n_keep=p["n_keep"],
        seed=42,
    )

def ejecutar_mcmc_minimo(mmm):
    """MCMC mínimo — solo para el pipeline completo con prior."""
    mmm.sample_posterior(n_chains=1, n_adapt=5, n_burnin=0, n_keep=2, seed=42)


# ──────────────────────────────────────────────────────────────────────────────
# HARNESS DE MEDICIÓN
# ──────────────────────────────────────────────────────────────────────────────

def medir_fase(nombre, fn, *args, **kwargs):
    print(f"\n▶  Midiendo: {nombre}")
    print(f"   {'─'*55}")

    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    profiler = cProfile.Profile()
    t0 = time.perf_counter()

    profiler.enable()
    resultado = fn(*args, **kwargs)
    profiler.disable()

    t1 = time.perf_counter()
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    wall = t1 - t0
    stats_mem = snap1.compare_to(snap0, "lineno")
    mem_mb = sum(s.size_diff for s in stats_mem) / 1024 / 1024

    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream)
    ps.sort_stats("cumulative")
    ps.print_stats(15)

    top_mem = sorted(stats_mem, key=lambda s: s.size_diff, reverse=True)[:5]

    print(f"   ⏱  Tiempo:        {wall:.3f}s")
    print(f"   🧠 Memoria delta: {mem_mb:.2f} MB")

    return resultado, {
        "nombre": nombre, "wall_time_s": wall, "mem_delta_mb": mem_mb,
        "profile_text": stream.getvalue(), "top_mem_lines": top_mem,
    }


# ──────────────────────────────────────────────────────────────────────────────
# IMPRESIÓN DE RESULTADOS
# ──────────────────────────────────────────────────────────────────────────────

def imprimir_diagnostico(cfg, idata):
    from meridian import constants as c
    print(f"\n{'='*65}")
    print(f"  DIMENSIONES DEL DATASET")
    print(f"{'='*65}")
    print(f"  kpi shape     = {idata.kpi.shape}")
    if idata.media is not None:
        print(f"  media shape   = {idata.media.shape}")
    if idata.reach is not None:
        print(f"  reach shape   = {idata.reach.shape}")
    print(f"  controls      = {idata.controls.shape}")
    print(f"  Datos (float32): ~{estimar_mb(cfg):.2f} MB")
    print(f"{'='*65}\n")

def imprimir_resumen(metricas, titulo=""):
    print(f"\n{'='*65}")
    if titulo:
        print(f"  {titulo}")
    print(f"  RESUMEN — {PERFIL.upper()} | {BACKEND.upper()}")
    print(f"{'='*65}")
    print(f"  {'Fase':<40} {'Tiempo (s)':>9} {'Mem Δ (MB)':>11}")
    print(f"  {'─'*60}")
    for m in metricas:
        print(f"  {m['nombre']:<40} {m['wall_time_s']:>9.3f} {m['mem_delta_mb']:>11.2f}")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<40} {sum(m['wall_time_s'] for m in metricas):>9.3f} "
          f"{sum(m['mem_delta_mb'] for m in metricas):>11.2f}")
    print(f"{'='*65}\n")

def imprimir_hotspots(metricas):
    print(f"\n{'='*65}")
    print(f"  HOTSPOTS DE CPU")
    print(f"{'='*65}")
    for m in metricas:
        print(f"\n  ▸ {m['nombre']}")
        print(f"    {'─'*55}")
        lines = [l for l in m["profile_text"].split("\n")
                 if l.strip() and "function calls" not in l
                 and "Ordered by" not in l and "ncalls" not in l]
        for l in lines[:10]:
            print(f"    {l}")

def imprimir_memoria(metricas):
    print(f"\n{'='*65}")
    print(f"  HOTSPOTS DE MEMORIA")
    print(f"{'='*65}")
    for m in metricas:
        relevantes = [s for s in m["top_mem_lines"]
                      if abs(s.size_diff) / 1024 / 1024 > 0.05]
        if not relevantes:
            continue
        print(f"\n  ▸ {m['nombre']}")
        for s in relevantes:
            print(f"    {s.size_diff/1024/1024:>+8.2f} MB  {s.traceback[0]}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Warmup — carga de módulos fuera de las mediciones
    print("  [warmup] Cargando módulos...")
    t0 = time.perf_counter()
    _ = construir_input_data(dict(n_geos=1, n_media_ch=1, n_rf_ch=1,
                                  n_controls=1, n_time=4, n_media_time=5))
    print(f"  [warmup] Listo en {time.perf_counter()-t0:.1f}s\n")

    metricas = []

    # ── FASE 1: InputData ──────────────────────────────────────────────────
    idata, m = medir_fase("1. InputData (xarray)", construir_input_data, CONFIG)
    metricas.append(m)
    imprimir_diagnostico(CONFIG, idata)

    # ── FASE 2: Instanciación ─────────────────────────────────────────────
    mmm, m = medir_fase("2. Meridian (instanciacion)", instanciar_modelo, idata)
    metricas.append(m)

    if MODO_MCMC:
        # ── MODO MCMC DE PRODUCCIÓN ────────────────────────────────────────
        _, m = medir_fase("3. MCMC produccion (4c×1500 steps)", ejecutar_mcmc_produccion, mmm)
        metricas.append(m)

        # Analyzer y BudgetOptimizer sobre el posterior real
        analyzer, m = medir_fase("4. Analyzer (incremental_outcome)", ejecutar_analyzer.__wrapped__ if hasattr(ejecutar_analyzer, '__wrapped__') else _analyzer_posterior, mmm)
        metricas.append(m)

    else:
        # ── MODO PIPELINE COMPLETO (prior) ─────────────────────────────────
        _, m = medir_fase("3. Sample prior (10 draws)", ejecutar_sample_prior, mmm)
        metricas.append(m)

        analyzer, m = medir_fase("4. Analyzer (incremental_outcome)", ejecutar_analyzer, mmm)
        metricas.append(m)

        _, m = medir_fase("5. BudgetOptimizer (optimize)", ejecutar_budget_optimizer, mmm, analyzer)
        metricas.append(m)

    imprimir_resumen(metricas)
    imprimir_hotspots(metricas)
    imprimir_memoria(metricas)


def _analyzer_posterior(mmm):
    """Analyzer sobre el posterior (solo en modo --mcmc)."""
    from meridian.analysis.analyzer import Analyzer
    analyzer = Analyzer(meridian=mmm)
    _ = analyzer.incremental_outcome(use_posterior=True, by_reach=False)
    return analyzer


if __name__ == "__main__":
    main()
