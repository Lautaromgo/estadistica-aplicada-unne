"""
generar_creditos_sinteticos.py — El dataset de la notebook 06 y de la práctica 3
=================================================================================

Arma `data/raw/creditos_sinteticos.csv`: 5.000 préstamos INVENTADOS, con el
resultado (pagó / no pagó) que habría visto una financiera.

Los datos no salen de ningún banco. Se generan con una regla conocida más ruido,
y eso es a propósito: como sabemos cómo se fabricaron, podemos comprobar qué
aprende un modelo y qué se inventa.

Lo que se buscó al calibrarlo:
- alrededor de 9 de cada 10 préstamos se pagan, como en la vida real;
- el riesgo sube con la cuota sobre el sueldo, los atrasos previos y las
  consultas recientes, y baja con la antigüedad laboral;
- `dia_solicitud` NO influye en nada: es la trampa de la actividad "Sé el
  modelo". Un modelo que la usa está memorizando ruido.

Se corre una sola vez desde la raíz del repo:

    python data/generar_creditos_sinteticos.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

N = 5_000
SEMILLA = 2026
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes"]


def generar(n=N, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)

    edad = rng.integers(20, 66, n)
    # Sueldos en pesos, asimétricos como los de verdad: muchos cerca del medio,
    # pocos muy altos.
    ingreso = np.round(rng.lognormal(np.log(1_000_000), 0.45, n), -3)
    cuota_sobre_ingreso = np.clip(rng.beta(2, 8, n), 0.03, 0.75)
    cuota = np.round(ingreso * cuota_sobre_ingreso, -3)
    antiguedad = np.round(np.clip(rng.exponential(5, n), 0, edad - 18), 1)
    atrasos = (rng.random(n) < 0.15).astype(int)
    consultas = rng.poisson(1.0, n)
    dia = rng.choice(DIAS, n)

    # La regla escondida: cuánto riesgo aporta cada cosa (en escala logit).
    riesgo = (-12
              + 26 * (cuota / ingreso)
              + 6 * atrasos
              + 1.0 * consultas
              - 0.3 * antiguedad
              - 0.02 * (edad - 40))
    prob_no_pago = 1 / (1 + np.exp(-riesgo))
    pago = (rng.random(n) >= prob_no_pago).astype(int)

    return pd.DataFrame({
        "edad": edad,
        "ingreso_mensual": ingreso.astype(int),
        "cuota": cuota.astype(int),
        "antiguedad_laboral": antiguedad,
        "atrasos_12m": atrasos,
        "consultas_recientes": consultas,
        "dia_solicitud": dia,
        "pago": pago,
    })


if __name__ == "__main__":
    destino = Path(__file__).parent / "raw" / "creditos_sinteticos.csv"
    df = generar()
    df.to_csv(destino, index=False)
    print(f"{len(df):,} filas → {destino}")
    print(f"pagaron: {df['pago'].mean():.1%}")
