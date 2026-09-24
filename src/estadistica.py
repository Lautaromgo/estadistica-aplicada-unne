"""
estadistica.py — El "utils" de la Unidad III
=============================================

Las funciones que usan las notebooks de la unidad, en un solo lugar.

Dónde vive y por qué
--------------------
El README del template del curso lo dice desde la Unidad II:

    "Si repetís una función en varias notebooks, movela a src/ para reutilizarla."

La Unidad III es la primera vez que `src/` deja de estar vacío. Esto no es una
carpeta más: es el equivalente al módulo de utilidades que cualquier equipo de
datos mantiene en una empresa. Una función se escribe una vez, se discute una
vez, y después la usa todo el mundo.

REGLA DE ORO
------------
Toda función recibe el DataFrame y los NOMBRES DE COLUMNA como parámetros.
Nada hardcodeado. Así el mismo código corre sobre el dataset de cualquiera:
el tuyo, el de tu compañero, o uno nuevo.

    describir_segmentos(df, "provincia", "valor_musd")     # ✓
    describir_segmentos(df)                                # ✗ (adivina las columnas)

Capas
-----
- Punto 1 : perfilar, clasificar_variable        -> ¿puedo confiar en estos datos?
- Punto 2 : varianza, desvio_estandar, coef_variacion,
            resumen_numerico, describir_segmentos -> ¿cómo es cada segmento por dentro?
- Punto 3 : covarianza, matriz_correlacion, tabla_cruzada,
            detectar_outliers, comparar_segmentos  -> ¿se ven distintos? ¿qué los relaciona?
- Punto 4 : graficar_relacion, dashboard_segmento   -> ¿cómo lo muestro y lo cuento?
- Punto 5 : intervalo_confianza_formula, intervalo_confianza,
            comparar_dos_grupos, graficar_bootstrap  -> ¿la diferencia es real o es azar?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO 1 — Perfilado: "¿Puedo confiar en estos datos?"
# ═══════════════════════════════════════════════════════════════════════════

# Textos que suelen significar "no hay dato" aunque estén escritos como si fueran uno.
_SENTINELAS = {"", " ", "-", "--", "?", "na", "n/a", "nan", "null", "none",
               "sin dato", "s/d", "s/i"}


def clasificar_variable(serie, umbral_categorico=15):
    """Sugiere el tipo de una variable a partir de su dtype y su cardinalidad.

    Devuelve una de:
        'cualitativa (nominal)', 'cuantitativa discreta',
        'cuantitativa continua', 'vacía'

    LA DECISIÓN QUE HAY QUE TOMAR
    -----------------------------
    `umbral_categorico` es el corte entre "esto es una categoría disfrazada de
    número" y "esto es una cantidad". No hay un valor correcto: es una decisión
    de criterio que el equipo toma una vez y que después hereda todo el mundo.

    Con umbral=15 sobre el dataset de exportaciones:
        ranking_destino (11 valores enteros) -> 'cuantitativa discreta'
        anio            (32 valores enteros) -> 'cuantitativa continua'

    Ninguna de las dos es del todo cierta, y por eso el umbral se discute.

    LÍMITE CONOCIDO
    ---------------
    Lo ORDINAL no se puede detectar automáticamente. Que 'bajo < medio < alto'
    tenga un orden es información que está en tu cabeza, no en los datos. La
    función marca todo lo no numérico como nominal y te deja la última palabra.
    """
    s = serie.dropna()

    if s.empty:
        return "vacía"

    # Los booleanos son categorías de dos valores, no números.
    if pd.api.types.is_bool_dtype(s):
        return "cualitativa (nominal)"

    if pd.api.types.is_numeric_dtype(s):
        es_entera = (s % 1 == 0).all()
        # Enteros con pocos valores distintos: probablemente sea un código o un conteo.
        if es_entera and s.nunique() <= umbral_categorico:
            return "cuantitativa discreta"
        return "cuantitativa continua"

    return "cualitativa (nominal)"


def _alerta_columna(serie, umbral_ceros=0.02):
    """Busca problemas de calidad típicos en UNA columna.

    Devuelve un texto con las alertas encontradas (vacío si no hay ninguna).
    Cada alerta corresponde a una forma real en que un dataset puede mentir.
    """
    alertas = []
    s = serie.dropna()

    if s.empty:
        return "columna vacía"

    # --- Problemas propios de columnas de texto ---
    if serie.dtype == object:
        como_texto = s.astype(str).str.strip().str.lower()
        es_centinela = como_texto.isin(_SENTINELAS)

        # 1) Números guardados como texto. Si casi todo lo que NO es centinela
        #    se puede convertir a número, la columna es numérica mal leída.
        sin_centinelas = s[~es_centinela]
        if not sin_centinelas.empty:
            convertible = pd.to_numeric(sin_centinelas, errors="coerce").notna().mean()
            if convertible > 0.9:
                alertas.append("¿número cargado como texto?")

        # 2) Nulos disfrazados de texto ("N/A", "s/d", "-").
        if es_centinela.any():
            alertas.append("posibles nulos disfrazados")

    # --- Problemas de columnas numéricas ---
    if pd.api.types.is_numeric_dtype(s):
        # 3) Demasiados ceros exactos.
        #    Un cero PASA todos los controles de nulos: no es NaN, es un número
        #    perfectamente válido. Pero muchas veces significa "no hubo dato",
        #    no "el valor fue cero". Es la trampa más difícil de ver.
        prop_ceros = (s == 0).mean()
        if prop_ceros > umbral_ceros:
            alertas.append(f"{prop_ceros:.1%} de ceros exactos (¿son ceros o faltantes?)")

        # 4) Un mismo valor repetido muchas veces en una variable continua.
        #    Suele delatar un tope o un valor de relleno.
        if s.nunique() > 1:
            top_frec = s.value_counts(normalize=True).iloc[0]
            valor_top = s.value_counts().index[0]
            if top_frec > 0.05 and valor_top != 0:
                alertas.append(f"el valor {valor_top:g} se repite {top_frec:.1%} (¿tope?)")

    # 5) Una sola categoría: la columna no aporta nada para comparar.
    if s.nunique() == 1:
        alertas.append("columna constante")

    return " · ".join(alertas)


def perfilar(df, umbral_categorico=15):
    """Auditoría del dataset en una sola pasada — la puerta de entrada del sistema.

    Antes de calcular una media hay que poder confiar en los datos. `perfilar`
    resume, columna por columna: tipo de dato, nulos, cardinalidad, tipo de
    variable sugerido y alertas de calidad.

    Devuelve un DataFrame (una fila por columna) e imprime un encabezado con
    filas, columnas y duplicados.

    Parameters
    ----------
    df : pd.DataFrame
    umbral_categorico : int -> se pasa a clasificar_variable (ver su docstring).
    """
    n_filas = len(df)

    if n_filas == 0:
        print("DataFrame vacío")
        return pd.DataFrame()

    n_dup = int(df.duplicated().sum())
    print(f"Filas: {n_filas:,}  |  Columnas: {df.shape[1]}  |  "
          f"Duplicadas: {n_dup} ({n_dup / n_filas:.1%})")

    filas = {}
    for col in df.columns:
        s = df[col]
        nulos = int(s.isna().sum())
        filas[col] = {
            "dtype": str(s.dtype),
            "nulos": nulos,
            "pct_nulos": round(nulos / n_filas * 100, 1),
            "unicos": int(s.nunique()),
            "tipo_sugerido": clasificar_variable(s, umbral_categorico),
            "alerta": _alerta_columna(s),
        }
    return pd.DataFrame(filas).T


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO 2 — Univariada: "¿Cómo es cada segmento por dentro?"
# ═══════════════════════════════════════════════════════════════════════════

def varianza(valores, muestra=True):
    """Varianza calculada a mano, sin usar .var().

    La idea en una frase: es el PROMEDIO DE LAS DISTANCIAS AL CUADRADO respecto
    de la media. Se elevan al cuadrado para que las distancias hacia arriba y
    hacia abajo no se cancelen entre sí.

    Parameters
    ----------
    valores : array-like -> los datos numéricos.
    muestra : bool       -> True  divide por (n-1)  [varianza muestral]
                            False divide por n      [varianza poblacional]

    ¿POR QUÉ (n-1)?
    ---------------
    Cuando trabajamos con una muestra medimos las distancias respecto de la
    media DE LA MUESTRA, que por construcción es el punto más cercano posible a
    esos datos. Eso hace que subestimemos la dispersión real de la población.
    Dividir por (n-1) en lugar de n corrige ese sesgo.

    CUÁNDO IMPORTA: casi nunca con n grande (con n=1408 la diferencia es del
    0,04%) y muchísimo con n chico (con n=5 es del 25%). En la notebook 02 hay
    una celda que lo mide.
    """
    x = np.asarray(valores, dtype=float)
    x = x[~np.isnan(x)]                      # los nulos no participan del cálculo
    n = len(x)

    if n < 2:
        return np.nan                        # con un solo dato no hay dispersión que medir

    media = x.sum() / n
    suma_cuadrados = ((x - media) ** 2).sum()
    divisor = (n - 1) if muestra else n
    return suma_cuadrados / divisor


def desvio_estandar(valores, muestra=True):
    """Desvío estándar = raíz cuadrada de la varianza.

    ¿Por qué no usamos la varianza directamente? Porque está en unidades AL
    CUADRADO. Si medimos exportaciones en millones de dólares, la varianza está
    en "millones de dólares al cuadrado", que no significa nada. La raíz nos
    devuelve a las unidades originales y vuelve el número interpretable.
    """
    return np.sqrt(varianza(valores, muestra=muestra))


def coef_variacion(valores, muestra=True):
    """Coeficiente de variación (CV) = desvío / media.

    Es la dispersión RELATIVA al tamaño de lo que se mide. Como no tiene
    unidades, permite comparar la variabilidad de grupos que están en escalas
    distintas: un desvío de 5 millones es enorme para Formosa y chico para
    Misiones.

    DÓNDE SE ROMPE
    --------------
    El CV divide por la media, así que:
      - si la media es 0, no está definido (devolvemos NaN);
      - si la variable tiene valores negativos, la media puede quedar cerca de
        cero y el CV se dispara sin significar nada;
      - sólo tiene sentido en variables de razón (con un cero absoluto), como
        importes o cantidades. En una variación porcentual NO lo uses.
    """
    x = np.asarray(valores, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return np.nan

    media = x.mean()
    if media == 0:
        return np.nan

    return desvio_estandar(x, muestra=muestra) / media


def resumen_numerico(df, columna):
    """Resumen univariado completo de UNA columna numérica.

    Junta las tres familias de medidas en una sola fila:
      - tendencia central : media, mediana, moda
      - dispersión        : desvío, coeficiente de variación
      - posición          : mínimo, Q1, mediana, Q3, máximo

    Mirar media y mediana juntas es lo más informativo de toda la tabla: si se
    parecen, la distribución es simétrica; si la media es mucho mayor, hay una
    cola larga hacia la derecha tirando de ella.

    Returns
    -------
    pd.Series con las métricas, nombrada como la columna.
    """
    x = df[columna].dropna()

    if x.empty:
        return pd.Series(dtype=float, name=columna)

    moda = x.mode()
    moda = moda.iloc[0] if not moda.empty else np.nan

    resumen = {
        "n": int(x.count()),
        "media": x.mean(),
        "mediana": x.median(),
        "moda": moda,
        "desvio": desvio_estandar(x),        # muestral, divide por (n-1)
        "cv": coef_variacion(x),             # dispersión relativa
        "min": x.min(),
        "q1": np.percentile(x, 25),
        "q3": np.percentile(x, 75),
        "max": x.max(),
    }
    return pd.Series(resumen, name=columna)


def describir_segmentos(df, columna_grupo, columna_valor):
    """Aplica resumen_numerico a CADA segmento y los pone lado a lado.

    Este es el corazón del punto 2: pasamos de describir un grupo a describir
    todos los grupos por separado, que es lo que hace falta para empezar a
    responder la pregunta madre: "¿el segmento A es distinto del B?".

    Ojo: esta tabla MUESTRA diferencias, no las CONFIRMA. Que dos medias sean
    distintas no quiere decir que la diferencia sea real — eso se responde en
    el punto 5, con intervalos de confianza.

    Parameters
    ----------
    df            : pd.DataFrame
    columna_grupo : str -> la categórica que define los segmentos ('provincia').
    columna_valor : str -> la numérica que queremos comparar ('valor_musd').

    Returns
    -------
    pd.DataFrame con una fila por segmento, ordenado por media descendente.
    """
    filas = {}
    for grupo, sub in df.groupby(columna_grupo):
        filas[grupo] = resumen_numerico(sub, columna_valor)

    tabla = pd.DataFrame(filas).T
    tabla.index.name = columna_grupo
    return tabla.sort_values("media", ascending=False)


def graficar_distribucion(df, columna, bins=30):
    """Histograma + boxplot de una columna, uno al lado del otro.

    Los dos gráficos cuentan cosas distintas y se complementan:
      - el histograma muestra la FORMA (simetría, sesgo, picos, huecos);
      - el boxplot resume mediana, cuartiles y marca los atípicos.

    Las líneas de media y mediana están puestas a propósito: cuando se separan,
    se ve de un vistazo por qué la media no alcanza para describir la variable.

    Nota sobre `bins`: no es un detalle cosmético. El mismo dato con 10 o con
    100 bins puede sugerir historias distintas. Probá varios antes de concluir.
    """
    x = df[columna].dropna()

    fig, (ax_hist, ax_box) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]}
    )

    sns.histplot(x, bins=bins, ax=ax_hist)
    ax_hist.axvline(x.mean(), color="crimson", linestyle="--",
                    label=f"media = {x.mean():,.1f}")
    ax_hist.axvline(x.median(), color="seagreen", linestyle="--",
                    label=f"mediana = {x.median():,.1f}")
    ax_hist.set_title(f"Distribución de {columna}  (bins={bins})")
    ax_hist.legend()

    sns.boxplot(y=x, ax=ax_box)
    ax_box.set_title(f"Boxplot de {columna}")

    fig.tight_layout()
    return fig


def graficar_boxplot_por_segmento(df, columna_grupo, columna_valor):
    """Un boxplot por segmento, todos en el mismo eje y ordenados por mediana.

    Es el momento "ajá" de la clase: ver las cajas lado a lado ya INSINÚA si los
    segmentos son distintos. Pero es sólo visual. Recién en el punto 5, con
    intervalos de confianza, vamos a poder afirmar si la diferencia es real o
    es azar.
    """
    orden = (df.groupby(columna_grupo)[columna_valor]
               .median()
               .sort_values(ascending=False)
               .index)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x=columna_grupo, y=columna_valor, order=orden, ax=ax)
    ax.set_title(f"{columna_valor} por {columna_grupo}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO 3 — Bivariada: "¿Se ven distintos? ¿Qué los relaciona?"
# ═══════════════════════════════════════════════════════════════════════════

def covarianza(x, y, muestra=True):
    """Covarianza entre dos series, calculada a mano.

    LA IDEA
    -------
    Para cada fila miramos si x está por encima o por debajo de SU media, y si
    y está por encima o por debajo de la SUYA. Si las dos se desvían para el
    mismo lado, el producto da positivo; si se desvían para lados opuestos, da
    negativo. La covarianza es el promedio de esos productos.

    Ojo con el error clásico: NO se comparan los valores de x con los de y
    (serían peras con manzanas). Se compara cada uno con su propio promedio.

    EL LÍMITE, Y ES GRANDE
    ----------------------
    El número que devuelve no se puede interpretar solo, porque está en las
    unidades de x multiplicadas por las de y. El mismo par de variables da
    150.847 si el precio está en dólares y 151 si está en miles: los datos no
    cambiaron, sólo la unidad. Por eso lo único que se lee de la covarianza es
    EL SIGNO, y para leer la intensidad se usa la correlación, que es esto
    mismo pero dividido por los dos desvíos (ver `matriz_correlacion`).

    Parameters
    ----------
    x, y    : pd.Series o array. Se descartan las filas donde alguno sea nulo.
    muestra : True divide por (n-1), como .cov() de pandas. False divide por n.
    """
    par = pd.DataFrame({"x": pd.Series(x).reset_index(drop=True),
                        "y": pd.Series(y).reset_index(drop=True)}).dropna()
    n = len(par)
    if n < 2:
        return np.nan

    desvios_x = par["x"] - par["x"].mean()
    desvios_y = par["y"] - par["y"].mean()
    return float((desvios_x * desvios_y).sum() / (n - 1 if muestra else n))


def matriz_correlacion(df, metodo="pearson", graficar=True, figsize=(9, 7)):
    """Matriz de correlación de las columnas numéricas, con heatmap opcional.

    CÓMO SE LEE
    -----------
    - La diagonal siempre vale 1: cada columna consigo misma.
    - Es simétrica, así que alcanza con mirar la mitad de arriba.
    - Va de -1 a 1 y NO tiene unidades, así que sí se puede comparar entre
      pares distintos (a diferencia de la covarianza).

    DOS TRAMPAS QUE HAY QUE TENER PRESENTES
    ---------------------------------------
    1. Una correlación altísima puede no ser ningún hallazgo. En el dataset de
       housing, total_bedrooms x households da 0,980 — pero las dos miden lo
       mismo (cuán grande es el barrio). Antes de festejar un número alto,
       preguntate qué mide cada columna.
    2. Pearson mide relación LINEAL. Un valor cercano a cero puede significar
       "no hay relación" o "hay una relación fuerte que no es una recta", y son
       cosas completamente distintas. Por eso `graficar_relacion` existe: nunca
       reportes un r sin haber mirado el scatter.

    Notar también que sólo entran las columnas numéricas: la categórica con la
    que comparás segmentos queda afuera en silencio, igual que pasaba con
    .describe() en el punto 1. Para cruzar categóricas está `tabla_cruzada`.

    Parameters
    ----------
    metodo : "pearson" (lineal) o "spearman" (por rangos, aguanta lo no lineal
             siempre que sea monótono).
    """
    matriz = df.corr(numeric_only=True, method=metodo)

    if graficar:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(matriz, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, center=0, square=True,
                    linewidths=0.5, ax=ax)
        # vmin/vmax/center fijos a propósito: sin ellos seaborn escala al rango
        # de los datos y un 0,3 se pinta del mismo rojo que un 0,95.
        ax.set_title(f"Matriz de correlación ({metodo})")
        fig.tight_layout()

    return matriz


def tabla_cruzada(df, columna_1, columna_2, normalizar="index", decimales=1):
    """Tabla de contingencia entre dos categóricas, con su columna n.

    POR QUÉ NORMALIZAR
    ------------------
    Los conteos crudos casi nunca son la tabla que querés mirar: te dicen qué
    grupo es más GRANDE, no cómo se COMPORTA cada grupo. En housing, INLAND
    tiene el número más alto de barrios baratos (4.928) simplemente porque
    tiene 6.551 filas. Normalizado por fila se ve lo que importa: el 75,2% de
    INLAND es barato, contra el 16,6% de <1H OCEAN.

    POR QUÉ LA COLUMNA n
    --------------------
    Un porcentaje esconde cuántos casos hay detrás. En housing, ISLAND tiene el
    100% de sus barrios en el tramo alto — y son cinco filas. La columna n va
    siempre, y por eso esta función la agrega sola.

    Parameters
    ----------
    normalizar : "index"   -> cada fila suma 100% (el perfil de cada grupo)
                 "columns" -> cada columna suma 100%
                 "all"     -> toda la tabla suma 100%
                 None      -> conteos crudos
                 Elegir mal el eje da una tabla bien calculada que responde
                 otra pregunta.
    """
    conteos = pd.crosstab(df[columna_1], df[columna_2])

    if normalizar is None:
        tabla = conteos
    else:
        tabla = (pd.crosstab(df[columna_1], df[columna_2], normalize=normalizar)
                 * 100).round(decimales)

    tabla["n"] = conteos.sum(axis=1)
    return tabla


def detectar_outliers(df, columna, k=1.5, devolver_limites=False):
    """Marca los valores lejos del centro según la regla de k * IQR.

    LA REGLA
    --------
    Se calculan los cuartiles (punto 2), se mide el rango intercuartílico
    IQR = Q3 - Q1, y se considera atípico todo lo que caiga a más de k veces el
    IQR por fuera de las cajas.

    EL k NO ES UNA LEY
    ------------------
    El 1,5 que trae todo el mundo por defecto es una convención, no un
    resultado. En median_house_value de housing:

        k = 1,5  ->  1.071 atípicos, de los cuales 965 son el tope de 500.001
        k = 3,0  ->      0 atípicos

    O sea: con el default marcás mil casos que en su mayoría son un problema de
    calidad que ya conocías, y con el otro valor razonable no marcás ninguno.
    No hay opción tibia, y por eso el k se decide mirando la variable —no se
    hereda del default.

    Y OJO CON LO QUE ESTA REGLA NO PUEDE VER
    ----------------------------------------
    Mira una columna por vez. Hay filas absurdas cuyos valores son normales en
    cada columna por separado y sólo son imposibles como combinación: el barrio
    de housing con 1.561 habitaciones (percentil 29, normalísimo) y 11 hogares
    da 142 ambientes por hogar. Para eso hay que mirar el par, no el valor.

    Returns
    -------
    Las filas atípicas. Con devolver_limites=True, la tupla
    (filas, lim_inferior, lim_superior).
    """
    valores = df[columna].dropna()
    q1, q3 = np.percentile(valores, [25, 75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - k * iqr, q3 + k * iqr

    atipicos = df[(df[columna] < lim_inf) | (df[columna] > lim_sup)]

    if devolver_limites:
        return atipicos, lim_inf, lim_sup
    return atipicos


def comparar_segmentos(df, columna_grupo, columna_valor, graficar=True):
    """Tabla comparativa de segmentos + boxplots agrupados. Corona el punto 3.

    Es `describir_segmentos` (punto 2) más lo que aprendimos en el punto 3:
    agrega la participación de cada grupo sobre el total y deja el n bien a la
    vista, porque la comparación entre grupos con n muy distintos es
    exactamente donde se cometen los errores.

    Sigue valiendo la advertencia del punto 2, y ahora con más razón: esta
    tabla MUESTRA diferencias, no las CONFIRMA.
    """
    tabla = describir_segmentos(df, columna_grupo, columna_valor)
    tabla["% del total"] = (tabla["n"] / tabla["n"].sum() * 100).round(1)

    if graficar:
        graficar_boxplot_por_segmento(df, columna_grupo, columna_valor)

    return tabla


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO 4 — Visualización aplicada: "¿Cómo lo muestro y lo cuento?"
# ═══════════════════════════════════════════════════════════════════════════

def graficar_relacion(df, columna_x, columna_y, columna_color=None,
                      muestra=4000, alpha=0.35, random_state=42, ax=None):
    """Grafica la relación entre dos columnas, ELIGIENDO SOLA el tipo de gráfico.

    Acá el sistema se usa a sí mismo: para decidir qué dibujar llama a
    `clasificar_variable`, la función que escribimos en vivo en la sincrónica 1.

        num x num  ->  scatter
        cat x num  ->  boxplot
        num x cat  ->  boxplot (dado vuelta)
        cat x cat  ->  barras apiladas (porcentajes por fila)

    Sobre `muestra`: con 20.640 puntos superpuestos el scatter es una mancha
    negra donde no se distingue nada. Bajar alpha y dibujar una muestra al azar
    no es cosmética — cambia lo que se puede concluir del gráfico. El
    random_state va fijo para que el gráfico no cambie en cada corrida.
    """
    tipo_x = clasificar_variable(df[columna_x])
    tipo_y = clasificar_variable(df[columna_y])
    es_cat_x = tipo_x.startswith("cualitativa")
    es_cat_y = tipo_y.startswith("cualitativa")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if not es_cat_x and not es_cat_y:
        datos = df
        if muestra and len(df) > muestra:
            datos = df.sample(muestra, random_state=random_state)
        sns.scatterplot(data=datos, x=columna_x, y=columna_y,
                        hue=columna_color, alpha=alpha, s=14, ax=ax)
        pie = f"scatter · {len(datos):,} de {len(df):,} filas"

    elif es_cat_x and not es_cat_y:
        orden = df.groupby(columna_x)[columna_y].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=columna_x, y=columna_y, order=orden, ax=ax)
        ax.tick_params(axis="x", rotation=30)
        pie = "boxplot por categoría"

    elif not es_cat_x and es_cat_y:
        orden = df.groupby(columna_y)[columna_x].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=columna_x, y=columna_y, order=orden, ax=ax)
        pie = "boxplot por categoría"

    else:
        cruce = pd.crosstab(df[columna_x], df[columna_y], normalize="index") * 100
        cruce.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("% dentro de cada categoría")
        ax.tick_params(axis="x", rotation=30)
        pie = "barras apiladas (100% por fila)"

    ax.set_title(f"{columna_y} vs {columna_x}   —   {pie}")
    return ax


def dashboard_segmento(df, columna_grupo, columna_valor, bins=50,
                       columna_relacion=None, figsize=(14, 9)):
    """Los cuatro paneles del punto 4, en una sola figura.

    No hay nada nuevo acá: los cuatro salen de funciones que ya existen. Es el
    momento en que el sistema deja de ser un cuaderno de ejercicios y empieza a
    producir algo que se le puede mostrar a otra persona.

        [0,0] distribución de la variable   [0,1] boxplot por segmento
        [1,0] heatmap de correlaciones      [1,1] relación, coloreada por grupo
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    x = df[columna_valor].dropna()

    sns.histplot(x, bins=bins, ax=axes[0, 0])
    axes[0, 0].axvline(x.mean(), color="crimson", linestyle="--", label="media")
    axes[0, 0].axvline(x.median(), color="seagreen", linestyle="--", label="mediana")
    axes[0, 0].set_title(f"Distribución de {columna_valor}")
    axes[0, 0].legend()

    orden = (df.groupby(columna_grupo)[columna_valor]
               .median().sort_values(ascending=False).index)
    sns.boxplot(data=df, x=columna_grupo, y=columna_valor, order=orden, ax=axes[0, 1])
    axes[0, 1].set_title(f"{columna_valor} por {columna_grupo}")
    axes[0, 1].tick_params(axis="x", rotation=30)

    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", vmin=-1, vmax=1,
                center=0, square=True, cbar_kws={"shrink": 0.7}, ax=axes[1, 0])
    axes[1, 0].set_title("Correlaciones")

    if columna_relacion is None:
        candidatas = [c for c in df.select_dtypes("number").columns
                      if c != columna_valor]
        columna_relacion = (df[candidatas].corrwith(df[columna_valor])
                            .abs().idxmax()) if candidatas else None

    if columna_relacion is not None:
        graficar_relacion(df, columna_relacion, columna_valor,
                          columna_color=columna_grupo, ax=axes[1, 1])
    else:
        axes[1, 1].axis("off")

    fig.suptitle(f"{columna_valor} según {columna_grupo}", fontsize=14)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO 5 — Inferencia: "¿La diferencia es real o es azar?"
# ═══════════════════════════════════════════════════════════════════════════

def _z(confianza):
    """Valor z para un nivel de confianza. Cubre los tres usuales, sin scipy."""
    return {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(round(confianza, 2), 1.960)


def intervalo_confianza_formula(df, columna, confianza=0.95):
    """Intervalo de confianza para la media, por la FÓRMULA clásica.

        media ± z · (desvío / √n)

    LA IDEA
    -------
    Si repitieras el muestreo muchas veces, las medias que obtendrías se
    reparten alrededor de la media real. El "error estándar" mide cuánto se
    mueven, y el margen de error es ese desvío escalado por z.

    LO QUE HAY QUE MIRAR: EL √n
    ---------------------------
    El margen cae con la RAÍZ de n, no con n. Para achicarlo a la mitad no
    necesitás el doble de datos: necesitás CUATRO veces más. Esa es la razón
    económica por la que las muestras grandes son caras y por la que, pasado
    cierto punto, agrandar la muestra deja de convenir.

    EL SUPUESTO, Y DÓNDE SE ROMPE
    -----------------------------
    Supone que la media muestral se distribuye aproximadamente normal. Con n
    grande eso vale casi siempre (teorema central del límite), y por eso la
    fórmula y el bootstrap coinciden. Con n chico y datos torcidos —una variable
    topeada, un grupo de cinco casos— el supuesto no se cumple y el intervalo
    que devuelve es optimista.

    Por eso el sistema usa bootstrap como método principal: ver
    `intervalo_confianza`.
    """
    x = df[columna].dropna()
    n = len(x)
    media = x.mean()
    error_estandar = x.std(ddof=1) / np.sqrt(n)
    margen = _z(confianza) * error_estandar

    return {"media": media, "error_estandar": error_estandar, "margen": margen,
            "inferior": media - margen, "superior": media + margen, "n": n}


def intervalo_confianza(df, columna, estadistico=np.mean, n_boot=2000,
                        confianza=0.95, random_state=42):
    """Intervalo de confianza por BOOTSTRAP. El método principal del sistema.

    LA IDEA, EN UNA FRASE
    ---------------------
    Tu muestra es tu mejor foto de la población. Si la re-sorteás con reemplazo
    miles de veces y calculás el estadístico en cada re-sorteo, ves cuánto
    podría variar tu resultado por puro azar. El intervalo son los percentiles
    de esas estimaciones.

    POR QUÉ ÉSTE Y NO LA FÓRMULA
    ----------------------------
    - Es más simple de entender: reusa el `for`, el re-muestreo y
      `np.percentile` que ya venimos usando desde el punto 2. No hay que
      creerle a ninguna fórmula.
    - No asume normalidad.
    - Funciona con CUALQUIER estadístico cambiando una palabra: pasale
      `estadistico=np.median` y tenés el intervalo de la mediana. Con la
      fórmula clásica eso no es posible.

    DÓNDE SE ROMPE
    --------------
    Con n muy chico el bootstrap re-sortea siempre los mismos pocos valores. En
    el dataset de housing, `ISLAND` tiene 5 filas: la media re-muestreada sólo
    puede tomar 55 valores distintos, y el histograma sale escalonado en vez de
    una campana. El intervalo se calcula igual, pero está construido sobre muy
    poco. Mirá siempre la distribución con `graficar_bootstrap` antes de
    reportar el número.

    `random_state` va fijo para que el resultado sea reproducible. Sin eso los
    números cambian un poco entre corridas — es esperable, no es un error.
    """
    x = df[columna].dropna().values
    n = len(x)
    rng = np.random.default_rng(random_state)

    estimaciones = np.empty(n_boot)
    for i in range(n_boot):
        muestra = rng.choice(x, size=n, replace=True)   # re-sorteo CON reemplazo
        estimaciones[i] = estadistico(muestra)

    alpha = 1 - confianza
    return {"estimacion": estadistico(x),
            "inferior": np.percentile(estimaciones, 100 * alpha / 2),
            "superior": np.percentile(estimaciones, 100 * (1 - alpha / 2)),
            "distribucion": estimaciones, "n": n, "confianza": confianza}


def comparar_dos_grupos(df, columna_grupo, columna_valor, grupo_a, grupo_b,
                        n_boot=2000, confianza=0.95, random_state=42):
    """La función que corona el sistema: ¿la diferencia entre dos segmentos es real?

    Hace bootstrap de la DIFERENCIA de medias y construye su intervalo.

    CÓMO SE LEE
    -----------
    - El intervalo NO incluye 0  → la diferencia se sostiene: en casi todos los
      re-sorteos el signo se mantiene.
    - El intervalo SÍ incluye 0  → no podemos afirmar que haya diferencia. El
      azar del muestreo alcanza para explicar lo que vemos.

    DOS ADVERTENCIAS QUE VALEN MÁS QUE LA FUNCIÓN
    ---------------------------------------------
    1. "No incluye 0" NO quiere decir "es importante". Con n muy grande, una
       diferencia trivial deja de incluir el 0. Que sea consistente es una
       pregunta; que valga la pena es otra, y la contesta el negocio.
    2. Un n chico NO invalida todo por igual. En housing, `ISLAND` tiene 5 filas
       y su intervalo de la media es 30 veces más ancho que el de un grupo
       grande — pero la diferencia contra los demás igual se sostiene, porque
       las cinco islas son todas caras. En cambio la CORRELACIÓN de ese mismo
       grupo tiene un intervalo de [-1, +1]: ahí no se puede afirmar ni el
       signo. El intervalo te dice qué conclusión sobrevive y cuál no.
    """
    a = df.loc[df[columna_grupo] == grupo_a, columna_valor].dropna().values
    b = df.loc[df[columna_grupo] == grupo_b, columna_valor].dropna().values
    rng = np.random.default_rng(random_state)

    diferencias = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        diferencias[i] = ra.mean() - rb.mean()

    alpha = 1 - confianza
    inferior = np.percentile(diferencias, 100 * alpha / 2)
    superior = np.percentile(diferencias, 100 * (1 - alpha / 2))
    incluye_cero = bool(inferior <= 0 <= superior)

    conclusion = (
        f"El intervalo INCLUYE 0 → no podemos afirmar que {grupo_a} y {grupo_b} "
        f"sean distintos."
        if incluye_cero else
        f"El intervalo NO incluye 0 → la diferencia entre {grupo_a} y {grupo_b} "
        f"se sostiene."
    )

    return {"grupo_a": grupo_a, "grupo_b": grupo_b, "n_a": len(a), "n_b": len(b),
            "dif_observada": a.mean() - b.mean(),
            "inferior": inferior, "superior": superior,
            "incluye_cero": incluye_cero, "distribucion": diferencias,
            "conclusion": conclusion}


def graficar_bootstrap(resultado, titulo="Distribución bootstrap", ax=None):
    """Histograma de las estimaciones re-muestreadas, con el intervalo marcado.

    Acepta el dict de `intervalo_confianza` o el de `comparar_dos_grupos`. Si el
    resultado es una comparación, marca además el cero: ahí se ve de un vistazo
    si el intervalo lo cruza.

    Mirar esta distribución es parte del método, no un adorno: con n chico se ve
    escalonada, y eso avisa que el intervalo está construido sobre pocos datos
    distintos.
    """
    dist = resultado["distribucion"]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.hist(dist, bins=45, color="#DCEAF7", edgecolor="#1E579B", linewidth=0.5)
    for limite, etiqueta in ((resultado["inferior"], "límite inferior"),
                             (resultado["superior"], "límite superior")):
        ax.axvline(limite, color="#E8A33D", ls="--", lw=2, label=etiqueta)
    if resultado.get("incluye_cero") is not None:
        ax.axvline(0, color="#D1495B", lw=2.5, label="cero (sin diferencia)")

    ax.set_title(titulo)
    ax.set_ylabel("re-muestreos")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    return ax
