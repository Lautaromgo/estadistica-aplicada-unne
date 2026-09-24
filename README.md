# Estadística Aplicada — Unidad III

**Diplomatura en Data Analytics · UNNE**

Notebooks de la Unidad III (estadística descriptiva e introducción a la inferencia) y las
prácticas para resolver. Todo trabaja sobre el dataset **California Housing**
(`data/raw/housing.csv`).

## Qué hay en el repo

```
estadistica-aplicada-unne/
├── data/raw/housing.csv                        el dataset (20.640 filas)
├── notebooks/
│   ├── 01_fundamentos_housing.ipynb            Punto 1 · ¿Puedo confiar en estos datos?
│   ├── 02_univariada_housing.ipynb             Punto 2 · Estadística univariada
│   ├── 03_bivariada_housing.ipynb              Punto 3 · Estadística bivariada
│   ├── 04_visualizacion_housing.ipynb          Punto 4 · Visualización
│   ├── 05_inferencia_housing.ipynb             Punto 5 · Introducción a la inferencia
│   └── practica/
│       ├── practica_1_univariada.ipynb                 después de los puntos 1 y 2
│       └── practica_2_bivariada_visualizacion.ipynb    después de los puntos 3 y 4
├── src/estadistica.py                          funciones que usan las notebooks 02 a 05
└── requirements.txt                            las librerías que hacen falta
```

- Las notebooks **01 a 05** acompañan los videos de cada punto. La idea es pausar el
  video en cada bloque y ejecutar las celdas correspondientes.
- Las **prácticas** traen funciones vacías con un `TODO` que explica el paso a paso.
  Abajo de cada una hay una celda que verifica tu resultado y muestra ✅ o ❌. Mientras
  no completes una función, las celdas que la usan muestran el error
  `NotImplementedError: Falta completar ...`. Es esperable: no rompiste nada.

## Cómo ponerlo a andar

Necesitás **Git** y **Python 3.10, 3.11 o 3.12**. Todavía no se puede usar 3.13: algunas
de las versiones de `requirements.txt` no tienen soporte para 3.13.

Para ver qué versión tenés instalada:

```bash
python --version
```

En macOS y Linux puede que el comando se llame `python3`. En ese caso, usá `python3` en
lugar de `python` en todos los pasos que siguen.

### 1. Clonar el repositorio

```bash
git clone https://github.com/Lautaromgo/estadistica-aplicada-unne.git
cd estadistica-aplicada-unne
```

### 2. Crear el ambiente virtual

Con el ambiente virtual, las librerías del curso se instalan en una carpeta `.venv/`
dentro del proyecto y no se mezclan con el resto de tu computadora. Se crea una sola vez:

```bash
python -m venv .venv
```

### 3. Activarlo

Hay que activarlo **cada vez que abras una terminal nueva** para trabajar en el curso.

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD)**

```bat
.venv\Scripts\activate.bat
```

Si quedó activado, vas a ver `(.venv)` al principio de la línea de la terminal.

> En PowerShell, si aparece un error que dice que *la ejecución de scripts está
> deshabilitada*, ejecutá primero
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y después volvé a activar.

### 4. Instalar las librerías

Esto también se hace una sola vez, con el ambiente ya activado:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Abrir las notebooks

```bash
jupyter notebook
```

Se te va a abrir el navegador. Entrá a la carpeta `notebooks/` y abrí la notebook que
quieras. Para ejecutar una celda, usá **Shift + Enter**.

Cuando termines, cerrá Jupyter con **Ctrl + C** en la terminal y desactivá el ambiente
con:

```bash
deactivate
```

### Para las próximas veces

Los pasos 1, 2 y 4 no se repiten. Cada vez que vuelvas al curso alcanza con:

```bash
cd estadistica-aplicada-unne
source .venv/bin/activate
jupyter notebook
```

En Windows, reemplazá la línea del `source` por la del paso 3.

Si subimos material nuevo, lo bajás con:

```bash
git pull
```

> **Ojo:** si modificaste una notebook que después actualizamos, `git pull` puede dar un
> conflicto. Para evitarlo, antes de resolver una práctica **hacé una copia**, por
> ejemplo `practica_1_univariada_mia.ipynb`, y trabajá sobre esa.

## Alternativa: Google Colab (sin instalar nada)

Las **prácticas** también se pueden abrir en Google Colab, desde el botón *Open in Colab*
que tienen al principio. No necesitan que clones el repo, porque bajan el dataset solas.
Antes de empezar, guardá una copia en tu Drive (**Archivo → Guardar una copia en Drive**).
Si no, se pierden los cambios.

## Problemas frecuentes

| Qué pasa | Qué hacer |
|---|---|
| `python: command not found` | Probá con `python3`. En Windows, reinstalá Python y marcá *Add Python to PATH*. |
| `ModuleNotFoundError: No module named 'pandas'` (u otra librería) | El ambiente no está activado, o Jupyter no se abrió desde el ambiente. Activalo y volvé a ejecutar `jupyter notebook`. |
| `pip install` falla compilando `numpy` | Estás usando Python 3.13 o superior. Instalá 3.12 y creá el ambiente de nuevo. |
