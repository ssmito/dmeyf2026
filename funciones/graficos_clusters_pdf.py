"""Generacion de PDFs para interpretar clusterizaciones cliente-mes."""

from __future__ import annotations

__version__ = "2.0"

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


ID_COL = "numero_de_cliente"
MES_COL = "foto_mes"
CLASE_COL = "ternaria"
CLAVES = [ID_COL, MES_COL]

BANDAS = {
    "ic95": ("media", "IC 95% de la media"),
    "desvio": ("media", "media +/- 1 desvio estandar"),
    "iqr": ("mediana", "rango intercuartil Q25-Q75"),
}

INK = "#14181a"
INK_2 = "#4b534e"
GRID = "#d7dbd3"


def _validar_columnas(df: pd.DataFrame, columnas: list[str], nombre: str) -> None:
    faltantes = [columna for columna in columnas if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {nombre}: {faltantes}")


def _validar_claves_unicas(df: pd.DataFrame, nombre: str) -> None:
    duplicados = df.duplicated(CLAVES, keep=False)
    if duplicados.any():
        cantidad = int(duplicados.sum())
        raise ValueError(
            f"{nombre} tiene {cantidad} filas con claves cliente-mes duplicadas."
        )


def _nombre_cluster(cluster: object) -> str:
    texto = str(cluster)
    if texto.lower().startswith("cluster"):
        return texto
    return f"cluster_{texto}"


def _ordenar_clusters(valores: pd.Series) -> list[object]:
    clusters = valores.dropna().unique().tolist()
    try:
        return sorted(clusters)
    except TypeError:
        return sorted(clusters, key=str)


def _clasificar_ternaria(ternaria: pd.Series) -> pd.Series:
    texto = ternaria.astype("string").str.strip().str.upper()
    clase = pd.Series(pd.NA, index=ternaria.index, dtype="string")
    clase.loc[texto.str.contains("BAJA", na=False)] = "BAJA"
    clase.loc[texto.eq("CONTINUA").fillna(False)] = "CONTINUA"
    return clase


def _colores_clusters(clusters: list[object]) -> dict[object, object]:
    if len(clusters) <= 10:
        cmap = plt.get_cmap("tab10")
    elif len(clusters) <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("hsv")
    return {cluster: cmap(i % cmap.N) for i, cluster in enumerate(clusters)}


def _aplicar_estilo(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _calcular_composicion(
    datos: pd.DataFrame,
    columna_cluster: str,
    clusters: list[object],
) -> pd.DataFrame:
    conteos = pd.crosstab(datos[columna_cluster], datos["_clase_plot"])
    conteos = conteos.reindex(
        index=clusters,
        columns=["BAJA", "CONTINUA"],
        fill_value=0,
    )
    return conteos.div(conteos.sum(axis=1), axis=0).mul(100).fillna(0)


def _etiqueta_leyenda(cluster: object, composicion: pd.DataFrame) -> str:
    baja = composicion.loc[cluster, "BAJA"]
    continua = composicion.loc[cluster, "CONTINUA"]
    return (
        f"{_nombre_cluster(cluster)} - "
        f"BAJA: {baja:.1f}% | CONTINUA: {continua:.1f}%"
    )


def _cargar_descripciones(
    data_dict: str | Path | pd.DataFrame | None,
) -> dict[str, str]:
    if data_dict is None:
        return {}

    if isinstance(data_dict, pd.DataFrame):
        diccionario = data_dict.copy()
    else:
        diccionario = pd.read_csv(Path(data_dict).expanduser())

    columnas_necesarias = ["campo", "Significado"]
    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in diccionario.columns
    ]
    if faltantes:
        raise ValueError(
            f"Faltan columnas en data_dict: {faltantes}. "
            "Se esperan 'campo' y 'Significado'."
        )

    descripciones = (
        diccionario[columnas_necesarias]
        .dropna(subset=["campo"])
        .drop_duplicates(subset=["campo"], keep="last")
        .set_index("campo")["Significado"]
        .dropna()
        .astype(str)
    )
    return descripciones.to_dict()


def _formato_consolidacion(operacion: str, columnas: list[str]) -> str:
    operacion = operacion.lower()
    if operacion == "sum":
        expresion = " + ".join(columnas)
    elif operacion == "avg":
        expresion = f"promedio({', '.join(columnas)})"
    elif operacion == "max":
        expresion = f"maximo({', '.join(columnas)})"
    elif operacion == "min":
        expresion = f"minimo({', '.join(columnas)})"
    else:
        raise ValueError(f"Operacion de consolidacion no valida: {operacion}")
    return f"Consolidación {operacion.upper()}: {expresion}"


def _texto_descripcion_variable(
    variable: str,
    descripciones: dict[str, str],
    diccionario_consolidaciones: dict[str, dict] | None,
) -> str:
    consolidaciones = diccionario_consolidaciones or {}

    if variable not in consolidaciones:
        descripcion = descripciones.get(variable)
        if not descripcion:
            return ""
        return textwrap.fill(
            f"Descripción: {descripcion}",
            width=145,
            subsequent_indent="  ",
        )

    configuracion = consolidaciones[variable]
    columnas = configuracion.get("columnas")
    operacion = configuracion.get("operacion")
    if not isinstance(columnas, list) or not columnas:
        raise ValueError(
            f"La consolidacion de {variable!r} debe incluir una lista "
            "no vacia en 'columnas'."
        )
    if not isinstance(operacion, str):
        raise ValueError(
            f"La consolidacion de {variable!r} debe incluir 'operacion'."
        )

    lineas = [
        textwrap.fill(
            _formato_consolidacion(operacion, columnas),
            width=145,
            subsequent_indent="  ",
        )
    ]
    for columna in columnas:
        descripcion = descripciones.get(columna)
        if descripcion:
            lineas.append(
                textwrap.fill(
                    f"{columna}: {descripcion}",
                    width=145,
                    subsequent_indent="  ",
                )
            )

    return "\n".join(lineas)


def _crear_pagina_resumen(
    pdf: PdfPages,
    datos: pd.DataFrame,
    columna_cluster: str,
    variables: list[str],
    clusters: list[object],
    composicion: pd.DataFrame,
    banda: str,
) -> None:
    tamanios = datos[columna_cluster].value_counts().reindex(clusters)
    meses = sorted(datos[MES_COL].dropna().unique().tolist())

    lineas_clusters = []
    for cluster in clusters:
        lineas_clusters.append(
            f"  {_etiqueta_leyenda(cluster, composicion)} | "
            f"registros: {int(tamanios.loc[cluster]):,}"
        )

    texto = [
        f"Columna de clustering: {columna_cluster}",
        f"Registros incluidos: {len(datos):,}",
        f"Clientes unicos: {datos[ID_COL].nunique():,}",
        f"Meses: {len(meses)} ({meses[0]}-{meses[-1]})",
        f"Variables graficadas: {len(variables)}",
        f"Línea: {BANDAS[banda][0]} mensual por cluster",
        f"Banda: {BANDAS[banda][1]}",
        "",
        "Composicion y tamano de los clusters:",
        *lineas_clusters,
        "",
        "Los porcentajes BAJA | CONTINUA se calculan dentro de cada cluster.",
        "La tabla de cada pagina muestra registros por cluster y foto_mes.",
    ]

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(
        0.06,
        0.92,
        "Tendencias mensuales por cluster",
        fontsize=18,
        color=INK,
        weight="bold",
        va="top",
    )
    fig.text(
        0.06,
        0.84,
        "\n".join(texto),
        fontsize=10.5,
        color=INK_2,
        va="top",
        linespacing=1.45,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _estadisticas_variable(
    datos: pd.DataFrame,
    columna_cluster: str,
    variable: str,
    banda: str,
) -> pd.DataFrame:
    agrupado = datos.groupby([columna_cluster, MES_COL], observed=True)[variable]
    n = agrupado.count()

    if banda == "iqr":
        centro = agrupado.median()
        inferior = agrupado.quantile(0.25)
        superior = agrupado.quantile(0.75)
    else:
        centro = agrupado.mean()
        desvio = agrupado.std()
        ancho = desvio if banda == "desvio" else 1.96 * desvio / np.sqrt(n)
        inferior = centro - ancho
        superior = centro + ancho

    minimo = datos[variable].min(skipna=True)
    maximo = datos[variable].max(skipna=True)
    if pd.notna(minimo):
        inferior = inferior.clip(lower=minimo)
    if pd.notna(maximo):
        superior = superior.clip(upper=maximo)

    return pd.concat(
        {
            "centro": centro,
            "inferior": inferior,
            "superior": superior,
            "n_validos": n,
        },
        axis=1,
    )


def _crear_pagina_variable(
    pdf: PdfPages,
    datos: pd.DataFrame,
    columna_cluster: str,
    variable: str,
    banda: str,
    clusters: list[object],
    meses: list[object],
    colores: dict[object, object],
    composicion: pd.DataFrame,
    tabla_n: pd.DataFrame,
    descripciones: dict[str, str],
    diccionario_consolidaciones: dict[str, dict] | None,
) -> None:
    estadisticas = _estadisticas_variable(
        datos,
        columna_cluster,
        variable,
        banda,
    )

    texto_descripcion = _texto_descripcion_variable(
        variable,
        descripciones,
        diccionario_consolidaciones,
    )
    lineas_descripcion = (
        texto_descripcion.count("\n") + 1 if texto_descripcion else 0
    )
    techo_grafico = max(0.55, 0.82 - 0.022 * lineas_descripcion)

    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.07, 0.31, 0.90, techo_grafico - 0.31])
    x = np.arange(len(meses))

    for cluster in clusters:
        indice = pd.MultiIndex.from_product(
            [[cluster], meses],
            names=[columna_cluster, MES_COL],
        )
        valores = estadisticas.reindex(indice)
        centro = valores["centro"].to_numpy(dtype=float)
        inferior = valores["inferior"].to_numpy(dtype=float)
        superior = valores["superior"].to_numpy(dtype=float)

        ax.fill_between(
            x,
            inferior,
            superior,
            color=colores[cluster],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            x,
            centro,
            color=colores[cluster],
            linewidth=2.0,
            marker="o",
            markersize=5,
        )

    ax.set_xticks(x, [str(mes) for mes in meses])
    ax.set_xlim(-0.3, len(meses) - 0.7)
    ax.set_ylabel(variable, color=INK_2)
    _aplicar_estilo(ax)

    handles = [
        Line2D(
            [0],
            [0],
            color=colores[cluster],
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=_etiqueta_leyenda(cluster, composicion),
        )
        for cluster in clusters
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")

    fig.text(0.07, 0.945, variable, fontsize=20, color=INK, weight="bold")
    fig.text(
        0.07,
        0.912,
        f"Línea = {BANDAS[banda][0]} mensual; banda = {BANDAS[banda][1]}. "
        "Los porcentajes de la leyenda corresponden a la composicion total "
        "de cada cluster.",
        fontsize=9.5,
        color=INK_2,
        va="top",
    )
    if texto_descripcion:
        fig.text(
            0.07,
            0.875,
            texto_descripcion,
            fontsize=8.5,
            color=INK_2,
            va="top",
            linespacing=1.25,
        )

    ax_tabla = fig.add_axes([0.07, 0.05, 0.90, 0.17])
    ax_tabla.axis("off")
    tabla = ax_tabla.table(
        cellText=tabla_n.values,
        rowLabels=[_nombre_cluster(cluster) for cluster in clusters],
        colLabels=[str(mes) for mes in meses],
        loc="center",
        cellLoc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.5)
    tabla.scale(1, 1.15)
    for (fila, _columna), celda in tabla.get_celld().items():
        celda.set_edgecolor(GRID)
        celda.get_text().set_color(INK_2 if fila == 0 else INK)
    ax_tabla.set_title(
        "Cantidad de registros por cluster y foto_mes",
        loc="left",
        fontsize=9,
        color=INK_2,
    )

    pdf.savefig(fig)
    plt.close(fig)


def generar_pdf_clusters(
    df_variables: pd.DataFrame,
    df_baja: pd.DataFrame,
    columna_cluster: str,
    archivo_salida: str | Path,
    banda: str = "ic95",
    data_dict: str | Path | pd.DataFrame | None = None,
    diccionario_consolidaciones: dict[str, dict] | None = None,
) -> Path:
    """Genera un PDF de tendencias mensuales para una clusterizacion.

    Parameters
    ----------
    df_variables:
        DataFrame con numero_de_cliente, foto_mes, ternaria y las variables a
        graficar. Se grafican todas sus columnas numericas excepto las claves.
    df_baja:
        DataFrame subsampleado usado en el clustering. Debe contener las claves
        y la columna indicada en ``columna_cluster``.
    columna_cluster:
        Nombre de la columna de df_baja con las asignaciones del clustering que
        se quiere interpretar.
    archivo_salida:
        Ruta absoluta completa del archivo PDF que se generara.
    banda:
        ``ic95``, ``desvio`` o ``iqr``.
    data_dict:
        Ruta al CSV del diccionario de datos, o DataFrame ya cargado. Debe
        contener las columnas ``campo`` y ``Significado``. Es opcional.
    diccionario_consolidaciones:
        Diccionario opcional usado para crear las variables consolidadas. Para
        cada variable debe contener ``columnas`` y ``operacion``.

    Returns
    -------
    pathlib.Path
        Ruta absoluta del PDF generado.
    """
    if banda not in BANDAS:
        raise ValueError(f"banda debe ser una de {list(BANDAS)}")

    salida = Path(archivo_salida).expanduser()
    if not salida.is_absolute():
        raise ValueError("archivo_salida debe ser una ruta absoluta")
    if salida.suffix.lower() != ".pdf":
        raise ValueError("archivo_salida debe terminar en .pdf")

    _validar_columnas(
        df_variables,
        [*CLAVES, CLASE_COL],
        "df_variables",
    )
    _validar_columnas(
        df_baja,
        [*CLAVES, columna_cluster],
        "df_baja",
    )
    _validar_claves_unicas(df_variables, "df_variables")
    _validar_claves_unicas(df_baja, "df_baja")

    if columna_cluster in df_variables.columns:
        raise ValueError(
            f"df_variables ya contiene la columna de cluster {columna_cluster!r}"
        )

    asignaciones = df_baja[[*CLAVES, columna_cluster]].copy()
    datos = df_variables.merge(
        asignaciones,
        on=CLAVES,
        how="inner",
        validate="one_to_one",
    )
    if datos.empty:
        raise ValueError("La union por cliente-mes no produjo ningun registro")
    if datos[columna_cluster].isna().any():
        raise ValueError("La columna de cluster contiene valores faltantes")

    datos["_clase_plot"] = _clasificar_ternaria(datos[CLASE_COL])
    desconocidas = datos["_clase_plot"].isna()
    if desconocidas.any():
        valores = datos.loc[desconocidas, CLASE_COL].value_counts(dropna=False)
        raise ValueError(
            "Hay valores de ternaria que no son BAJA ni CONTINUA: "
            f"{valores.to_dict()}"
        )

    variables = [
        columna
        for columna in df_variables.select_dtypes(include="number").columns
        if columna not in CLAVES
    ]
    if not variables:
        raise ValueError("df_variables no contiene variables numericas para graficar")

    clusters = _ordenar_clusters(datos[columna_cluster])
    meses = sorted(datos[MES_COL].dropna().unique().tolist())
    if not clusters:
        raise ValueError("No hay clusters para graficar")
    if not meses:
        raise ValueError("No hay meses para graficar")

    composicion = _calcular_composicion(datos, columna_cluster, clusters)
    descripciones = _cargar_descripciones(data_dict)
    colores = _colores_clusters(clusters)
    tabla_n = (
        datos.groupby([columna_cluster, MES_COL], observed=True)
        .size()
        .unstack(MES_COL)
        .reindex(index=clusters, columns=meses)
        .fillna(0)
        .astype(int)
    )

    salida.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(salida) as pdf:
        _crear_pagina_resumen(
            pdf,
            datos,
            columna_cluster,
            variables,
            clusters,
            composicion,
            banda,
        )
        for variable in variables:
            _crear_pagina_variable(
                pdf,
                datos,
                columna_cluster,
                variable,
                banda,
                clusters,
                meses,
                colores,
                composicion,
                tabla_n,
                descripciones,
                diccionario_consolidaciones,
            )

    return salida


__all__ = ["generar_pdf_clusters"]
