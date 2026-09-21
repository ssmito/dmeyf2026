"""Generacion de PDFs con distribuciones por cluster para registros cliente-mes."""

from __future__ import annotations

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


def _etiqueta_leyenda(
    cluster: object,
    composicion: pd.DataFrame,
    ceros: dict[object, tuple[int, float]] | None = None,
) -> str:
    baja = composicion.loc[cluster, "BAJA"]
    continua = composicion.loc[cluster, "CONTINUA"]
    etiqueta = (
        f"{_nombre_cluster(cluster)} - "
        f"BAJA: {baja:.1f}% | CONTINUA: {continua:.1f}%"
    )
    if ceros is not None:
        cantidad, porcentaje = ceros[cluster]
        etiqueta += f" | ceros: {cantidad:,} ({porcentaje:.1f}%)"
    return etiqueta


def _cargar_diccionario_datos(
    data_dict: str | Path | pd.DataFrame | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if data_dict is None:
        return {}, {}

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
    unidades = {}
    if "unidad" in diccionario.columns:
        unidades = (
            diccionario[["campo", "unidad"]]
            .dropna(subset=["campo", "unidad"])
            .drop_duplicates(subset=["campo"], keep="last")
            .set_index("campo")["unidad"]
            .astype(str)
            .to_dict()
        )

    return descripciones.to_dict(), unidades


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


def _nombre_parece_monto(variable: str) -> bool:
    return (
        variable.startswith("m")
        or variable.startswith("Visa_m")
        or variable.startswith("Master_m")
    )


def _es_variable_monto(
    variable: str,
    unidades: dict[str, str],
    diccionario_consolidaciones: dict[str, dict] | None,
) -> bool:
    unidad = unidades.get(variable, "").strip().lower()
    if "peso" in unidad:
        return True

    consolidaciones = diccionario_consolidaciones or {}
    if variable in consolidaciones:
        columnas = consolidaciones[variable].get("columnas", [])
        if columnas:
            return all(
                "peso" in unidades.get(columna, "").strip().lower()
                or _nombre_parece_monto(columna)
                for columna in columnas
            )

    return _nombre_parece_monto(variable)


def _valores_validos(
    serie: pd.Series,
    escala_log_monto: bool = False,
) -> np.ndarray:
    valores = pd.to_numeric(serie, errors="coerce").to_numpy(dtype=float)
    valores = valores[np.isfinite(valores)]
    if escala_log_monto:
        valores = np.sign(valores) * np.log1p(np.abs(valores))
    return valores


def _suavizar_conteos(conteos: np.ndarray, sigma: float) -> np.ndarray:
    radio = max(1, int(np.ceil(3 * sigma)))
    posiciones = np.arange(-radio, radio + 1, dtype=float)
    kernel = np.exp(-0.5 * (posiciones / sigma) ** 2)
    kernel /= kernel.sum()
    extendidos = np.pad(conteos.astype(float), radio, mode="edge")
    suavizados = np.convolve(extendidos, kernel, mode="same")
    return suavizados[radio:-radio]


def _calcular_bordes(valores: np.ndarray, bins: int) -> np.ndarray:
    unicos = np.unique(valores)

    if len(unicos) == 1:
        valor = unicos[0]
        amplitud = max(abs(valor) * 0.05, 0.5)
        return np.array([valor - amplitud, valor + amplitud], dtype=float)

    if len(unicos) <= 20:
        puntos_medios = (unicos[:-1] + unicos[1:]) / 2
        borde_inicial = unicos[0] - (unicos[1] - unicos[0]) / 2
        borde_final = unicos[-1] + (unicos[-1] - unicos[-2]) / 2
        return np.concatenate(([borde_inicial], puntos_medios, [borde_final]))

    return np.histogram_bin_edges(valores, bins=bins)


def _calcular_histogramas(
    datos: pd.DataFrame,
    columna_cluster: str,
    variable: str,
    clusters: list[object],
    bordes: np.ndarray,
    escala_log_monto: bool,
) -> tuple[
    dict[object, np.ndarray],
    dict[object, np.ndarray],
    dict[object, tuple[int, float]],
    int,
    int,
]:
    histogramas = {}
    histogramas_sin_ceros = {}
    ceros = {}
    maximo_completo = 0
    maximo_sin_ceros = 0

    for cluster in clusters:
        valores = _valores_validos(
            datos.loc[datos[columna_cluster].eq(cluster), variable],
            escala_log_monto,
        )
        conteos, _ = np.histogram(valores, bins=bordes)
        histogramas[cluster] = conteos

        cantidad_ceros = int(np.count_nonzero(valores == 0))
        porcentaje_ceros = (
            100 * cantidad_ceros / len(valores) if len(valores) else 0.0
        )
        ceros[cluster] = (cantidad_ceros, porcentaje_ceros)

        valores_sin_ceros = valores[valores != 0]
        conteos_sin_ceros, _ = np.histogram(valores_sin_ceros, bins=bordes)
        histogramas_sin_ceros[cluster] = conteos_sin_ceros

        if conteos.size:
            maximo_completo = max(maximo_completo, int(conteos.max()))
        if conteos_sin_ceros.size:
            maximo_sin_ceros = max(
                maximo_sin_ceros,
                int(conteos_sin_ceros.max()),
            )

    return (
        histogramas,
        histogramas_sin_ceros,
        ceros,
        maximo_completo,
        maximo_sin_ceros,
    )


def _crear_pagina_resumen(
    pdf: PdfPages,
    datos: pd.DataFrame,
    columna_cluster: str,
    variables: list[str],
    clusters: list[object],
    composicion: pd.DataFrame,
    variables_monto: set[str],
    bins: int,
    recortar_pico_cero: bool,
    factor_recorte: float,
    suavizado_sigma: float,
) -> None:
    tamanios = datos[columna_cluster].value_counts().reindex(clusters)
    meses = sorted(datos[MES_COL].dropna().unique().tolist())

    lineas_clusters = [
        f"  {_etiqueta_leyenda(cluster, composicion)} | "
        f"registros: {int(tamanios.loc[cluster]):,}"
        for cluster in clusters
    ]

    recorte = (
        f"Activo: limite Y = {factor_recorte:.0%} del maximo por bin "
        "calculado sin valores iguales a cero, solo cuando el pico de ceros domina."
        if recortar_pico_cero
        else "Desactivado."
    )

    texto = [
        f"Columna de clustering: {columna_cluster}",
        f"Registros incluidos: {len(datos):,}",
        f"Clientes unicos: {datos[ID_COL].nunique():,}",
        f"Meses: {len(meses)} ({meses[0]}-{meses[-1]})",
        f"Variables graficadas: {len(variables)}",
        f"Variables monetarias en log con signo: {len(variables_monto)}",
        f"Bins solicitados para variables continuas: {bins}",
        "Eje Y: cantidad de registros; no se normalizan las distribuciones.",
        f"Suavizado: kernel gaussiano sobre los bins (sigma={suavizado_sigma:g}).",
        f"Recorte del pico de ceros: {recorte}",
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
        "Distribuciones por cluster",
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
        linespacing=1.4,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _crear_pagina_variable(
    pdf: PdfPages,
    datos: pd.DataFrame,
    columna_cluster: str,
    variable: str,
    bins: int,
    recortar_pico_cero: bool,
    factor_recorte: float,
    suavizado_sigma: float,
    alpha_barras: float,
    escala_log_monto: bool,
    clusters: list[object],
    meses: list[object],
    colores: dict[object, object],
    composicion: pd.DataFrame,
    tabla_n: pd.DataFrame,
    descripciones: dict[str, str],
    diccionario_consolidaciones: dict[str, dict] | None,
) -> None:
    valores_globales = _valores_validos(datos[variable], escala_log_monto)
    texto_descripcion = _texto_descripcion_variable(
        variable,
        descripciones,
        diccionario_consolidaciones,
    )
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.07, 0.31, 0.90, 0.49])
    recorte_aplicado = False
    limite_y = None
    hay_ceros = False

    if valores_globales.size:
        bordes = _calcular_bordes(valores_globales, bins)
        (
            histogramas,
            histogramas_sin_ceros,
            ceros,
            maximo_completo,
            maximo_sin_ceros,
        ) = (
            _calcular_histogramas(
                datos,
                columna_cluster,
                variable,
                clusters,
                bordes,
                escala_log_monto,
            )
        )
        hay_ceros = any(cantidad > 0 for cantidad, _ in ceros.values())
        centros = (bordes[:-1] + bordes[1:]) / 2
        anchos = np.diff(bordes)

        if (
            recortar_pico_cero
            and hay_ceros
            and maximo_sin_ceros > 0
            and maximo_completo > factor_recorte * maximo_sin_ceros
        ):
            limite_y = factor_recorte * maximo_sin_ceros
            ax.set_ylim(0, limite_y)
            recorte_aplicado = True

        for cluster in clusters:
            ax.bar(
                centros,
                histogramas[cluster],
                width=anchos,
                align="center",
                color=colores[cluster],
                alpha=alpha_barras,
                edgecolor="none",
            )
            conteos_para_linea = (
                histogramas_sin_ceros[cluster]
                if recorte_aplicado
                else histogramas[cluster]
            )
            suavizados = _suavizar_conteos(
                conteos_para_linea,
                suavizado_sigma,
            )
            ax.plot(
                centros,
                suavizados,
                color=colores[cluster],
                linewidth=2.7,
                alpha=1.0,
                marker="o" if len(centros) == 1 else None,
            )

        handles = [
            Line2D(
                [0],
                [0],
                color=colores[cluster],
                linewidth=2.2,
                label=_etiqueta_leyenda(
                    cluster,
                    composicion,
                    ceros if hay_ceros else None,
                ),
            )
            for cluster in clusters
        ]
        ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right")
    else:
        ax.text(
            0.5,
            0.5,
            "La variable no tiene valores numericos finitos.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=INK_2,
        )

    etiqueta_x = (
        f"{variable} - log con signo"
        if escala_log_monto
        else variable
    )
    ax.set_xlabel(etiqueta_x, color=INK_2)
    ax.set_ylabel("Cantidad de registros", color=INK_2)
    _aplicar_estilo(ax)

    if escala_log_monto:
        subtitulo = (
            "Histograma combinado en escala logaritmica con signo: "
            "sign(x) * log1p(abs(x)); conserva ceros y valores negativos. "
        )
    else:
        subtitulo = "Histograma combinado en escala original. "
    subtitulo += (
        "Bins compartidos entre clusters; barras transparentes y linea suavizada. "
        "El eje Y muestra cantidades, no densidades."
    )
    if recorte_aplicado:
        subtitulo += (
            f" Eje Y recortado en {limite_y:,.0f}; "
            "la linea se suaviza sin los ceros y la leyenda informa "
            "sus cantidades reales."
        )

    subtitulo = textwrap.fill(subtitulo, width=135)
    lineas_subtitulo = subtitulo.count("\n") + 1
    lineas_descripcion = (
        texto_descripcion.count("\n") + 1 if texto_descripcion else 0
    )
    y_descripcion = 0.862 - 0.026 * (lineas_subtitulo - 1)
    techo_grafico = 0.82 - 0.026 * (lineas_subtitulo - 1)
    techo_grafico -= 0.022 * lineas_descripcion
    techo_grafico = max(0.55, techo_grafico)

    posicion_original = ax.get_position()
    ax.set_position(
        [
            posicion_original.x0,
            posicion_original.y0,
            posicion_original.width,
            techo_grafico - posicion_original.y0,
        ]
    )

    fig.text(0.07, 0.945, variable, fontsize=20, color=INK, weight="bold")
    fig.text(
        0.07,
        0.912,
        subtitulo,
        fontsize=9.5,
        color=INK_2,
        va="top",
    )
    if texto_descripcion:
        fig.text(
            0.07,
            y_descripcion,
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


def generar_pdf_distribuciones_clusters(
    df_variables: pd.DataFrame,
    df_baja: pd.DataFrame,
    columna_cluster: str,
    archivo_salida: str | Path,
    bins: int = 50,
    recortar_pico_cero: bool = True,
    factor_recorte: float = 1.10,
    suavizado_sigma: float = 1.2,
    alpha_barras: float = 0.12,
    data_dict: str | Path | pd.DataFrame | None = None,
    diccionario_consolidaciones: dict[str, dict] | None = None,
) -> Path:
    """Genera un PDF con una distribucion por variable y todos los clusters.

    Cada registro cliente-mes se considera una observacion independiente. Los
    histogramas usan bins compartidos y cantidades sin normalizar en el eje Y.
    Las variables monetarias se transforman con logaritmo con signo para
    conservar los valores negativos y los ceros.

    Parameters
    ----------
    df_variables:
        DataFrame con numero_de_cliente, foto_mes, ternaria y las variables a
        graficar. Se grafican todas sus columnas numericas excepto las claves.
    df_baja:
        DataFrame subsampleado usado en el clustering. Debe contener las claves
        y la columna indicada en ``columna_cluster``.
    columna_cluster:
        Nombre de la columna de df_baja con las asignaciones del clustering.
    archivo_salida:
        Ruta absoluta completa del archivo PDF que se generara.
    bins:
        Cantidad de bins para variables continuas. Las variables con hasta 20
        valores distintos usan un intervalo por valor observado.
    recortar_pico_cero:
        Si es True, recorta el eje Y cuando el pico producido por los ceros
        supera el limite calculado sin ceros.
    factor_recorte:
        Multiplicador aplicado al maximo por bin calculado sin ceros. El valor
        1.10 deja un margen del 10%.
    suavizado_sigma:
        Intensidad del suavizado gaussiano aplicado a los conteos por bin.
    alpha_barras:
        Transparencia de las barras del histograma, entre 0 y 1.
    data_dict:
        Ruta al CSV del diccionario de datos, o DataFrame ya cargado. Debe
        contener las columnas ``campo`` y ``Significado``. Si incluye
        ``unidad``, los campos en pesos se identifican como montos.
    diccionario_consolidaciones:
        Diccionario opcional usado para crear las variables consolidadas. Para
        cada variable debe contener ``columnas`` y ``operacion``.

    Returns
    -------
    pathlib.Path
        Ruta absoluta del PDF generado.
    """
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins debe ser un entero mayor que cero")
    if factor_recorte <= 1:
        raise ValueError("factor_recorte debe ser mayor que 1")
    if suavizado_sigma <= 0:
        raise ValueError("suavizado_sigma debe ser mayor que 0")
    if not 0 < alpha_barras <= 1:
        raise ValueError("alpha_barras debe estar entre 0 y 1")

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
    descripciones, unidades = _cargar_diccionario_datos(data_dict)
    variables_monto = {
        variable
        for variable in variables
        if _es_variable_monto(
            variable,
            unidades,
            diccionario_consolidaciones,
        )
    }
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
            variables_monto,
            bins,
            recortar_pico_cero,
            factor_recorte,
            suavizado_sigma,
        )
        for variable in variables:
            _crear_pagina_variable(
                pdf,
                datos,
                columna_cluster,
                variable,
                bins,
                recortar_pico_cero,
                factor_recorte,
                suavizado_sigma,
                alpha_barras,
                variable in variables_monto,
                clusters,
                meses,
                colores,
                composicion,
                tabla_n,
                descripciones,
                diccionario_consolidaciones,
            )

    return salida


__all__ = ["generar_pdf_distribuciones_clusters"]
