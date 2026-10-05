import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats


def plot_distribuciones_clusters_tablas(
    df_clusters,
    df_distribuciones,
    col_id,
    col_cluster,
    columnas,
    ajustes=None,
    label_sin_cluster="SIN_CLUSTER",
    porcentaje_tabla=60,
    escala_eje="real",
):
    ajustes = ajustes or {}

    if not 0 < porcentaje_tabla < 100:
        raise ValueError("porcentaje_tabla debe estar entre 0 y 100.")
    if escala_eje not in ("real", "transformada"):
        raise ValueError(
            "escala_eje debe ser 'real' o 'transformada'."
        )

    df_plot = df_distribuciones.merge(
        df_clusters[[col_id, col_cluster]],
        on=col_id,
        how="left",
        validate="one_to_one",
    )
    df_plot[col_cluster] = (
        df_plot[col_cluster].astype(object).fillna(label_sin_cluster)
    )

    porcentaje_continua = (
        df_clusters.groupby(col_cluster)["grupo"]
        .apply(lambda s: 100 * s.eq(0).mean())
    )

    resultados = {}

    for feature in columnas:
        config = ajustes.get(feature, {})
        transformacion = config.get("transformacion")
        xscale = config.get("xscale", "linear")
        yscale = config.get("yscale", "linear")

        if transformacion not in (None, "log"):
            raise ValueError(f"Transformación desconocida: {transformacion}")
        if transformacion == "log" and xscale != "linear":
            raise ValueError(
                f"{feature}: no combines transformacion='log' "
                f"con xscale='{xscale}'."
            )

        grupos = list(pd.unique(df_plot[col_cluster]))
        series = [
            df_plot.loc[df_plot[col_cluster] == grupo, feature].dropna()
            for grupo in grupos
        ]

        tabla = pd.DataFrame({
            col_cluster: grupos,
            "n_valido": [len(s) for s in series],
            "% CONTINUA": [
                porcentaje_continua.get(grupo, np.nan)
                for grupo in grupos
            ],
            "media": [s.mean() for s in series],
            "sd": [s.std(ddof=1) for s in series],
        })

        validos = [
            i for i, s in enumerate(series)
            if len(s) >= 2 and s.nunique() >= 2
        ]

        p_global = np.nan
        p_pares = {}

        if len(validos) >= 2:
            muestras = [
                series[i].to_numpy(dtype=float) for i in validos
            ]
            posthoc = stats.tukey_hsd(
                *muestras,
                equal_var=False,
            )

            for pos_i, i in enumerate(validos):
                for pos_j, j in enumerate(validos):
                    if i < j:
                        p_pares[i, j] = float(
                            posthoc.pvalue[pos_i, pos_j]
                        )

            if len(validos) == len(grupos):
                p_global = float(
                    stats.f_oneway(
                        *muestras,
                        equal_var=False,
                    ).pvalue
                )

        for grupo in grupos:
            tabla[f"vs {grupo}"] = ""

        comparaciones = []

        for i, grupo_fila in enumerate(grupos):
            tabla.at[i, f"vs {grupo_fila}"] = "—"

            for j in range(i + 1, len(grupos)):
                diferencia = float(
                    tabla.at[i, "media"] - tabla.at[j, "media"]
                )
                p_ajustado = p_pares.get((i, j), np.nan)

                comparaciones.append({
                    "cluster_fila": grupo_fila,
                    "cluster_columna": grupos[j],
                    "diferencia_medias": diferencia,
                    "p_ajustado": p_ajustado,
                })

                p_texto = (
                    f"{p_ajustado:.3g}"
                    if np.isfinite(p_ajustado) else "NA"
                )
                tabla.at[i, f"vs {grupos[j]}"] = (
                    f"Δ={diferencia:,.2f}\np={p_texto}"
                )

        resultados[feature] = {
            "tabla": tabla,
            "comparaciones": pd.DataFrame(
                comparaciones,
                columns=[
                    "cluster_fila",
                    "cluster_columna",
                    "diferencia_medias",
                    "p_ajustado",
                ],
            ),
            "p_global": p_global,
        }

        ancho = max(16, 8 + 2 * len(grupos))
        alto = max(5, 3 + 0.5 * len(grupos))

        fig, (ax, ax_tabla) = plt.subplots(
            1, 2,
            figsize=(ancho, alto),
            gridspec_kw={
                "width_ratios": [
                    100 - porcentaje_tabla,
                    porcentaje_tabla,
                ]
            },
            layout="constrained",
        )

        curvas = 0

        for grupo, subset in zip(grupos, series):
            if len(subset) < 2 or subset.nunique() < 2:
                continue

            if transformacion == "log":
                subset = np.sign(subset) * np.log1p(np.abs(subset))

            sns.kdeplot(
                subset,
                ax=ax,
                label=str(grupo),
                fill=True,
                alpha=0.3,
            )
            curvas += 1

        ax.set_title(feature)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if transformacion == "log" and escala_eje == "real":
            def formato_real(valor_transformado, _):
                valor = (
                    np.sign(valor_transformado)
                    * np.expm1(np.abs(valor_transformado))
                )
                return (
                    f"{valor:,.0f}"
                    if abs(valor) >= 1
                    else f"{valor:.2g}"
                )

            ax.xaxis.set_major_formatter(
                FuncFormatter(formato_real)
            )
            ax.set_xlabel(feature)
        else:
            ax.set_xlabel(
                f"sign({feature}) · log1p(abs({feature}))"
                if transformacion == "log" else feature
            )

        if "xlim" in config:
            limites = config["xlim"]

            if transformacion == "log" and escala_eje == "real":
                limites = tuple(
                    None if valor is None
                    else np.sign(valor) * np.log1p(np.abs(valor))
                    for valor in limites
                )

            ax.set_xlim(*limites)

        if "ylim" in config:
            ax.set_ylim(*config["ylim"])

        if curvas:
            ax.legend(title=col_cluster)

        ax_tabla.axis("off")
        p_texto = f"{p_global:.3g}" if np.isfinite(p_global) else "NA"
        ax_tabla.set_title(f"Welch ANOVA: p={p_texto}", pad=12)

        tabla_visible = tabla.copy()
        for col in ("media", "sd"):
            tabla_visible[col] = tabla_visible[col].map(
                lambda v: f"{v:,.2f}" if pd.notna(v) else "NA"
            )
        tabla_visible["% CONTINUA"] = tabla_visible["% CONTINUA"].map(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "NA"
        )

        tabla_grafico = ax_tabla.table(
            cellText=tabla_visible.astype(str).to_numpy(),
            colLabels=[str(c) for c in tabla_visible.columns],
            cellLoc="center",
            colLoc="center",
            bbox=[0, 0.05, 1, 0.85],
        )
        tabla_grafico.auto_set_font_size(False)
        tabla_grafico.set_fontsize(8)

        plt.show()

    return resultados

def plot_distribuciones_clusters(
    df_clusters,
    df_distribuciones,
    col_id,
    col_cluster,
    columnas,
    ajustes=None,
    label_sin_cluster="SIN_CLUSTER",
):
    ajustes = ajustes or {}

    df_plot = df_distribuciones.merge(
        df_clusters[[col_id, col_cluster]],
        on=col_id,
        how="left",
        validate="one_to_one",
    )
    df_plot[col_cluster] = (
        df_plot[col_cluster].astype(object).fillna(label_sin_cluster)
    )

    for feature in columnas:
        config = ajustes.get(feature, {})
        transformacion = config.get("transformacion")
        xscale = config.get("xscale", "linear")
        yscale = config.get("yscale", "linear")

        if transformacion not in (None, "log"):
            raise ValueError(f"Transformación desconocida: {transformacion}")
        if transformacion == "log" and xscale != "linear":
            raise ValueError(
                f"{feature}: no combines transformacion='log' "
                f"con xscale='{xscale}'."
            )

        fig, ax = plt.subplots(figsize=(8, 4))
        curvas = 0

        for cluster in df_plot[col_cluster].unique():
            subset = df_plot.loc[
                df_plot[col_cluster] == cluster, feature
            ].dropna()

            if len(subset) < 2 or subset.nunique() < 2:
                continue

            if transformacion == "log":
                subset = np.sign(subset) * np.log1p(np.abs(subset))

            sns.kdeplot(
                subset,
                ax=ax,
                label=str(cluster),
                fill=True,
                alpha=0.3,
            )
            curvas += 1

        ax.set_title(feature)
        ax.set_xlabel(
            f"sign({feature}) · log1p(abs({feature}))"
            if transformacion == "log"
            else feature
        )
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if "xlim" in config:
            ax.set_xlim(*config["xlim"])
        if "ylim" in config:
            ax.set_ylim(*config["ylim"])

        if curvas:
            ax.legend(title=col_cluster)

        plt.tight_layout()
        plt.show()




def plot_distr_series_clusters_tablas(
    df_clusters,
    df_distribuciones,
    df_original,
    col_id,
    col_cluster,
    columnas,
    ajustes=None,
    label_sin_cluster="SIN_CLUSTER",
    porcentaje_distribucion=40,
    escala_eje="real",
    banda="ic95",
):
    ajustes = ajustes or {}

    if not 0 < porcentaje_distribucion < 100:
        raise ValueError("porcentaje_distribucion debe estar entre 0 y 100.")
    if escala_eje not in ("real", "transformada"):
        raise ValueError("escala_eje debe ser 'real' o 'transformada'.")
    if banda not in ("ic95", "desvio", "iqr"):
        raise ValueError("banda debe ser 'ic95', 'desvio' o 'iqr'.")

    for nombre, datos, necesarias in (
        ("df_clusters", df_clusters, [col_id, col_cluster, "grupo"]),
        ("df_distribuciones", df_distribuciones,
         [col_id, "grupo", "punto_0", *columnas]),
        ("df_original", df_original, [col_id, "foto_mes"]),
    ):
        faltantes = set(necesarias) - set(datos.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas en {nombre}: {sorted(faltantes)}")

    corte = df_distribuciones["punto_0"].dropna().unique()
    if len(corte) != 1:
        raise ValueError("df_distribuciones debe tener un único punto_0.")
    corte = int(corte[0])
    corte_idx = (corte // 100) * 12 + (corte % 100) - 1

    mes = pd.to_numeric(df_original["foto_mes"], errors="raise")
    mes_idx = (mes // 100) * 12 + (mes % 100) - 1
    ids_baja = df_clusters.loc[df_clusters["grupo"].eq(1), col_id]
    meses_baja = mes_idx.loc[
        df_original[col_id].isin(ids_baja) & mes_idx.lt(corte_idx)
    ]
    if meses_baja.empty:
        raise ValueError(
            "No hay meses anteriores al corte para los BAJA clusterizados."
        )
    inicio_idx = int(meses_baja.min())
    fin_idx = int(meses_baja.max())
    meses_serie = list(range(inicio_idx, fin_idx + 1))
    etiquetas_meses = [
        str((indice // 12) * 100 + (indice % 12) + 1)
        for indice in meses_serie
    ]

    # Solo los CONTINUA sin cluster necesitan historial completo,
    # incluyendo el mes de corte.
    disponibles = (
        df_original.loc[
            mes_idx.between(inicio_idx, corte_idx), col_id
        ]
        .to_frame()
        .assign(_mes_idx=mes_idx.loc[
            mes_idx.between(inicio_idx, corte_idx)
        ].to_numpy())
        .groupby(col_id)["_mes_idx"]
        .nunique()
    )
    ids_completos = set(
        disponibles.index[disponibles.eq(corte_idx - inicio_idx + 1)]
    )
    ids_clusterizados = set(df_clusters[col_id])

    df_plot = df_distribuciones.merge(
        df_clusters[[col_id, col_cluster]],
        on=col_id,
        how="left",
        validate="one_to_one",
    )
    tiene_cluster = df_plot[col_id].isin(ids_clusterizados)
    continua_sin_cluster = (
        ~tiene_cluster
        & df_plot["grupo"].eq(0)
        & df_plot[col_id].isin(ids_completos)
    )
    df_plot = df_plot.loc[tiene_cluster | continua_sin_cluster].copy()
    df_plot[col_cluster] = (
        df_plot[col_cluster].astype(object).fillna(label_sin_cluster)
    )
    if df_plot.empty:
        raise ValueError("No hay clientes para graficar.")

    porcentaje_continua = (
        df_clusters.groupby(col_cluster)["grupo"]
        .apply(lambda s: 100 * s.eq(0).mean())
    )
    grupos = list(pd.unique(df_plot[col_cluster]))
    paleta = sns.color_palette(
        "tab10" if len(grupos) <= 10 else "husl", len(grupos)
    )
    colores = dict(zip(grupos, paleta))

    variables_temporales = list(dict.fromkeys(
        feature for feature in columnas if feature in df_original.columns
    ))
    mascara_tiempo = (
        mes_idx.between(inicio_idx, fin_idx)
        & df_original[col_id].isin(df_plot[col_id])
    )
    df_tiempo = df_original.loc[
        mascara_tiempo, [col_id, *variables_temporales]
    ].copy()
    df_tiempo["_mes_idx"] = mes_idx.loc[mascara_tiempo].to_numpy()
    df_tiempo = df_tiempo.merge(
        df_plot[[col_id, col_cluster]],
        on=col_id,
        how="inner",
        validate="many_to_one",
    )

    resultados = {}

    for feature in columnas:
        config = ajustes.get(feature, {})
        transformacion = config.get("transformacion")
        xscale = config.get("xscale", "linear")
        yscale = config.get("yscale", "linear")

        if transformacion not in (None, "log"):
            raise ValueError(f"Transformación desconocida: {transformacion}")
        if transformacion == "log" and xscale != "linear":
            raise ValueError(
                f"{feature}: no combines transformacion='log' "
                f"con xscale='{xscale}'."
            )

        series = [
            df_plot.loc[df_plot[col_cluster] == grupo, feature].dropna()
            for grupo in grupos
        ]
        tabla = pd.DataFrame({
            col_cluster: grupos,
            "n_valido": [len(s) for s in series],
            "% CONTINUA": [
                porcentaje_continua.get(grupo, np.nan)
                for grupo in grupos
            ],
            "media": [s.mean() for s in series],
            "sd": [s.std(ddof=1) for s in series],
        })

        validos = [
            i for i, s in enumerate(series)
            if len(s) >= 2 and s.nunique() >= 2
        ]
        p_global = np.nan
        p_pares = {}

        if len(validos) >= 2:
            muestras = [series[i].to_numpy(dtype=float) for i in validos]
            posthoc = stats.tukey_hsd(*muestras, equal_var=False)

            for pos_i, i in enumerate(validos):
                for pos_j, j in enumerate(validos):
                    if i < j:
                        p_pares[i, j] = float(
                            posthoc.pvalue[pos_i, pos_j]
                        )

            if len(validos) == len(grupos):
                p_global = float(
                    stats.f_oneway(*muestras, equal_var=False).pvalue
                )

        for grupo in grupos:
            tabla[f"vs {grupo}"] = ""

        comparaciones = []
        for i, grupo_fila in enumerate(grupos):
            tabla.at[i, f"vs {grupo_fila}"] = "—"
            for j in range(i + 1, len(grupos)):
                diferencia = float(
                    tabla.at[i, "media"] - tabla.at[j, "media"]
                )
                p_ajustado = p_pares.get((i, j), np.nan)
                comparaciones.append({
                    "cluster_fila": grupo_fila,
                    "cluster_columna": grupos[j],
                    "diferencia_medias": diferencia,
                    "p_ajustado": p_ajustado,
                })
                p_texto = (
                    f"{p_ajustado:.3g}"
                    if np.isfinite(p_ajustado) else "NA"
                )
                tabla.at[i, f"vs {grupos[j]}"] = (
                    f"Δ={diferencia:,.2f}\np={p_texto}"
                )

        resultados[feature] = {
            "tabla": tabla,
            "comparaciones": pd.DataFrame(
                comparaciones,
                columns=[
                    "cluster_fila", "cluster_columna",
                    "diferencia_medias", "p_ajustado",
                ],
            ),
            "p_global": p_global,
        }

        ancho = max(16, 8 + 2 * len(grupos))
        if feature in variables_temporales:
            alto = max(7, 4 + 0.5 * len(grupos))
            fig = plt.figure(figsize=(ancho, alto), layout="constrained")
            rejilla = fig.add_gridspec(
                2, 2,
                width_ratios=[
                    porcentaje_distribucion,
                    100 - porcentaje_distribucion,
                ],
                height_ratios=[3, 2],
            )
            ax = fig.add_subplot(rejilla[0, 0])
            ax_tiempo = fig.add_subplot(rejilla[0, 1])
            ax_tabla = fig.add_subplot(rejilla[1, :])
        else:
            alto = max(5, 3 + 0.5 * len(grupos))
            fig, (ax, ax_tabla) = plt.subplots(
                1, 2,
                figsize=(ancho, alto),
                gridspec_kw={"width_ratios": [
                    porcentaje_distribucion,
                    100 - porcentaje_distribucion,
                ]},
                layout="constrained",
            )

        curvas = 0
        for grupo, subset in zip(grupos, series):
            if len(subset) < 2 or subset.nunique() < 2:
                continue
            if transformacion == "log":
                subset = np.sign(subset) * np.log1p(np.abs(subset))

            sns.kdeplot(
                subset,
                ax=ax,
                label=str(grupo),
                color=colores[grupo],
                fill=True,
                alpha=0.3,
            )
            curvas += 1

        ax.set_title(feature)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if transformacion == "log" and escala_eje == "real":
            def formato_real(valor_transformado, _):
                valor = (
                    np.sign(valor_transformado)
                    * np.expm1(np.abs(valor_transformado))
                )
                return (
                    f"{valor:,.0f}" if abs(valor) >= 1
                    else f"{valor:.2g}"
                )

            ax.xaxis.set_major_formatter(FuncFormatter(formato_real))
            ax.set_xlabel(feature)
        else:
            ax.set_xlabel(
                f"sign({feature}) · log1p(abs({feature}))"
                if transformacion == "log" else feature
            )

        if "xlim" in config:
            limites = config["xlim"]
            if transformacion == "log" and escala_eje == "real":
                limites = tuple(
                    None if valor is None else
                    np.sign(valor) * np.log1p(np.abs(valor))
                    for valor in limites
                )
            ax.set_xlim(*limites)
        if "ylim" in config:
            ax.set_ylim(*config["ylim"])
        if curvas:
            ax.legend(title=col_cluster)

        if feature in variables_temporales:
            agrupado = df_tiempo.groupby(
                [col_cluster, "_mes_idx"], observed=True
            )[feature]
            n = agrupado.count()
            if banda == "iqr":
                centro = agrupado.median()
                inferior = agrupado.quantile(0.25)
                superior = agrupado.quantile(0.75)
            else:
                centro = agrupado.mean()
                desvio = agrupado.std()
                ancho_banda = (
                    desvio if banda == "desvio"
                    else 1.96 * desvio / np.sqrt(n)
                )
                inferior = centro - ancho_banda
                superior = centro + ancho_banda

            minimo = df_tiempo[feature].min(skipna=True)
            maximo = df_tiempo[feature].max(skipna=True)
            if pd.notna(minimo):
                inferior = inferior.clip(lower=minimo)
            if pd.notna(maximo):
                superior = superior.clip(upper=maximo)

            x = np.arange(len(meses_serie))
            for grupo in grupos:
                indice = pd.MultiIndex.from_product(
                    [[grupo], meses_serie],
                    names=[col_cluster, "_mes_idx"],
                )
                y = centro.reindex(indice).to_numpy(dtype=float)
                bajo = inferior.reindex(indice).to_numpy(dtype=float)
                alto_banda = superior.reindex(indice).to_numpy(dtype=float)
                ax_tiempo.fill_between(
                    x, bajo, alto_banda,
                    color=colores[grupo], alpha=0.12, linewidth=0,
                )
                ax_tiempo.plot(
                    x, y, color=colores[grupo], linewidth=2,
                    marker="o", markersize=5, label=str(grupo),
                )

            ax_tiempo.set_title(
                f"{feature}: " + {
                    "ic95": "media e IC 95%",
                    "desvio": "media ± desvío",
                    "iqr": "mediana y Q25–Q75",
                }[banda]
            )
            ax_tiempo.set_xticks(x, etiquetas_meses)
            ax_tiempo.set_xlim(-0.3, len(x) - 0.7)
            ax_tiempo.set_ylabel(feature)
            ax_tiempo.grid(axis="y", alpha=0.25)
            ax_tiempo.legend(title=col_cluster)

        ax_tabla.axis("off")
        p_texto = f"{p_global:.3g}" if np.isfinite(p_global) else "NA"
        ax_tabla.set_title(f"Welch ANOVA: p={p_texto}", pad=12)

        tabla_visible = tabla.copy()
        for col in ("media", "sd"):
            tabla_visible[col] = tabla_visible[col].map(
                lambda v: f"{v:,.2f}" if pd.notna(v) else "NA"
            )
        tabla_visible["% CONTINUA"] = tabla_visible["% CONTINUA"].map(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "NA"
        )
        tabla_grafico = ax_tabla.table(
            cellText=tabla_visible.astype(str).to_numpy(),
            colLabels=[str(c) for c in tabla_visible.columns],
            cellLoc="center",
            colLoc="center",
            bbox=[0, 0.05, 1, 0.85],
        )
        tabla_grafico.auto_set_font_size(False)
        tabla_grafico.set_fontsize(8)
        plt.show()

    return resultados

def plot_distr_series_clusters_tablas_no_alineado(
    df_clusters,
    df_distribuciones,
    df_original,
    col_id,
    col_cluster,
    columnas,
    k,
    ajustes=None,
    label_sin_cluster="SIN_CLUSTER",
    porcentaje_distribucion=40,
    escala_eje="real",
    banda="ic95",
):
    ajustes = ajustes or {}

    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k debe ser un entero mayor o igual a 1.")

    if not 0 < porcentaje_distribucion < 100:
        raise ValueError("porcentaje_distribucion debe estar entre 0 y 100.")
    if escala_eje not in ("real", "transformada"):
        raise ValueError("escala_eje debe ser 'real' o 'transformada'.")
    if banda not in ("ic95", "desvio", "iqr"):
        raise ValueError("banda debe ser 'ic95', 'desvio' o 'iqr'.")

    for nombre, datos, necesarias in (
        ("df_clusters", df_clusters, [col_id, col_cluster, "grupo"]),
        ("df_distribuciones", df_distribuciones,
         [col_id, "grupo", "punto_0", *columnas]),
        ("df_original", df_original, [col_id, "foto_mes"]),
    ):
        faltantes = set(necesarias) - set(datos.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas en {nombre}: {sorted(faltantes)}")

    ids_clusterizados = set(df_clusters[col_id])

    df_plot = df_distribuciones.merge(
        df_clusters[[col_id, col_cluster]],
        on=col_id,
        how="left",
        validate="one_to_one",
    )
    tiene_cluster = df_plot[col_id].isin(ids_clusterizados)
    continua_sin_cluster = ~tiene_cluster & df_plot["grupo"].eq(0)
    df_plot = df_plot.loc[tiene_cluster | continua_sin_cluster].copy()
    df_plot[col_cluster] = (
        df_plot[col_cluster].astype(object).fillna(label_sin_cluster)
    )
    if df_plot.empty:
        raise ValueError("No hay clientes para graficar.")
    if df_plot["punto_0"].isna().any():
        raise ValueError("Hay clientes seleccionados sin punto_0.")

    porcentaje_continua = (
        df_clusters.groupby(col_cluster)["grupo"]
        .apply(lambda s: 100 * s.eq(0).mean())
    )
    grupos = list(pd.unique(df_plot[col_cluster]))
    paleta = sns.color_palette(
        "tab10" if len(grupos) <= 10 else "husl", len(grupos)
    )
    colores = dict(zip(grupos, paleta))

    variables_temporales = list(dict.fromkeys(
        feature for feature in columnas if feature in df_original.columns
    ))
    df_tiempo = df_original.loc[
        df_original[col_id].isin(df_plot[col_id]),
        [col_id, "foto_mes", *variables_temporales],
    ].copy()
    df_tiempo = df_tiempo.merge(
        df_plot[[col_id, col_cluster, "punto_0"]],
        on=col_id,
        how="inner",
        validate="many_to_one",
    )
    mes = pd.to_numeric(df_tiempo["foto_mes"], errors="raise")
    corte = pd.to_numeric(df_tiempo["punto_0"], errors="raise")
    mes_idx = (mes // 100) * 12 + (mes % 100) - 1
    corte_idx = (corte // 100) * 12 + (corte % 100) - 1
    df_tiempo["_tiempo_relativo"] = mes_idx - corte_idx
    df_tiempo = df_tiempo.loc[
        df_tiempo["_tiempo_relativo"].between(-k, -1)
    ].copy()
    for feature in variables_temporales:
        df_tiempo[feature] = pd.to_numeric(
            df_tiempo[feature], errors="raise"
        ).astype(float)

    resultados = {}

    for feature in columnas:
        config = ajustes.get(feature, {})
        transformacion = config.get("transformacion")
        xscale = config.get("xscale", "linear")
        yscale = config.get("yscale", "linear")

        if transformacion not in (None, "log"):
            raise ValueError(f"Transformación desconocida: {transformacion}")
        if transformacion == "log" and xscale != "linear":
            raise ValueError(
                f"{feature}: no combines transformacion='log' "
                f"con xscale='{xscale}'."
            )

        series = [
            pd.to_numeric(
                df_plot.loc[df_plot[col_cluster] == grupo, feature],
                errors="raise",
            ).astype(float).dropna()
            for grupo in grupos
        ]
        tabla = pd.DataFrame({
            col_cluster: grupos,
            "n_valido": [len(s) for s in series],
            "% CONTINUA": [
                porcentaje_continua.get(grupo, np.nan)
                for grupo in grupos
            ],
            "media": [s.mean() for s in series],
            "sd": [s.std(ddof=1) for s in series],
        })

        validos = [
            i for i, s in enumerate(series)
            if len(s) >= 2 and s.nunique() >= 2
        ]
        p_global = np.nan
        p_pares = {}

        if len(validos) >= 2:
            muestras = [series[i].to_numpy(dtype=float) for i in validos]
            posthoc = stats.tukey_hsd(*muestras, equal_var=False)

            for pos_i, i in enumerate(validos):
                for pos_j, j in enumerate(validos):
                    if i < j:
                        p_pares[i, j] = float(
                            posthoc.pvalue[pos_i, pos_j]
                        )

            if len(validos) == len(grupos):
                p_global = float(
                    stats.f_oneway(*muestras, equal_var=False).pvalue
                )

        for grupo in grupos:
            tabla[f"vs {grupo}"] = ""

        comparaciones = []
        for i, grupo_fila in enumerate(grupos):
            tabla.at[i, f"vs {grupo_fila}"] = "—"
            for j in range(i + 1, len(grupos)):
                diferencia = float(
                    tabla.at[i, "media"] - tabla.at[j, "media"]
                )
                p_ajustado = p_pares.get((i, j), np.nan)
                comparaciones.append({
                    "cluster_fila": grupo_fila,
                    "cluster_columna": grupos[j],
                    "diferencia_medias": diferencia,
                    "p_ajustado": p_ajustado,
                })
                p_texto = (
                    f"{p_ajustado:.3g}"
                    if np.isfinite(p_ajustado) else "NA"
                )
                tabla.at[i, f"vs {grupos[j]}"] = (
                    f"Δ={diferencia:,.2f}\np={p_texto}"
                )

        resultados[feature] = {
            "tabla": tabla,
            "comparaciones": pd.DataFrame(
                comparaciones,
                columns=[
                    "cluster_fila", "cluster_columna",
                    "diferencia_medias", "p_ajustado",
                ],
            ),
            "p_global": p_global,
        }

        ancho = max(16, 8 + 2 * len(grupos))
        if feature in variables_temporales:
            alto = max(7, 4 + 0.5 * len(grupos))
            fig = plt.figure(figsize=(ancho, alto), layout="constrained")
            rejilla = fig.add_gridspec(
                2, 2,
                width_ratios=[
                    porcentaje_distribucion,
                    100 - porcentaje_distribucion,
                ],
                height_ratios=[3, 2],
            )
            ax = fig.add_subplot(rejilla[0, 0])
            ax_tiempo = fig.add_subplot(rejilla[0, 1])
            ax_tabla = fig.add_subplot(rejilla[1, :])
        else:
            alto = max(5, 3 + 0.5 * len(grupos))
            fig, (ax, ax_tabla) = plt.subplots(
                1, 2,
                figsize=(ancho, alto),
                gridspec_kw={"width_ratios": [
                    porcentaje_distribucion,
                    100 - porcentaje_distribucion,
                ]},
                layout="constrained",
            )

        curvas = 0
        for grupo, subset in zip(grupos, series):
            if len(subset) < 2 or subset.nunique() < 2:
                continue
            if transformacion == "log":
                subset = np.sign(subset) * np.log1p(np.abs(subset))

            sns.kdeplot(
                subset,
                ax=ax,
                label=str(grupo),
                color=colores[grupo],
                fill=True,
                alpha=0.3,
            )
            curvas += 1

        ax.set_title(feature)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if transformacion == "log" and escala_eje == "real":
            def formato_real(valor_transformado, _):
                valor = (
                    np.sign(valor_transformado)
                    * np.expm1(np.abs(valor_transformado))
                )
                return (
                    f"{valor:,.0f}" if abs(valor) >= 1
                    else f"{valor:.2g}"
                )

            ax.xaxis.set_major_formatter(FuncFormatter(formato_real))
            ax.set_xlabel(feature)
        else:
            ax.set_xlabel(
                f"sign({feature}) · log1p(abs({feature}))"
                if transformacion == "log" else feature
            )

        if "xlim" in config:
            limites = config["xlim"]
            if transformacion == "log" and escala_eje == "real":
                limites = tuple(
                    None if valor is None else
                    np.sign(valor) * np.log1p(np.abs(valor))
                    for valor in limites
                )
            ax.set_xlim(*limites)
        if "ylim" in config:
            ax.set_ylim(*config["ylim"])
        if curvas:
            ax.legend(title=col_cluster)

        if feature in variables_temporales:
            agrupado = df_tiempo.groupby(
                [col_cluster, "_tiempo_relativo"], observed=True
            )[feature]
            n = agrupado.count()
            if banda == "iqr":
                centro = agrupado.median()
                inferior = agrupado.quantile(0.25)
                superior = agrupado.quantile(0.75)
            else:
                centro = agrupado.mean()
                desvio = agrupado.std()
                ancho_banda = (
                    desvio if banda == "desvio"
                    else 1.96 * desvio / np.sqrt(n)
                )
                inferior = centro - ancho_banda
                superior = centro + ancho_banda

            minimo = df_tiempo[feature].min(skipna=True)
            maximo = df_tiempo[feature].max(skipna=True)
            if pd.notna(minimo):
                inferior = inferior.clip(lower=minimo)
            if pd.notna(maximo):
                superior = superior.clip(upper=maximo)

            x = np.arange(-k, 0)
            for grupo in grupos:
                indice = pd.MultiIndex.from_product(
                    [[grupo], range(-k, 0)],
                    names=[col_cluster, "_tiempo_relativo"],
                )
                y = centro.reindex(indice).to_numpy(dtype=float)
                bajo = inferior.reindex(indice).to_numpy(dtype=float)
                alto_banda = superior.reindex(indice).to_numpy(dtype=float)
                ax_tiempo.fill_between(
                    x, bajo, alto_banda,
                    color=colores[grupo], alpha=0.12, linewidth=0,
                )
                ax_tiempo.plot(
                    x, y, color=colores[grupo], linewidth=2,
                    marker="o", markersize=5, label=str(grupo),
                )

            ax_tiempo.set_title(
                f"{feature}: " + {
                    "ic95": "media e IC 95%",
                    "desvio": "media ± desvío",
                    "iqr": "mediana y Q25–Q75",
                }[banda]
            )
            ax_tiempo.set_xticks(x)
            ax_tiempo.set_xlim(-k - 0.3, -0.7)
            ax_tiempo.set_xlabel("Mes relativo a mes de baja (mes random q garantice k meses de historia para los que continuan)")
            ax_tiempo.set_ylabel(feature)
            ax_tiempo.grid(axis="y", alpha=0.25)
            ax_tiempo.legend(title=col_cluster)

        ax_tabla.axis("off")
        p_texto = f"{p_global:.3g}" if np.isfinite(p_global) else "NA"
        ax_tabla.set_title(f"Welch ANOVA: p={p_texto}", pad=12)

        tabla_visible = tabla.copy()
        for col in ("media", "sd"):
            tabla_visible[col] = tabla_visible[col].map(
                lambda v: f"{v:,.2f}" if pd.notna(v) else "NA"
            )
        tabla_visible["% CONTINUA"] = tabla_visible["% CONTINUA"].map(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "NA"
        )
        tabla_grafico = ax_tabla.table(
            cellText=tabla_visible.astype(str).to_numpy(),
            colLabels=[str(c) for c in tabla_visible.columns],
            cellLoc="center",
            colLoc="center",
            bbox=[0, 0.05, 1, 0.85],
        )
        tabla_grafico.auto_set_font_size(False)
        tabla_grafico.set_fontsize(8)
        plt.show()

    return resultados
