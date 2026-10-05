import pandas as pd
import numpy as np
import re

def agregar_rangos_por_mes(df, columnas):
    resultado = df.copy()

    for columna in columnas:
        resultado[f"{columna}_rango"] = (
            resultado.groupby("foto_mes")[columna]
            .rank(method="min", ascending=True)
            .astype("Int64")
        )

    return resultado

def mes_a_indice(serie):
    """
    Convierte YYYYMM a un índice mensual consecutivo.
    Ejemplo: la diferencia entre 202501 y 202412 será 1.
    """
    mes = pd.to_numeric(serie, errors="coerce").astype("Int64")
    return (mes // 100) * 12 + (mes % 100) - 1


def indice_a_mes(indice):
    """Convierte el índice mensual nuevamente a YYYYMM."""
    anio, mes_desde_cero = divmod(int(indice), 12)
    return anio * 100 + mes_desde_cero + 1



def construir_dataset_por_cliente(
    df,
    columnas_features,
    k,
    proporcion_continua=1.0,
    seed=214363,
):
    """
    Construye una fila por cliente.

    Para las bajas:
        punto_0 = mes_baja real.

    Para los CONTINUA:
        punto_0 = mes ficticio, respetando la distribución
        mensual de las bajas.

    Parámetros
    ----------
    df:
        DataFrame original cliente/mes.

    columnas_features:
        Lista de variables que se quiere reconstruir hacia atrás.

    k:
        Cantidad máxima de meses anteriores al punto 0.

    proporcion_continua:
        Cantidad de CONTINUA respecto de las bajas.
        1.0 genera aproximadamente uno por cada baja.
        0.2 genera aproximadamente uno cada cinco bajas.
    """

    datos = df.copy()

    columnas_necesarias = {
        "numero_de_cliente",
        "foto_mes",
        "mes_baja",
        *columnas_features,
    }

    faltantes = columnas_necesarias - set(datos.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    # Debe existir como máximo una fila por cliente y mes.
    if datos.duplicated(["numero_de_cliente", "foto_mes"]).any():
        raise ValueError(
            "Hay más de una fila para algún numero_de_cliente/foto_mes"
        )

    # Índices mensuales para poder restar meses correctamente.
    datos["_mes_idx"] = mes_a_indice(datos["foto_mes"])
    datos["_baja_idx"] = mes_a_indice(datos["mes_baja"])

    # ---------------------------------------------------------
    # 1. Puntos 0 reales de los clientes que se dieron de baja
    # ---------------------------------------------------------

    puntos_baja = (
        datos.loc[
            datos["_baja_idx"].notna(),
            ["numero_de_cliente", "_baja_idx"],
        ]
        .drop_duplicates()
    )

    # Cada cliente debe tener un único mes de baja.
    if puntos_baja["numero_de_cliente"].duplicated().any():
        raise ValueError("Hay clientes con más de un mes_baja")

    puntos_baja = puntos_baja.rename(
        columns={"_baja_idx": "_punto_0_idx"}
    )

    puntos_baja["grupo"] = "BAJA"
    puntos_baja["punto_0_real"] = True

    ids_baja = set(puntos_baja["numero_de_cliente"])

    # Clientes que nunca tienen mes_baja.
    ids_continua = (
        set(datos["numero_de_cliente"].unique()) - ids_baja
    )

    # ---------------------------------------------------------
# 2. Punto 0 ficticio para TODOS los clientes CONTINUA
# ---------------------------------------------------------

    ids_continua = sorted(
        set(datos["numero_de_cliente"].unique()) - ids_baja
    )

    rng = np.random.default_rng(seed)

    # Meses de baja reales, una vez por cliente con baja.
    # Al sortear desde esta lista se mantiene su distribución temporal.
    meses_baja = puntos_baja["_punto_0_idx"].to_numpy()

    puntos_continua = pd.DataFrame({
        "numero_de_cliente": ids_continua,

        # Se sortea el mes, no el cliente.
        "_punto_0_idx": rng.choice(
            meses_baja,
            size=len(ids_continua),
            replace=True,
        ),

        "grupo": "CONTINUA",
        "punto_0_real": False,
    })
    # Unimos todos los clientes con su punto 0.
    puntos = pd.concat(
        [puntos_baja, puntos_continua],
        ignore_index=True,
    )

    # Convertimos nuevamente el punto 0 al formato YYYYMM.
    puntos["punto_0"] = puntos["_punto_0_idx"].map(
        indice_a_mes
    )
    # ---------------------------------------------------------
    # 3. Calculamos el tiempo relativo de cada observación
    # ---------------------------------------------------------

    trayectoria = datos.merge(
        puntos[
            ["numero_de_cliente", "_punto_0_idx"]
        ],
        on="numero_de_cliente",
        how="inner",
    )

    trayectoria["_tiempo_relativo"] = (
        trayectoria["_mes_idx"]
        - trayectoria["_punto_0_idx"]
    )

    # Conservamos solamente los K meses anteriores.
    trayectoria = trayectoria.loc[
        trayectoria["_tiempo_relativo"].between(-k, -1)
    ].copy()

    # ---------------------------------------------------------
    # 4. Pasamos de cliente/mes a una fila por cliente
    # ---------------------------------------------------------

    columnas_esperadas = pd.MultiIndex.from_product(
        [
            columnas_features,
            range(-1, -k - 1, -1),
        ]
    )

    valores = trayectoria.pivot(
        index="numero_de_cliente",
        columns="_tiempo_relativo",
        values=columnas_features,
    )

    # Agrega columnas para meses no observados.
    valores = valores.reindex(columns=columnas_esperadas)

    valores.columns = [
        f"{feature}_{periodo}"
        for feature, periodo in valores.columns
    ]

    # Indicadores de qué meses fueron realmente observados.
    observados = (
        trayectoria.assign(_observado=1)
        .pivot(
            index="numero_de_cliente",
            columns="_tiempo_relativo",
            values="_observado",
        )
        .reindex(columns=range(-1, -k - 1, -1))
        .fillna(0)
        .astype("int8")
    )

    observados.columns = [
        f"observado_{periodo}"
        for periodo in observados.columns
    ]

    # Incluimos también clientes sin observaciones dentro de K.
    ids_salida = puntos["numero_de_cliente"].tolist()

    valores = valores.reindex(ids_salida)

    # Los indicadores se usan internamente, pero no se agregan a la salida.
    observados = observados.reindex(
        ids_salida,
        fill_value=0,
    )

    # Calculamos la historia consecutiva como una serie independiente.
    historia_consecutiva = (
        observados
        .cumprod(axis=1)
        .sum(axis=1)
        .rename("historia_consecutiva")
    )

    # Creamos la salida incluyendo únicamente las columnas necesarias.
    salida = (
        puntos[
            [
                "numero_de_cliente",
                "grupo",
                "punto_0",
            ]
        ]
        .set_index("numero_de_cliente")
        .join(valores)
        .join(historia_consecutiva)
        .reset_index()
    )

    # Debe existir exactamente una fila por cliente.
    assert len(salida) == datos["numero_de_cliente"].nunique()

    return salida



def agregar_delta_lags(df, columnas_excluir):
    """
    Crea deltas únicamente entre períodos consecutivos.

    Ejemplo:
        delta_saldo_-1_-2 = saldo_-1 - saldo_-2

    columnas_excluir puede contener nombres de columnas completas
    o nombres base de features.
    """

    resultado = df.copy()
    excluidas = set(columnas_excluir)

    # Reconoce columnas terminadas en _-1, _-2, etc.
    patron = re.compile(r"^(.*)_(-\d+)$")

    # {feature: {periodo: nombre_columna}}
    columnas_por_feature = {}

    for columna in df.columns:
        coincidencia = patron.match(columna)

        if coincidencia is None:
            continue

        feature = coincidencia.group(1)
        periodo = int(coincidencia.group(2))

        # Permite excluir la columna o la feature completa.
        if columna in excluidas or feature in excluidas:
            continue

        # Evita volver a procesar deltas si se ejecuta dos veces.
        if feature.startswith("delta_"):
            continue

        columnas_por_feature.setdefault(feature, {})[periodo] = columna

    # Creamos únicamente deltas de un paso.
    for feature, columnas_periodos in columnas_por_feature.items():

        for periodo_actual in sorted(
            columnas_periodos,
            reverse=True,
        ):
            periodo_anterior = periodo_actual - 1

            if periodo_anterior not in columnas_periodos:
                continue

            columna_actual = columnas_periodos[periodo_actual]
            columna_anterior = columnas_periodos[periodo_anterior]

            nombre_delta = (
                f"delta_{feature}_"
                f"{periodo_actual}_{periodo_anterior}"
            )

            resultado[nombre_delta] = (
                df[columna_actual]
                - df[columna_anterior]
            )

    return resultado


def agregar_mes_baja(
    df,
    cliente_col="numero_de_cliente",
    mes_col="foto_mes"
):
    resultado = df.copy()

    ultimo_mes_dataset = resultado[mes_col].max()

    ultimo_mes_cliente = (
        resultado
        .groupby(cliente_col)[mes_col]
        .max()
    )

    mes_baja = pd.Series(
        pd.NA,
        index=ultimo_mes_cliente.index,
        dtype="Int64"
    )

    clientes_baja = ultimo_mes_cliente < ultimo_mes_dataset
    ultimo_mes = ultimo_mes_cliente.loc[clientes_baja]

    # Calcula correctamente el mes siguiente, incluyendo diciembre → enero
    mes_baja.loc[clientes_baja] = (
        (ultimo_mes // 100 + (ultimo_mes % 100 == 12)) * 100
        + (ultimo_mes % 100) % 12 + 1
    ).astype("Int64")

    resultado["mes_baja"] = (
        resultado[cliente_col]
        .map(mes_baja)
        .astype("Int64")
    )

    return resultado

def dataset_por_cliente_meses_alineados(
    df,
    columnas_features,
    k,
    mes_corte=202108,
):
    datos = df.copy()

    columnas_necesarias = {
        "numero_de_cliente",
        "foto_mes",
        *columnas_features,
    }

    faltantes = columnas_necesarias - set(datos.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {sorted(faltantes)}"
        )

    if datos.duplicated(
        ["numero_de_cliente", "foto_mes"]
    ).any():
        raise ValueError(
            "Hay más de una fila para algún "
            "numero_de_cliente/foto_mes"
        )

    # Convertimos YYYYMM a un índice mensual consecutivo.
    datos["_mes_idx"] = mes_a_indice(
        datos["foto_mes"]
    )

    anio_corte = mes_corte // 100
    numero_mes_corte = mes_corte % 100

    corte_idx = (
        anio_corte * 12
        + numero_mes_corte
        - 1
    )

    mes_anterior_idx = corte_idx - 1

    # Se consideran solamente clientes que ya existían
    # hasta el mes de corte. Esto evita incluir clientes
    # que aparecen por primera vez después del corte.
    clientes = (
        datos.loc[
            datos["_mes_idx"] <= corte_idx,
            "numero_de_cliente",
        ]
        .drop_duplicates()
        .sort_values()
    )

    presentes_corte = set(
        datos.loc[
            datos["_mes_idx"] == corte_idx,
            "numero_de_cliente",
        ]
    )

    presentes_mes_anterior = set(
        datos.loc[
            datos["_mes_idx"] == mes_anterior_idx,
            "numero_de_cliente",
        ]
    )

    salida_base = pd.DataFrame({
        "numero_de_cliente": clientes
    })

    # CONTINUA:
    # aparece en el mes de corte.
    #
    # BAJA:
    # aparece el mes anterior, pero no en el corte.
    #
    # BAJA_PREVIA:
    # no aparece ni en el corte ni en el mes anterior.
    salida_base["grupo"] = np.select(
        [
            salida_base["numero_de_cliente"].isin(
                presentes_corte
            ),
            salida_base["numero_de_cliente"].isin(
                presentes_mes_anterior
            ),
        ],
        [
            "CONTINUA",
            "BAJA",
        ],
        default="BAJA_PREVIA",
    )

    salida_base["punto_0"] = mes_corte

    # Cantidad total de meses observados antes del corte.
    largo_historial = (
        datos.loc[
            datos["_mes_idx"] < corte_idx
        ]
        .groupby("numero_de_cliente")["_mes_idx"]
        .nunique()
        .reindex(clientes, fill_value=0)
        .rename("largo_historial")
    )

    # Calculamos la posición relativa respecto del corte.
    trayectoria = datos.loc[
        datos["numero_de_cliente"].isin(clientes)
    ].copy()

    trayectoria["_tiempo_relativo"] = (
        trayectoria["_mes_idx"] - corte_idx
    )

    # Conservamos únicamente los K meses anteriores.
    trayectoria = trayectoria.loc[
        trayectoria["_tiempo_relativo"].between(
            -k,
            -1,
        )
    ]

    # Pasamos de una fila por cliente/mes
    # a una fila por cliente.
    valores = trayectoria.pivot(
        index="numero_de_cliente",
        columns="_tiempo_relativo",
        values=columnas_features,
    )

    columnas_esperadas = pd.MultiIndex.from_product(
        [
            columnas_features,
            range(-1, -k - 1, -1),
        ]
    )

    valores = valores.reindex(
        columns=columnas_esperadas
    )

    valores.columns = [
        f"{feature}_{periodo}"
        for feature, periodo in valores.columns
    ]

    valores = valores.reindex(clientes)

    salida = (
        salida_base
        .set_index("numero_de_cliente")
        .join(largo_historial)
        .join(valores)
        .reset_index()
    )

    # Deltas entre meses consecutivos:
    # valor más nuevo menos valor más viejo.
    for feature in columnas_features:
        for periodo_actual in range(-1, -k, -1):

            periodo_anterior = periodo_actual - 1

            columna_actual = (
                f"{feature}_{periodo_actual}"
            )

            columna_anterior = (
                f"{feature}_{periodo_anterior}"
            )

            nombre_delta = (
                f"delta_{feature}_"
                f"{periodo_actual}_{periodo_anterior}"
            )

            salida[nombre_delta] = (
                salida[columna_actual]
                - salida[columna_anterior]
            )

    return salida

def dataset_por_cliente_tendencia(
    df,
    columnas_features,
    k,
    mes_corte=202108,
    columnas_sin_calculos_temporales=(),
):
    datos = df.copy()

    columnas_necesarias = {
        "numero_de_cliente",
        "foto_mes",
        *columnas_features,
    }
    faltantes = columnas_necesarias - set(datos.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    if datos.duplicated(["numero_de_cliente", "foto_mes"]).any():
        raise ValueError(
            "Hay más de una fila para algún numero_de_cliente/foto_mes"
        )

    datos["_mes_idx"] = mes_a_indice(datos["foto_mes"])

    anio_corte = mes_corte // 100
    numero_mes_corte = mes_corte % 100
    corte_idx = anio_corte * 12 + numero_mes_corte - 1
    mes_anterior_idx = corte_idx - 1

    clientes = (
        datos.loc[
            datos["_mes_idx"] <= corte_idx,
            "numero_de_cliente",
        ]
        .drop_duplicates()
        .sort_values()
    )

    presentes_corte = set(
        datos.loc[
            datos["_mes_idx"] == corte_idx,
            "numero_de_cliente",
        ]
    )
    presentes_mes_anterior = set(
        datos.loc[
            datos["_mes_idx"] == mes_anterior_idx,
            "numero_de_cliente",
        ]
    )

    salida_base = pd.DataFrame({"numero_de_cliente": clientes})
    salida_base["grupo"] = np.select(
        [
            salida_base["numero_de_cliente"].isin(presentes_corte),
            salida_base["numero_de_cliente"].isin(presentes_mes_anterior),
        ],
        ["CONTINUA", "BAJA"],
        default="BAJA_PREVIA",
    )
    salida_base["punto_0"] = mes_corte

    largo_historial = (
        datos.loc[datos["_mes_idx"] < corte_idx]
        .groupby("numero_de_cliente")["_mes_idx"]
        .nunique()
        .reindex(clientes, fill_value=0)
        .rename("largo_historial")
    )

    trayectoria = datos.loc[
        datos["numero_de_cliente"].isin(clientes)
    ].copy()
    trayectoria["_tiempo_relativo"] = (
        trayectoria["_mes_idx"] - corte_idx
    )
    trayectoria = trayectoria.loc[
        trayectoria["_tiempo_relativo"].between(-k, -1)
    ]

    valores = trayectoria.pivot(
        index="numero_de_cliente",
        columns="_tiempo_relativo",
        values=columnas_features,
    )
    columnas_esperadas = pd.MultiIndex.from_product(
        [columnas_features, range(-1, -k - 1, -1)]
    )
    valores = valores.reindex(columns=columnas_esperadas)
    valores.columns = [
        f"{feature}_{periodo}"
        for feature, periodo in valores.columns
    ]
    valores = valores.reindex(clientes)

    salida = (
        salida_base
        .set_index("numero_de_cliente")
        .join(largo_historial)
    )

    for feature in columnas_features:
        meses = valores[
            [
                f"{feature}_{periodo}"
                for periodo in range(-1, -k - 1, -1)
            ]
        ]
        salida[feature] = meses.iloc[:, 0]

        if feature in columnas_sin_calculos_temporales:
            continue

        completo = meses.notna().all(axis=1)
        columna_delta = f"{feature}_delt_prom"
        columna_relativo = f"{feature}_delt_rel"
        columna_cambio = f"{feature}_delt_tend"

        if k < 2:
            salida[columna_delta] = np.nan
            salida[columna_relativo] = np.nan
        else:
            delta = (
                (meses.iloc[:, 0] - meses.iloc[:, -1]) / (k - 1)
            ).where(completo)
            nivel = meses.abs().mean(axis=1, skipna=False)
            relativo = delta / nivel.replace(0, np.nan)
            relativo = relativo.where(nivel.ne(0), 0.0)

            salida[columna_delta] = delta
            salida[columna_relativo] = relativo

        if k < 3:
            salida[columna_cambio] = np.nan
        else:
            tramos_recientes = (k - 1) // 2
            tramos_antiguos = (k - 1) - tramos_recientes
            medio = meses.iloc[:, tramos_recientes]

            pendiente_reciente = (
                meses.iloc[:, 0] - medio
            ) / tramos_recientes
            pendiente_antigua = (
                medio - meses.iloc[:, -1]
            ) / tramos_antiguos

            salida[columna_cambio] = (
                pendiente_reciente - pendiente_antigua
            ).where(completo)

    return salida.reset_index()

def dataset_por_clientes_tendencia_no_alineado(
    df,
    columnas_features,
    k,
    columnas_sin_calculos_temporales=(),
    seed=214363,
):
    if k < 1:
        raise ValueError("k debe ser al menos 1.")

    datos = df.copy()
    necesarias = {
        "numero_de_cliente", "foto_mes", "mes_baja",
        *columnas_features,
    }
    faltantes = necesarias - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    if datos.duplicated(["numero_de_cliente", "foto_mes"]).any():
        raise ValueError(
            "Hay más de una fila para algún numero_de_cliente/foto_mes"
        )

    datos["_mes_idx"] = mes_a_indice(datos["foto_mes"])
    datos["_baja_idx"] = mes_a_indice(datos["mes_baja"])

    # Un cliente con mes_baja informado nunca puede ser CONTINUA.
    puntos_baja = (
        datos.loc[
            datos["_baja_idx"].notna(),
            ["numero_de_cliente", "_baja_idx"],
        ]
        .drop_duplicates()
    )
    if puntos_baja["numero_de_cliente"].duplicated().any():
        raise ValueError("Hay clientes con más de un mes_baja")
    ids_baja = set(puntos_baja["numero_de_cliente"])
    puntos_baja = puntos_baja.rename(
        columns={"_baja_idx": "_punto_0_idx"}
    )

    # Se conservan las BAJA con registros en los k meses previos.
    meses_baja = datos[["numero_de_cliente", "_mes_idx"]].merge(
        puntos_baja,
        on="numero_de_cliente",
        how="inner",
        validate="many_to_one",
    )
    diferencia = (
        meses_baja["_mes_idx"] - meses_baja["_punto_0_idx"]
    )
    n_previos = (
        meses_baja.loc[
            diferencia.between(-k, -1)
        ]
        .groupby("numero_de_cliente")
        .size()
    )
    puntos_baja = puntos_baja.loc[
        puntos_baja["numero_de_cliente"].map(n_previos).eq(k).fillna(False)
    ].copy()
    if puntos_baja.empty:
        raise ValueError("No hay BAJA con los k meses anteriores completos.")
    puntos_baja["grupo"] = "BAJA"

    # Los CONTINUA deben permanecer sin interrupciones desde su ingreso
    # hasta el último mes disponible en el dataset.
    ultimo_mes = int(datos["_mes_idx"].max())
    historia = datos.groupby("numero_de_cliente")["_mes_idx"].agg(
        ["min", "max", "nunique"]
    )
    historia = historia.loc[~historia.index.isin(ids_baja)]
    historia = historia.loc[
        historia["max"].eq(ultimo_mes)
        & historia["nunique"].eq(ultimo_mes - historia["min"] + 1)
    ]

    # La distribución objetivo se calcula con las BAJA elegibles.
    frecuencias = puntos_baja["_punto_0_idx"].value_counts().sort_index()
    meses_corte = frecuencias.index.to_numpy(dtype=int)
    elegibles = []
    for cliente, primer_mes in historia["min"].items():
        posibles = meses_corte[meses_corte - k >= int(primer_mes)]
        if len(posibles):
            elegibles.append((cliente, posibles))

    rng = np.random.default_rng(seed)
    objetivos = (
        frecuencias / frecuencias.sum() * len(elegibles)
    ).to_dict()
    asignados = {mes: 0 for mes in meses_corte}
    cortes_continua = {}

    # Se asignan primero los clientes con menos cortes posibles.
    orden = rng.permutation(len(elegibles))
    orden = sorted(orden, key=lambda i: len(elegibles[i][1]))
    for i in orden:
        cliente, posibles = elegibles[i]
        deficit = np.array([
            objetivos[mes] - asignados[mes] for mes in posibles
        ])
        mejores = posibles[np.isclose(deficit, deficit.max())]
        mes_elegido = int(rng.choice(mejores))
        cortes_continua[cliente] = mes_elegido
        asignados[mes_elegido] += 1

    puntos_continua = pd.DataFrame({
        "numero_de_cliente": list(cortes_continua),
        "_punto_0_idx": list(cortes_continua.values()),
        "grupo": "CONTINUA",
    })
    puntos = pd.concat(
        [puntos_baja, puntos_continua],
        ignore_index=True,
    )
    puntos["punto_0"] = puntos["_punto_0_idx"].map(indice_a_mes)

    trayectoria = datos.merge(
        puntos[["numero_de_cliente", "_punto_0_idx"]],
        on="numero_de_cliente",
        how="inner",
        validate="many_to_one",
    )
    trayectoria["_tiempo_relativo"] = (
        trayectoria["_mes_idx"] - trayectoria["_punto_0_idx"]
    )
    largo_historial = (
        trayectoria.loc[trayectoria["_tiempo_relativo"] < 0]
        .groupby("numero_de_cliente")["_mes_idx"]
        .nunique()
        .rename("largo_historial")
    )
    trayectoria = trayectoria.loc[
        trayectoria["_tiempo_relativo"].between(-k, -1)
    ]

    valores = trayectoria.pivot(
        index="numero_de_cliente",
        columns="_tiempo_relativo",
        values=columnas_features,
    )
    esperadas = pd.MultiIndex.from_product([
        columnas_features, range(-1, -k - 1, -1)
    ])
    valores = valores.reindex(columns=esperadas)
    valores.columns = [
        f"{feature}_{periodo}" for feature, periodo in valores.columns
    ]
    valores = valores.reindex(puntos["numero_de_cliente"])

    salida = (
        puntos[["numero_de_cliente", "grupo", "punto_0"]]
        .set_index("numero_de_cliente")
        .join(largo_historial)
    )
    for feature in columnas_features:
        meses = valores[[
            f"{feature}_{periodo}"
            for periodo in range(-1, -k - 1, -1)
        ]]
        salida[feature] = meses.iloc[:, 0]

        if feature in columnas_sin_calculos_temporales:
            continue

        meses = meses.apply(pd.to_numeric).astype(float)
        completo = meses.notna().all(axis=1)
        if k < 2:
            salida[f"{feature}_delt_prom"] = np.nan
            salida[f"{feature}_delt_rel"] = np.nan
        else:
            delta = (
                (meses.iloc[:, 0] - meses.iloc[:, -1]) / (k - 1)
            ).where(completo)
            nivel = meses.abs().mean(axis=1, skipna=False)
            relativo = delta / nivel.replace(0, np.nan)
            relativo = relativo.where(nivel.ne(0), 0.0)
            salida[f"{feature}_delt_prom"] = delta
            salida[f"{feature}_delt_rel"] = relativo

        if k < 3:
            salida[f"{feature}_delt_tend"] = np.nan
        else:
            tramos_recientes = (k - 1) // 2
            tramos_antiguos = (k - 1) - tramos_recientes
            medio = meses.iloc[:, tramos_recientes]
            pendiente_reciente = (
                meses.iloc[:, 0] - medio
            ) / tramos_recientes
            pendiente_antigua = (
                medio - meses.iloc[:, -1]
            ) / tramos_antiguos
            salida[f"{feature}_delt_tend"] = (
                pendiente_reciente - pendiente_antigua
            ).where(completo)

    return salida.reset_index()

def dataset_por_clientes_tendencia_no_alineado_con_embargo(
    df,
    columnas_features,
    k,
    columnas_sin_calculos_temporales=(),
    seed=214363,
):
    if k < 1:
        raise ValueError("k debe ser al menos 1.")

    datos = df.copy()
    necesarias = {
        "numero_de_cliente", "foto_mes", "mes_baja",
        *columnas_features,
    }
    faltantes = necesarias - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    if datos.duplicated(["numero_de_cliente", "foto_mes"]).any():
        raise ValueError(
            "Hay más de una fila para algún numero_de_cliente/foto_mes"
        )

    datos["_mes_idx"] = mes_a_indice(datos["foto_mes"])
    datos["_baja_idx"] = mes_a_indice(datos["mes_baja"])

    # Un cliente con mes_baja informado nunca puede ser CONTINUA.
    puntos_baja = (
        datos.loc[
            datos["_baja_idx"].notna(),
            ["numero_de_cliente", "_baja_idx"],
        ]
        .drop_duplicates()
    )
    if puntos_baja["numero_de_cliente"].duplicated().any():
        raise ValueError("Hay clientes con más de un mes_baja")
    ids_baja = set(puntos_baja["numero_de_cliente"])
    puntos_baja = puntos_baja.rename(
        columns={"_baja_idx": "_punto_0_idx"}
    )

    # BAJA debe aparecer en -1. Ese mes no se usa como feature.
    # Además debe tener los k meses de -2 a -(k+1).
    meses_baja = datos[["numero_de_cliente", "_mes_idx"]].merge(
        puntos_baja,
        on="numero_de_cliente",
        how="inner",
        validate="many_to_one",
    )
    diferencia = (
        meses_baja["_mes_idx"] - meses_baja["_punto_0_idx"]
    )
    n_previos = (
        meses_baja.loc[
            diferencia.between(-k - 1, -1)
        ]
        .groupby("numero_de_cliente")
        .size()
    )
    puntos_baja = puntos_baja.loc[
        puntos_baja["numero_de_cliente"]
        .map(n_previos)
        .eq(k + 1)
        .fillna(False)
    ].copy()
    if puntos_baja.empty:
        raise ValueError(
            "No hay BAJA con el mes -1 y los k meses anteriores completos."
        )
    puntos_baja["grupo"] = "BAJA"

    # Los CONTINUA permanecen sin interrupciones desde su ingreso
    # hasta el último mes disponible en el dataset.
    ultimo_mes = int(datos["_mes_idx"].max())
    historia = datos.groupby("numero_de_cliente")["_mes_idx"].agg(
        ["min", "max", "nunique"]
    )
    historia = historia.loc[~historia.index.isin(ids_baja)]
    historia = historia.loc[
        historia["max"].eq(ultimo_mes)
        & historia["nunique"].eq(ultimo_mes - historia["min"] + 1)
    ]

    # Se usan los cortes de las BAJA elegibles para asignar
    # los puntos_0 ficticios de CONTINUA.
    frecuencias = puntos_baja["_punto_0_idx"].value_counts().sort_index()
    meses_corte = frecuencias.index.to_numpy(dtype=int)
    elegibles = []
    for cliente, primer_mes in historia["min"].items():
        posibles = meses_corte[
            meses_corte - (k + 1) >= int(primer_mes)
        ]
        if len(posibles):
            elegibles.append((cliente, posibles))

    rng = np.random.default_rng(seed)
    objetivos = (
        frecuencias / frecuencias.sum() * len(elegibles)
    ).to_dict()
    asignados = {mes: 0 for mes in meses_corte}
    cortes_continua = {}

    # Se asignan primero los clientes con menos cortes posibles.
    orden = rng.permutation(len(elegibles))
    orden = sorted(orden, key=lambda i: len(elegibles[i][1]))
    for i in orden:
        cliente, posibles = elegibles[i]
        deficit = np.array([
            objetivos[mes] - asignados[mes] for mes in posibles
        ])
        mejores = posibles[np.isclose(deficit, deficit.max())]
        mes_elegido = int(rng.choice(mejores))
        cortes_continua[cliente] = mes_elegido
        asignados[mes_elegido] += 1

    puntos_continua = pd.DataFrame({
        "numero_de_cliente": list(cortes_continua),
        "_punto_0_idx": list(cortes_continua.values()),
        "grupo": "CONTINUA",
    })
    puntos = pd.concat(
        [puntos_baja, puntos_continua],
        ignore_index=True,
    )
    puntos["punto_0"] = puntos["_punto_0_idx"].map(indice_a_mes)

    trayectoria = datos.merge(
        puntos[["numero_de_cliente", "_punto_0_idx"]],
        on="numero_de_cliente",
        how="inner",
        validate="many_to_one",
    )
    trayectoria["_tiempo_relativo"] = (
        trayectoria["_mes_idx"] - trayectoria["_punto_0_idx"]
    )

    # largo_historial sigue contando -1.
    largo_historial = (
        trayectoria.loc[trayectoria["_tiempo_relativo"] < 0]
        .groupby("numero_de_cliente")["_mes_idx"]
        .nunique()
        .rename("largo_historial")
    )

    # Solo los k meses de -2 a -(k+1) entran en nivel y deltas.
    trayectoria = trayectoria.loc[
        trayectoria["_tiempo_relativo"].between(-k - 1, -2)
    ]

    valores = trayectoria.pivot(
        index="numero_de_cliente",
        columns="_tiempo_relativo",
        values=columnas_features,
    )
    esperadas = pd.MultiIndex.from_product([
        columnas_features, range(-2, -k - 2, -1)
    ])
    valores = valores.reindex(columns=esperadas)
    valores.columns = [
        f"{feature}_{periodo}" for feature, periodo in valores.columns
    ]
    valores = valores.reindex(puntos["numero_de_cliente"])

    salida = (
        puntos[["numero_de_cliente", "grupo", "punto_0"]]
        .set_index("numero_de_cliente")
        .join(largo_historial)
    )
    for feature in columnas_features:
        meses = valores[[
            f"{feature}_{periodo}"
            for periodo in range(-2, -k - 2, -1)
        ]]
        salida[feature] = meses.iloc[:, 0]

        if feature in columnas_sin_calculos_temporales:
            continue

        meses = meses.apply(pd.to_numeric).astype(float)
        completo = meses.notna().all(axis=1)
        if k < 2:
            salida[f"{feature}_delt_prom"] = np.nan
            salida[f"{feature}_delt_rel"] = np.nan
        else:
            delta = (
                (meses.iloc[:, 0] - meses.iloc[:, -1]) / (k - 1)
            ).where(completo)
            nivel = meses.abs().mean(axis=1, skipna=False)
            relativo = delta / nivel.replace(0, np.nan)
            relativo = relativo.where(nivel.ne(0), 0.0)
            salida[f"{feature}_delt_prom"] = delta
            salida[f"{feature}_delt_rel"] = relativo

        if k < 3:
            salida[f"{feature}_delt_tend"] = np.nan
        else:
            tramos_recientes = (k - 1) // 2
            tramos_antiguos = (k - 1) - tramos_recientes
            medio = meses.iloc[:, tramos_recientes]
            pendiente_reciente = (
                meses.iloc[:, 0] - medio
            ) / tramos_recientes
            pendiente_antigua = (
                medio - meses.iloc[:, -1]
            ) / tramos_antiguos
            salida[f"{feature}_delt_tend"] = (
                pendiente_reciente - pendiente_antigua
            ).where(completo)

    return salida.reset_index()