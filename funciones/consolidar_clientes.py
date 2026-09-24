import pandas as pd
import numpy as np
import re


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