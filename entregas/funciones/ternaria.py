import cudf
import cupy as cp
import numpy as np
import pandas as pd


def ternaria(df):
    # Índice de mes continuo para cada fila.
    year = df['foto_mes'] // 100
    month = df['foto_mes'] % 100
    df['mes_idx'] = year * 12 + month

    # Tabla de existencia: una fila por cliente y mes.
    presencia = (
        df[['numero_de_cliente', 'mes_idx']]
        .drop_duplicates()
        .assign(presente=cp.int8(1))
    )

    # Presencia en t+1: desplazamos la clave para que coincida con el mes actual t.
    presencia_t1 = presencia.rename(columns={'mes_idx': 'mes_idx_t1', 'presente': 'presente_t1'})
    presencia_t1['mes_idx'] = presencia_t1['mes_idx_t1'] - 1
    presencia_t1 = presencia_t1[['numero_de_cliente', 'mes_idx', 'presente_t1']]

    # Presencia en t+2.
    presencia_t2 = presencia.rename(columns={'mes_idx': 'mes_idx_t2', 'presente': 'presente_t2'})
    presencia_t2['mes_idx'] = presencia_t2['mes_idx_t2'] - 2
    presencia_t2 = presencia_t2[['numero_de_cliente', 'mes_idx', 'presente_t2']]

    df = df.merge(presencia_t1, on=['numero_de_cliente', 'mes_idx'], how='left')
    df = df.merge(presencia_t2, on=['numero_de_cliente', 'mes_idx'], how='left')

    tiene_t1 = df['presente_t1'].fillna(0).astype('bool')
    tiene_t2 = df['presente_t2'].fillna(0).astype('bool')

    # Último mes observado del dataset: no tiene horizonte futuro verificable.
    ultimo_mes_idx = int(df['mes_idx'].max())

    # df['ternaria'] = cudf.Series(None, index=df.index, dtype='str')
    df['ternaria'] = pd.Series(None, index=df.index, dtype='str')
    df.loc[df['mes_idx'] < ultimo_mes_idx - 1, 'ternaria'] = 'BAJA+1'
    df.loc[(df['mes_idx'] < ultimo_mes_idx - 1) & tiene_t1, 'ternaria'] = 'BAJA+2'
    df.loc[(df['mes_idx'] < ultimo_mes_idx - 1) & tiene_t1 & tiene_t2, 'ternaria'] = 'continua'

    # Penúltimo mes: sólo puede determinarse baja+1; si continúa queda sin clase.
    df.loc[(df['mes_idx'] == ultimo_mes_idx - 1) & ~tiene_t1, 'ternaria'] = 'BAJA+1'

    # Quitamos columnas auxiliares.
    df = df.drop(columns=['mes_idx', 'presente_t1', 'presente_t2'])
    # Nulos en vez de NAN en CUDF
    df.loc[df["ternaria"] == "", "ternaria"] = None

    return df

def rankear_por_mes(
    df,
    columnas_excluidas,
    escala="comun",
    empates="dense"
):
    """
    Reemplaza las variables por rankings calculados por columna y foto_mes.

    df: dataframe de entrada, con registros cliente/mes.
    columnas_excluidas: columnas que permanecen sin modificaciones.
    escala:
        "comun": mismo paso para positivos y negativos; máximo absoluto = 1.
        "separada": cada signo se normaliza por separado hasta +1 o -1.
    empates:
        "dense": valores iguales comparten rango; el siguiente es consecutivo.
        "average": valores iguales reciben el promedio de sus posiciones.

    Conserva los ceros como 0, los NaN como NaN y el signo de los demás valores.
    Un grupo constante no nulo queda en +1 o -1 según su signo.
    Devuelve una copia, conservando columnas, índice y orden de las filas.
    """
    df = df.copy()
    meses = df["foto_mes"].copy()

    columnas = [
        columna
        for columna in df.columns
        if columna not in columnas_excluidas
    ]

    for columna in columnas:
        valores = df[columna]

        ranking_positivo = (
            valores.where(valores > 0)
            .groupby(meses, sort=False)
            .rank(method=empates)
        )
        ranking_negativo = (
            (-valores.where(valores < 0))
            .groupby(meses, sort=False)
            .rank(method=empates)
        )

        maximo_positivo = (
            ranking_positivo.groupby(meses, sort=False).transform("max")
        )
        maximo_negativo = (
            ranking_negativo.groupby(meses, sort=False).transform("max")
        )

        if escala == "comun":
            maximo = pd.concat(
                [maximo_positivo, maximo_negativo], axis=1
            ).max(axis=1)
            maximo_positivo = maximo
            maximo_negativo = maximo
        elif escala != "separada":
            raise ValueError('escala debe ser "comun" o "separada"')

        ranking = (ranking_positivo / maximo_positivo).fillna(
            -ranking_negativo / maximo_negativo
        )
        df[columna] = ranking.mask(valores.eq(0).fillna(False), 0.0)

    return df