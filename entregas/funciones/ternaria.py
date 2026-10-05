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