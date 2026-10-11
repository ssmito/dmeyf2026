# Entrega 1

## 1. ENTREGA

Ejecutar `entregas/p005_3models_runner.ipynb` desde `entregas/`. Necesita los datasets generados y los `.rds` originales con los hiperparámetros de los modelos, ya copiados a `entregas/parametros/`.

Correr toda la notebook como está, en orden, hasta el punto anterior a **Documentación**. Entrena los 3 modelos con 50 semillas, combina las predicciones y guarda el CSV de entrega en `entregas/entregas/`.

**carga 3 datasets en memoria, local con 64 gb de ram no tuve OOM**


## 2. DATASETS

Punto de partida: colocar `competencia_01_crudo.csv` en `entregas/dataset/`, junto a las dos notebooks.

Desde `entregas/dataset/`, ejecutar completas, como están:

1. `versiones_dataset_1.ipynb`: genera los datasets necesarios para la cadena de LAG-009.
2. `versiones_dataset_2.ipynb`: genera los demás datasets utilizados en la entrega.

**Sí, soy consciente q en LAG-009 los HP son encontrados para un dataset y en las predicciones los uso con 2 datasets diferentes, pero con varias BO no pude mejorar los resultados de esos mismos HP incluso para los otros datasets, y los resultados mandan**

## 3. BO

Ejecutar los tres pasos desde `entregas/bo`, con kernel Python para las BO y kernel R para las validaciones.

1. **Replicar las BO:** `lag009_replica.ipynb` y `lag028_replica.ipynb` toman los parámetros, rangos y puntos iniciales de las carpetas originales `LAG-009/` y `LAG-028/` y reproducen la optimización en `LAG-009-REPLICA/` y `LAG-028-REPLICA/`.
2. **Replicar el benchmark multisemilla:** `validacion_multisemilla_lag009_replica.ipynb` y `validacion_multisemilla_lag028_replica.ipynb` toman la configuración de los benchmarks originales, evalúan los modelos de las BO replicadas con varias semillas y seleccionan sus mejores cortes.
3. **Replicar los `.rds`:** `validacion_manual_lag009_replica.ipynb` y `validacion_manual_lag028_replica.ipynb` toman el modelo elegido de cada benchmark replicado y generan `40-REPLICA.rds` y `62-REPLICA.rds`. Sus referencias originales son `entregas/parametros/40.rds` y `entregas/parametros/62.rds`, utilizados para la entrega.

Las carpetas de benchmark originales contienen los resultados de referencia; las réplicas generan sus salidas en las carpetas con sufijo `-REPLICA`:

| BO | Benchmark original en `entregas/bo/` | Benchmark replicado en `entregas/bo/` |
| --- | --- | --- |
| LAG-009 | `benchmark_multisemilla_lags20260924_114539/` | `benchmark_multisemilla_lags20260924_114539-REPLICA/` |
| LAG-028 | `benchmark_multisemilla_lags20261009_200322/` | `benchmark_multisemilla_lags20261009_200322-REPLICA/` |

Cada benchmark replicado guarda `PARAM.rds` y `mejor_corte_por_modelo.tsv`, los dos archivos necesarios para crear el `.rds` de parámetros.
