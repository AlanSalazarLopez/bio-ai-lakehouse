# Schema Design — Bio-AI Lakehouse
**Proyecto:** Delta Lakehouse · GTEx Gene Expression  
**Fecha:** 2025-01-01  
**Estado:** Paso 6 completado — schemas aprobados, listos para implementación

---

## Flujo de Linaje

```
GTEx raw (.parquet)
  └─[shutil.copy2]──────────► Bronze  (wide,  74,628 × 19,790 cols)
       └─[reshape + join]────► Silver  (long,  ~1.47B filas)
            └─[agg + log1p]──► Gold    (agg,   ~4M filas)
```

El consumidor final de Gold es el clasificador de patogenicidad del Proyecto C.  
Gold debe entregar features pre-calculados, sin joins pendientes para el modelo.

---

## Bronze — `gene_tpm_raw.parquet`

**Formato:** Parquet wide · Sin transformación · Copia inmutable del raw  
**Ruta:** `data/bronze/gtex/gene_tpm_raw.parquet`  
**Compresión:** Snappy  
**Partición:** ninguna  

| Columna | Tipo | Nullable | Origen |
|---|---|---|---|
| `Name` | STRING | NO | **Índice del parquet** — Ensembl ID (ej. `ENSG00000290825.2`). Key única. En Pandas: `df.index`. En Spark: se materializa como columna normal al leer. |
| `Description` | STRING | NO | Columna regular — gene symbol (ej. `DDX11L16`). 1,307 duplicados esperados y válidos — múltiples Ensembl IDs pueden compartir símbolo. |
| `GTEX-XXXX-...` × 19,788 | FLOAT | NO | TPM por muestra — una columna por sample ID |

**Dimensiones confirmadas:** 74,628 filas × 19,790 columnas  
**Nulls:** 0.0% (confirmado en profiling)  
**Zeros:** 51.89% — biológicamente válidos, preservar  

### Quality Gate Bronze (Paso 5) — APROBADO ✅

| Check | Criterio | Resultado | Threshold de fallo |
|---|---|---|---|
| Row count | == 74,628 | ✅ 74,628 | cualquier diferencia → STOP |
| Duplicados `Name` (índice) | == 0 | ✅ 0 | cualquier duplicado → STOP |
| Duplicados `Description` | 1,307 esperados | ✅ válidos biológicamente | no aplica — key es `Name` |
| Nulls en índice `Name` | == 0% | ✅ 0% | > 0% → STOP |
| Nulls en `Description` | == 0% | ✅ 0% | > 0% → STOP |
| Tipo sample cols | FLOAT | ✅ confirmado | tipo incorrecto → STOP |

> Threshold global: cualquier fallo en la key `Name` (índice) detiene el pipeline.  
> Duplicados en `Description` son biológicamente esperados — no son un fallo.

---

## Silver — `gene_expression_long` (Delta Lake)

**Formato:** Delta Lake · Long format · Particionado por tejido  
**Ruta:** `data/silver/gtex/gene_expression_long/`  
**Compresión:** Snappy  
**Partición:** `tissue_id`  
**Write mode:** overwrite (idempotente)  

| Columna | Tipo | Nullable | Origen |
|---|---|---|---|
| `gene_id` | STRING | **NO** | `Name` de Bronze |
| `gene_symbol` | STRING | **NO** | `Description` de Bronze |
| `sample_id` | STRING | **NO** | nombre de columna Bronze |
| `tpm_value` | FLOAT | **NO** | valor de celda Bronze |
| `tissue_id` | STRING | **NO** | join `sample_id → SMTSD` vía `gtex_metadata.txt` |

**Filas esperadas:** ~1,477,040,064 (74,628 × 19,788)  
**Zeros:** preservados — `tpm_value = 0.0` es dato válido en Silver  
**Tejidos únicos:** ~54 (define número de particiones)  

### Transformaciones ordenadas por costo

```
1. Leer Bronze en chunks de 200 cols          [sin shuffle · Pandas · bajo costo]
2. Reshape wide→long por chunk                [sin shuffle · Pandas · bajo costo]
3. Cast tipos explícitos                      [sin shuffle · bajo costo]
4. Join sample_id → tissue_id                 [SHUFFLE · el más costoso]
5. Separar unmatched → cuarentena             [filter · sin shuffle · bajo costo]
6. Write Delta particionado por tissue_id     [shuffle por partición · alto costo]
```

> Regla aplicada: shuffle al final siempre que sea posible.  
> El join opera sobre 4 columnas, no sobre las 19,790 del Bronze original.

### Quarantine Protocol Bronze → Silver

**Trigger:** `sample_id` sin match en `gtex_metadata.txt`  
**Ruta:** `data/quarantine/silver_unmatched_samples.parquet`  

| Columna | Tipo | Descripción |
|---|---|---|
| `gene_id` | STRING | de Bronze |
| `gene_symbol` | STRING | de Bronze |
| `sample_id` | STRING | el sample sin match |
| `tpm_value` | FLOAT | valor original |
| `reason` | STRING | `"no_tissue_match"` |

**Verificación de cierre:**
```
Silver rows + Quarantine rows == 74,628 × 19,788
```
Si este número no cierra → pipeline STOP. No se escribe Gold.

### Strategy chunks (Pandas pre-Spark)

| Parámetro | Valor | Fuente |
|---|---|---|
| Columnas por chunk | 200 | profiling_report.json |
| Iteraciones | 99 | ceil(19,788 / 200) |
| RAM segura Spark | 3g | chunk_calculator (override 4g rechazado) |
| Particiones Spark | calculadas por chunk_calculator en runtime |

---

## Gold — `gene_tissue_summary` (Delta Lake)

**Formato:** Delta Lake · Agregado por gen × tejido  
**Ruta:** `data/gold/gtex/gene_tissue_summary/`  
**Compresión:** ZSTD  
**Partición:** `tissue_id`  
**Write mode:** overwrite (idempotente)  

| Columna | Tipo | Nullable | Origen |
|---|---|---|---|
| `gene_id` | STRING | **NO** | Silver |
| `gene_symbol` | STRING | **NO** | Silver — disponible directo, sin join extra |
| `tissue_id` | STRING | **NO** | Silver (partición) |
| `mean_log1p_tpm` | FLOAT | NO | `mean(log1p(tpm_value))` |
| `median_log1p_tpm` | FLOAT | NO | `percentile_approx(tpm_value, 0.5)` post log1p |
| `std_log1p_tpm` | FLOAT | NO | `stddev(log1p(tpm_value))` |
| `sample_count` | INT | NO | `count(sample_id)` |
| `zero_fraction` | FLOAT | NO | `sum(tpm_value == 0) / count(sample_id)` |

**Filas esperadas:** ~4,029,912 (74,628 genes × ~54 tejidos)  
**Transformación log1p:** aplicada aquí, no en Silver  
**Zeros en Gold:** contribuyen a `zero_fraction`, excluidos del cálculo de log1p media  

### Transformaciones ordenadas por costo

```
1. Leer Silver (Delta, particionado)            [sin shuffle · bajo costo]
2. Calcular log1p(tpm_value)                   [sin shuffle · bajo costo]
3. groupBy(gene_id, gene_symbol, tissue_id)    [SHUFFLE · el más costoso]
4. agg: mean / stddev de log1p                 [compute · medio costo]
5. agg: percentile_approx para median          [compute · medio costo]
6. agg: count(sample_id) → sample_count        [bajo costo]
7. agg: zero_fraction                          [bajo costo]
8. Write Delta ZSTD particionado tissue_id     [shuffle por partición]
```

### Skew handling

| Tejido | Muestras | Factor skew vs mínimo |
|---|---|---|
| Whole Blood | 4,369 | 2,184x |
| Liver - Portal Tract | 2 | 1x (mínimo) |

**Estrategia:** Spark 3.5 AQE (Adaptive Query Execution) activo por default.  
No se requiere solución manual — AQE detecta el skew en el groupBy y ajusta las  
particiones automáticamente en runtime.

**Verificar en sesión Spark:**
```python
spark.conf.get("spark.sql.adaptive.enabled")  # debe ser "true"
```

### Diseño hacia el consumidor (Proyecto C)

El modelo de patogenicidad del Proyecto C necesita:

| Necesidad del modelo | Cómo Gold la satisface |
|---|---|
| Features numéricos sin joins | Todos los campos son calculados — sin FK pendientes |
| Contexto de expresión por tejido | `mean_log1p_tpm`, `std_log1p_tpm` por `tissue_id` |
| Signal de genes inactivos | `zero_fraction` — feature biológico directo |
| Lookup de símbolo del gen | `gene_symbol` incluido — sin join a tabla de dimensiones |
| Queries frecuentes por tejido | Particionado por `tissue_id` — scan mínimo |

---

## Decisiones de diseño registradas

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| `gene_symbol` en Silver | dejarlo solo en Gold | disponible en el momento del reshape sin costo extra |
| Zeros preservados en Silver | filtrar zeros | biológicamente válidos — gen inactivo ≠ error |
| `log1p` solo en Gold | aplicar en Silver | Silver es transformación pura, no análisis |
| Quarantine en archivo separado | solo loguear y descartar | el lineage debe cerrar matemáticamente |
| AQE para skew de Whole Blood | salting manual | Spark 3.5 lo maneja automáticamente |
| Chunks de 200 cols en Pandas | Spark para el reshape | Bronze tiene 19,790 cols — OOM confirmado con Spark directo |

---

*Generado en Paso 6 del Data Engineering Decision Framework*  
*Siguiente paso: `src/jobs/silver_transform.py`*