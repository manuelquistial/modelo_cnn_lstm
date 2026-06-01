# Auditoría: implementación vs Das et al. (2025)

Comparación entre el artículo (*Scientific Reports*, 2025, DOI 10.1038/s41598-025-07427-2) y el código en `das2025_replication/`.

## Resumen ejecutivo

| Área | Estado | Notas |
|------|--------|-------|
| Dataset PhysioNet v1.0.0, 160 Hz, 64 ch | ✅ | Correcto |
| Exclusión S038, S088, S089, S092, S100, S104 → 103 sujetos | ✅ | Correcto |
| Runs MI 4,6,8,10,12,14 | ✅ | Correcto |
| 6 ROIs (Tabla 3) | ✅ | ROI_5 sin duplicar canales (el paper repite CP1–CP4) |
| Band-pass 0.5–50 Hz | ✅ | En epoching |
| Normalización z-score | ✅ | Post-split en train |
| 5 clasificadores ML + NB | ⚠️ | Falta **Decision Tree** (Fig. 10 del paper) |
| CNN, LSTM, híbrido CNN+LSTM+Attention | ⚠️ | Arquitectura CNN no detallada en Tabla 5; nuestra es plausible |
| Métricas Acc, Prec, Rec, F1, MCC | ✅ | + Cohen κ extra (no en paper) |
| GAN WGAN-GP, 100 épocas, batch 64 | ⚠️ | Implementado; **FID** no implementado |
| Wavelet + Riemannian + PCA + t-SNE | ⚠️ | Parcial; t-SNE/PCA solo visualización |
| CSP + ICA en preprocesado | ⚠️ | CSP opcional ML; ICA desactivado por defecto |
| **Entrada DL 640×2 (par contralateral)** | ❌→✅ | **Gap crítico** — ver `paper_input.py` |
| Tarea binaria mano izq./der. (Resultados) | ✅ | `mode="binary"` |
| Tarea 5 clases E,F,G,H,I (Tablas 6–7) | ⚠️ | `mode="multiclass"`; etiquetas no renombradas E–I |
| Split train/test | ❓ | **No especificado en el paper**; nosotros: trial-wise + subject-wise |
| Épocas por ROI (Tabla 6) | ❌→✅ | 35,42,47,41,50,29 — añadido en config |
| Tabla 8: 25/50 épocas × 1s/5s | ✅ | `run_segment_length_experiment` |

---

## 1. Dataset (Sección 3 — Dataset description)

**Paper:** PhysioNet EEG Motor Movement/Imagery v1.0.0; BCI2000; 64 canales 10–10; 160 Hz; 109 sujetos → excluyen 6 → **103 sujetos**; runs 4,6,8,10,12,14 para imagery; T0/T1/T2; épocas de **5 s**; cinco clases E,F,G,H,I.

**Código:** Coincide en sujetos, runs, filtrado y mapeo de eventos.  
**Inconsistencia del paper:** menciona matriz **640×2** (640/160 = **4 s**) mientras dice épocas de 5 s (800 muestras). El modo `paper_input` usa **4 s → 640 muestras** para alinear con la matriz declarada.

---

## 2. ROIs (Tabla 3, Fig. 4)

**Paper:** 6 ROIs con listas de canales; Tabla 3 titulada “channel **pairs**”; ROI_5 lista CP1–CP4 duplicados.

**Código:** `ROI_DEFINITIONS` correctas; ROI_5 con 8 canales únicos.  
**Gap:** el paper alimenta el modelo con **un par contralateral** (2 columnas), no con los 6–18 canales apilados. Implementación nueva: `ROI_PRIMARY_CONTRALATERAL_PAIR` + `to_paper_input_shape()`.

---

## 3. Preprocesado (Sección Pre-processing)

| Paso | Paper | Código |
|------|-------|--------|
| Normalización (μ, σ) | Eq. (1) | ✅ channel-wise / StandardScaler |
| Band-pass 0.5–50 Hz | Sí | ✅ |
| Filtrado espacial CSP | Eq. (3) | ⚠️ solo ML opcional |
| ICA artefactos | Eq. (4) | ⚠️ opcional, off por defecto |
| WT + Riemannian | Sí | ✅ wavelet + pyriemann |
| PCA + t-SNE | Reducción / visualización | ✅ visualización |
| GAN augmentación | WGAN-GP, 100 ep, batch 64, FID | ⚠️ sin FID |

---

## 4. Features ML (Tabla 4, Fig. 5)

**Paper:** Mean, Median, Variance, Std_dev (tiempo); Mean_freq, Median_freq, Variance_freq, Std_dev_freq (frecuencia).

**Código:** Incluye más features (min, max, skew, kurtosis, bandas δ–γ). Nombres alineados en `features.py` con alias `variance_freq` / `std_dev_freq`.

**No implementado del related work:** AR, STFT, DWT como pipeline principal (el paper dice que *su* enfoque usa WT, no STFT/DWT obligatorio).

---

## 5. Clasificadores ML (Sección Classification, Fig. 10)

**Paper:** KNN, SVC, RF, LR, NB + **Decision Tree** en Fig. 10; RF ~91%.

**Código:** 5 modelos; **falta Decision Tree** → añadido en actualización.

---

## 6. Modelos deep (Tabla 5, Fig. 6)

**Paper (Tabla 5 — capas visibles):** Flatten → LSTM 64 → LSTM 64 → **Attention** → Dense 128 ReLU; CNN previo en Fig. 6 (sin hiperparámetros numéricos en texto).

**Código:** CNN 32/64/128 + pooling + LSTM 64×2 + atención + Dense 128 `embedding_dense`.  
**LSTM standalone 16.13% en paper:** arquitectura del LSTM solo no está especificada; nuestra LSTM es razonable pero puede no reproducir ese valor.

**Atención:** paper menciona capa de atención (L10); ✅ implementada.

---

## 7. Tarea y resultados reportados

| Experimento | Paper | Código |
|-------------|-------|--------|
| Binario izq./der. (Sec. Experimental results) | Sí | `mode="binary"`, runs 4,8,12 |
| 5 clases E–I (Tablas 6–7, ROI) | Sí | `mode="multiclass"` |
| ROI 6 → 96.06% accuracy | Tabla 6 | `run_roi_experiments` |
| Tabla 8: 25/50 ep × 1s/5s | Sí | `run_segment_length_experiment` |
| Riemannian izq./der. (Fig. 9) | Binario | `run_riemannian_baselines` |
| GAN +10% accuracy | Cualitativo | `run_gan` opcional |

**Importante:** el **96.06%** del paper es con **5 clases promediadas** en test (Tabla 6, ROI 6), no necesariamente solo binario. La narrativa de resultados mezcla binario (manos) y 5 clases.

---

## 8. Split train / test

**Paper:** no documenta split por sujeto ni aleatorio por trials. Alta accuracy sugiere **split por trials** (posible leakage inter-sujeto).

**Código:** Por defecto `subjectwise` (más riguroso para tesis). Usar `split_strategy="trialwise"` para acercarse al paper.

---

## 9. Entorno

**Paper:** MacBook M1 8GB, Python 3.7, Google Colab.  
**Código:** agnóstico; no replica Colab.

---

## Cómo ejecutar modo más fiel al paper

```python
from das2025_replication.run_experiments import run_complete_das2025_replication

run_complete_das2025_replication(
    mode="multiclass",           # Tablas 6–7 (5 clases)
    split_strategy="trialwise",  # más parecido al paper (no documentado)
    paper_input=True,            # forma (640, 2) par contralateral
    segment_length=4.0,          # 640 muestras @ 160 Hz
    dl_epochs=50,
)
```

Para binario mano izquierda/derecha (texto resultados + Fig. 8–9):

```python
run_complete_das2025_replication(
    mode="binary",
    split_strategy="trialwise",
    paper_input=True,
    segment_length=5.0,  # Tabla 8 usa 5s para pico 96.04%
)
```

---

## Conclusión

La implementación cubre **~75–85%** del pipeline declarado. Los gaps que impiden replicar **exactamente** el 96.06% son:

1. Formato de entrada **640×2** (par contralateral) vs tensores multicanal.
2. Split no especificado en el paper (usar `trialwise` para comparar).
3. Arquitectura CNN exacta de Fig. 6 no publicada en números.
4. Decision Tree, FID-GAN, ICA/CSP en pipeline completo DL.
5. Inconsistencias internas del paper (5 s vs 640 muestras; 5 clases vs binario).

Para la tesis: reportar resultados **subject-wise** como evaluación rigurosa y **trial-wise + paper_input** como “réplica metodológica según Das et al.”.
