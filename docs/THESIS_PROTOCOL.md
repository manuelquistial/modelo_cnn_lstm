# Protocolo de implementación — Das et al. (2025)

Documento de decisiones metodológicas para tesis y réplica del artículo.

## Modos de clasificación

| Modo | Clases | Runs | Uso |
|------|--------|------|-----|
| **A — multiclass** | E,F,G,H,I | 4,6,8,10,12,14 | Tablas 6/7 (principal) |
| **B — binary** | left_hand vs right_hand | 4,8,12 | Tabla 8, Fig. 8 t-SNE |

Activar Modo A + B: `PAPER=1` o `--paper-protocol`.

## Split train/validation/test

**Principal (tesis):** `subjectwise_3way` — 70/15/15 por sujeto (`GroupShuffleSplit`).

- Sin fuga inter-sujeto
- GAN entrenado **solo** en partición train
- Test = solo datos reales

**No usar como métrica principal:** `trialwise` (infla accuracy).

**Opcional futuro:** `GroupKFold` por sujeto (`make_group_kfold_splits`).

## Preprocesamiento (paper)

1. Band-pass 0.5–50 Hz  
2. ICA (artefactos)  
3. ROI → CSP (fit train) → 640×2  
4. Z-score (estadísticas train)  

## Modelos comparados

- ML clásico (+ CSP features)  
- Riemannian baseline (separado, no dentro del CNN-LSTM)  
- CNN, LSTM, CNN-LSTM-Attention  
- CNN-LSTM sin GAN vs con WGAN-GP (`gan_ablation_results.csv`)  

## Arquitectura CNN-LSTM

Compatible con Fig. 6 / Tabla 5 de Das et al.; hiperparámetros conv no publicados → ver `models_deep.py`.

## Referencia al 96.06%

Tratar como *test accuracy reportado* en Tabla 6, **no** como evidencia de generalización cross-subject.
