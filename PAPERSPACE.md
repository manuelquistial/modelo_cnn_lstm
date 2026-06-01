# Ejecución en Paperspace (Gradient)

Repositorio: `git@github.com:manuelquistial/modelo_cnn_lstm.git`

## 1. Crear máquina

- **Gradient** → Notebook o Machine con **GPU** (p. ej. A4000 / A5000).
- Imagen: **Ubuntu 22.04** + **CUDA** (template con GPU).
- Python **3.10** o **3.11** (evitar 3.14; TensorFlow aún no es estable ahí).

## 2. Clonar e instalar

```bash
git clone git@github.com:manuelquistial/modelo_cnn_lstm.git
cd modelo_cnn_lstm
chmod +x scripts/*.sh
./scripts/paperspace_setup.sh
```

## 3. Prueba rápida (5 sujetos, ~30–60 min)

```bash
QUICK=1 ./scripts/paperspace_run.sh
```

## 4. Réplica cercana al paper (multiclass, trial-wise, entrada 640×2)

```bash
PAPER=1 EPOCHS=50 ./scripts/paperspace_run.sh
```

## 5. Experimento completo (103 sujetos — varias horas / días)

```bash
EPOCHS=50 SPLIT=subjectwise ./scripts/paperspace_run.sh
```

Con GAN (muy lento):

```bash
./scripts/paperspace_run.sh --gan
```

## Variables útiles

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODE` | `binary` | `binary` o `multiclass` |
| `SPLIT` | `subjectwise` | `subjectwise` o `trialwise` |
| `EPOCHS` | `50` | Épocas de entrenamiento DL |
| `MAX_SUBJECTS` | (vacío) | Limitar sujetos p. ej. `20` |
| `OUTPUT_DIR` | `outputs/das2025_replication` | Salida CSV y figuras |
| `MNE_DATA` | `./mne_data` | Cache descarga PhysioNet |
| `QUICK` | `0` | `1` = smoke test |
| `PAPER` | `0` | `1` = flags paper-input + trialwise |

## Persistencia en Paperspace

Monta un **Persistent Storage** y apunta salidas y datos ahí:

```bash
export OUTPUT_DIR=/notebooks/persistent/outputs/das2025_replication
export MNE_DATA=/notebooks/persistent/mne_data
./scripts/paperspace_run.sh
```

## Verificar GPU

```bash
source .venv/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Stack

- **TensorFlow / Keras** — CNN, LSTM, híbrido, GAN
- **MNE** — PhysioNet EEG MI
- **scikit-learn** — ML clásico
