# Ejecución en Paperspace (Gradient)

Repositorio: https://github.com/manuelquistial/modelo_cnn_lstm

> **En Paperspace no uses SSH** (`git@github.com:...`) salvo que hayas añadido una clave pública a GitHub.  
> Usa **HTTPS** (abajo). Si el repo es privado, usa un [Personal Access Token](https://github.com/settings/tokens).

## 1. Crear máquina

- **Gradient** → Notebook o Machine con **GPU** (p. ej. A4000 / A5000).
- Imagen: **Ubuntu 22.04** + **CUDA** (template con GPU).
- Python **3.10** o **3.11** (evitar 3.14; TensorFlow aún no es estable ahí).

## 2. Clonar e instalar

```bash
# Clonar por HTTPS (recomendado en Paperspace)
git clone https://github.com/manuelquistial/modelo_cnn_lstm.git
cd modelo_cnn_lstm
chmod +x scripts/*.sh
./scripts/paperspace_setup.sh
```

Si el repositorio es **privado**:

```bash
git clone https://<TU_TOKEN>@github.com/manuelquistial/modelo_cnn_lstm.git
# o: git clone https://github.com/manuelquistial/modelo_cnn_lstm.git
#     usuario: tu GitHub login | contraseña: el token (no tu password)
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
chmod +x scripts/check_gpu.sh
./scripts/check_gpu.sh
```

O manualmente:

```bash
source .venv/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si ves `[PhysicalDevice(name='/physical_device:GPU:0', ...)]`, **TensorFlow usará la GPU** al entrenar (CNN/LSTM/GAN). Los mensajes `cpu_feature_guard` al importar TensorFlow son **normales**: indican optimizaciones CPU para operaciones que aún corren en CPU; no significan que la GPU esté desactivada.

Durante el entrenamiento, en otra terminal:

```bash
watch -n 2 nvidia-smi
```

Deberías ver uso de memoria GPU y proceso `python` cuando empiece `model.fit`.

## Stack

- **TensorFlow / Keras** — CNN, LSTM, híbrido, GAN
- **MNE** — PhysioNet EEG MI
- **scikit-learn** — ML clásico
