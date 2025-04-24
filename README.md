# Control de Programas EEG

## Secuencia de Ejecución Recomendada

1. `InputGUI.py` → **Out:** `AURA_Power` stream  
2. `realtime_processor.py` → **In:** `AURA_Power` ● **Out:** `FEATURE_STREAM`  
3. `cnn_interference.py` → **In:** `FEATURE_STREAM` ● **Out:** `CNN_COMMANDS`  
4. `CommandSender.py` → **In:** `CNN_COMMANDS`

> **Nota:** Esperar unos segundos entre cada inicio para permitir que los streams LSL se establezcan.

---

## Descripción del Proyecto

Este proyecto implementa un sistema completo de adquisición, procesamiento y clasificación de señales EEG en tiempo real, con las siguientes etapas:

1. **InputGUI.py**: Interfaz gráfica y captura de datos EEG desde el dispositivo AURA (LSL).  
2. **realtime_processor.py**: Procesa la señal EEG cruda y extrae características (bandpower).  
3. **cnn_interference.py**: Aplicación de un modelo CNN multiclase para inferencia de comandos a partir de las características.  
4. **CommandSender.py**: Envía los comandos inferidos a la aplicación cliente o dispositivo final.

## Estructura de Carpetas

```
project_root/
├── OrganizedData/         # Datos organizados por clase (entrenamiento)
│   ├── LeftArmThinking/   # Carpeta con CSVs para la clase LeftArmThinking
│   ├── RightArmThinking/
│   ├── LeftFistThinking/
│   └── RightFistThinking/
├── Results/               # Salida del entrenamiento y validación
│   ├── confusion_matrix.png
│   └── Results.txt
├── cnn_interference.py    # Código de entrenamiento/inferencia del CNN
├── realtime_processor.py  # Procesamiento de señales EEG
├── InputGUI.py            # Captura GUI y publicación LSL
├── CommandSender.py       # Envío de comandos inferidos
├── PredictEEG.py          # Script para predicciones con el modelo entrenado
├── environment.yml        # Dependencias de Conda (o requirements.txt)
└── README.md              # Documentación de uso
```

---

## Requisitos de Instalación

Se recomienda crear un entorno virtual (venv, Conda) e instalar las dependencias:

```bash
pip install -r requirements.txt
```

Contenido mínimo de `requirements.txt`:
```
umpy
pandas
scipy
matplotlib
scikit-learn
torch
pylsl
pykalman
joblib
```

---

## Entrenamiento del Modelo CNN (`cnn_interference.py`)

1. Asegúrate de tener la carpeta `OrganizedData/` con subcarpetas nombradas según la etiqueta de cada clase.
2. Ejecuta:
   ```bash
   python cnn_interference.py --train
   ```
3. El script leerá los CSV, entrenará el modelo, guardará la matriz de confusión en `Results/confusion_matrix.png` y un informe en `Results/Results.txt`.

**Formato de `Results/Results.txt`**:
```
Ensemble creado: YYYY-MM-DD HH:MM:SS
Accuracy global: 0.97XX
Accuracy por clase:
  - LeftArmThinking: 0.95XX
  - RightArmThinking: 0.95XX
  - LeftFistThinking: 0.98XX
  - RightFistThinking: 1.0000

Modelos incluidos:
  - Modelo 1: 0.84XX
  - Modelo 2: 0.82XX
  - Modelo 3: 0.81XX
```

---

## Predicción en Tiempo Real (`PredictEEG.py`)

Una vez entrenado el modelo, puedes usar `PredictEEG.py` para clasificar nuevos CSV generados por `realtime_processor.py`:

```bash
python PredictEEG.py \
  --model models/eeg_model_best.pth \
  --model-info models/model_info.json \
  --scaler models/eeg_scaler.pkl \
  --encoder models/eeg_label_encoder.pkl \
  --csv data/<participant>/<sample>.csv
```

El script imprimirá la etiqueta más probable y las probabilidades de cada clase.

---

## Flujo Completo

1. Inicia `InputGUI.py` y espera ~2s.  
2. Inicia `realtime_processor.py` y espera ~2s.  
3. Inicia `cnn_interference.py` (inicia entrenamiento o inferencia).  
4. Inicia `CommandSender.py` para recibir y ejecutar comandos.

Para comodidad, puedes usar la GUI de control incluida en `ControlPanel.py` que lanza y detiene automáticamente en secuencia.

---

## Licencia

Este proyecto está bajo licencia MIT.")

