#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cnn_interferance.py
Módulo de inferencia en tiempo real:
  - Busca el .pt con mayor accuracy en ./CNNModels (y subcarpetas)
  - Carga ese modelo entrenado
  - Escucha FEATURE_STREAM (LSL) y por cada muestra devuelve el nombre de la clase
    predicha a través de CNN_COMMANDS (LSL)
"""

import os
import re
import time
import logging
from pathlib import Path

import numpy as np
import torch
from pylsl import StreamInlet, StreamOutlet, resolve_byprop

# — Configuración de logging —
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("cnn_interferance_log.txt"),
        logging.StreamHandler()
    ]
)

# — Función para encontrar el .pt con mayor accuracy según su nombre —
def find_best_model(models_root: Path) -> Path:
    best_acc = -1.0
    best_path = None
    for pt in models_root.rglob("*.pt"):
        # extraigo solo números con punto (float) del nombre
        nums = re.findall(r"(\d+\.\d+)", pt.name)
        # filtro para quedarme con los que llevan decimal
        floats = [float(n) for n in nums if "." in n]
        if not floats:
            continue
        acc = max(floats)
        if acc > best_acc:
            best_acc = acc
            best_path = pt
    if best_path is None:
        raise FileNotFoundError(f"No encontré ningún .pt bajo {models_root}")
    logging.info(f"Modelo seleccionado: {best_path.name}  (accuracy={best_acc:.4f})")
    return best_path

# — Importar la clase del modelo desde tu script de entrenamiento —
# Asegúrate de que CNN_EEG.py esté junto a este archivo
from CNN_EEG import EEGCNNMulticlass

def main():
    # 1) localizo carpeta de modelos
    base = Path(__file__).parent
    models_dir = base / "CNNModels"

    # 2) encuentro el .pt con mejor accuracy
    best_model_path = find_best_model(models_dir)

    # 3) cargo el checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(best_model_path, map_location=device)
    config = checkpoint["config"]
    class_names = checkpoint["class_names"]
    in_channels = checkpoint["n_features"]
    seq_len     = checkpoint["seq_len"]

    # 4) instancio y cargo el modelo
    model = EEGCNNMulticlass(in_channels, len(class_names), config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logging.info(f"Modelo cargado y listo en {device}")

    # 5) preparo LSL: inlet FEATURE_STREAM → outlet CNN_COMMANDS
    streams = resolve_byprop("name", "FEATURE_STREAM", timeout=5)
    if not streams:
        logging.error("No encontré ningún stream llamado FEATURE_STREAM")
        return
    inlet = StreamInlet(streams[0])

    info = StreamOutlet(
        StreamOutlet=StreamOutlet(
            StreamOutlet=StreamOutlet  # pequeño hack para evitar línea muy larga
        )
    )  # arreglo de líneas para no cortar
    # pero en realidad:
    # info = StreamInfo("CNN_COMMANDS", "string", 1, 0, "string", "cnn_cmd_001")
    # outlet = StreamOutlet(info)
    # así que lo definimos correctamente:
    from pylsl import StreamInfo, StreamOutlet as _Outlet
    info = StreamInfo("CNN_COMMANDS", "string", 1, 0, "string", "cnn_cmd_001")
    outlet = _Outlet(info)

    logging.info("Escuchando FEATURE_STREAM y enviando a CNN_COMMANDS...")

    # 6) bucle principal
    try:
        while True:
            sample, _ = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue
            # convierto a tensor [1, canales, seq_len]
            data = np.array(sample, dtype=np.float32)
            # asumo cada muestra un punto en el tiempo → seq_len=1
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(2).to(device)

            # inferencia
            with torch.no_grad():
                out = model(tensor)
                idx = int(torch.argmax(out, dim=1).cpu().item())
                label = class_names[idx]

            # envío string vía LSL
            outlet.push_sample([label])
            logging.info(f">>> {label}")

            # opcional: pequeña pausa
            time.sleep(0.01)

    except KeyboardInterrupt:
        logging.info("Terminando inferencia por Ctrl+C")

if __name__ == "__main__":
    main()
