#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN_Inference.py
Realiza inferencia en tiempo real usando el modelo CNN entrenado.
Recibe datos de FEATURE_STREAM (LSL) y envía predicciones a CNN_COMMANDS (LSL).
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
import logging
from pathlib import Path
import traceback

# === Configuración de rutas y logs === #
BASE_DIR = Path(__file__).parent.resolve()
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "cnn_inference_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CNNInference")

# === CONFIGURACIONES === #
# Nombres de streams LSL
INPUT_STREAM_NAME = "FEATURE_STREAM"    # Stream de entrada (desde EEGProcessor)
OUTPUT_STREAM_NAME = "CNN_COMMANDS"     # Stream de salida (hacia CommandSender)

# Configuración del modelo
MODEL_PATH = BASE_DIR / "models" / "eeg_model_cv_best_1.0000.pt"
INPUT_FEATURES = 4  # 4 características: Canal4_Alpha, Canal4_Beta, Canal5_Alpha, Canal5_Beta
SEQUENCE_LENGTH = 55  # Longitud de la secuencia temporal
NUM_CLASSES = 4  # Número de clases de salida

# Clases de salida
CLASS_NAMES = [
    "LeftArmThinking",
    "RightArmThinking",
    "LeftFistThinking",
    "RightFistThinking"
]

class EEGCNN(nn.Module):
    def __init__(self, input_features, num_classes):
        super(EEGCNN, self).__init__()
        
        # Primera capa convolucional (16 canales de salida)
        self.conv1 = nn.Conv1d(input_features, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Segunda capa convolucional (32 canales)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Tercera capa convolucional (64 canales)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Capas de activación y pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)
        
        # Capas fully connected
        self.fc1 = nn.Linear(64 * 7, 128)  # 7 = 55 // 2^3 (3 capas de pooling)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Primera capa
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Segunda capa
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Tercera capa
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Aplanar y fully connected
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNInference:
    def __init__(self):
        self.running = True
        self.connected = False
        self.inlet = None
        self.outlet = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar modelo
        self._load_model()
        
        # Configurar streams LSL
        self._setup_streams()
        
        # Estadísticas
        self.predictions_sent = 0
        self.start_time = time.time()
    
    def _load_model(self):
        """Carga el modelo CNN desde el archivo .pt"""
        try:
            # Crear modelo con la arquitectura correcta
            self.model = EEGCNN(INPUT_FEATURES, NUM_CLASSES)
            
            # Cargar pesos
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Configurar para inferencia
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Modelo cargado correctamente desde {MODEL_PATH}")
            logger.info(f"Arquitectura del modelo: {self.model}")
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    def _setup_streams(self):
        """Configura los streams LSL de entrada y salida"""
        try:
            # Buscar stream de entrada
            streams = resolve_byprop('name', INPUT_STREAM_NAME, timeout=5.0)
            if not streams:
                raise Exception(f"No se encontró el stream '{INPUT_STREAM_NAME}'")
            
            # Conectar al stream de entrada
            self.inlet = StreamInlet(streams[0])
            logger.info(f"Conectado al stream '{INPUT_STREAM_NAME}'")
            
            # Crear stream de salida
            info = StreamInfo(
                name=OUTPUT_STREAM_NAME,
                type='Predictions',
                channel_count=1,
                nominal_srate=0,
                channel_format='string',
                source_id='CNNInference'
            )
            
            # Agregar metadatos
            desc = info.desc()
            desc.append_child_value("format", "CLASS_LABEL")
            desc.append_child_value("classes", ",".join(CLASS_NAMES))
            
            self.outlet = StreamOutlet(info)
            logger.info(f"Stream de salida '{OUTPUT_STREAM_NAME}' configurado")
            
            self.connected = True
            
        except Exception as e:
            logger.error(f"Error al configurar streams: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    def process_sample(self, sample):
        """Procesa una muestra y realiza la inferencia"""
        try:
            # Convertir a tensor y dar forma correcta (1, 4, 55)
            data = np.array(sample).reshape(1, INPUT_FEATURES, SEQUENCE_LENGTH)
            data = torch.FloatTensor(data).to(self.device)
            
            # Inferencia
            with torch.no_grad():
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                predicted_class = CLASS_NAMES[predicted.item()]
            
            return predicted_class
            
        except Exception as e:
            logger.error(f"Error al procesar muestra: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def run(self):
        """Bucle principal de inferencia"""
        logger.info("Iniciando inferencia...")
        
        while self.running:
            try:
                # Obtener muestra del stream
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    # Procesar muestra
                    predicted_class = self.process_sample(sample)
                    
                    if predicted_class:
                        # Enviar predicción
                        self.outlet.push_sample([predicted_class])
                        self.predictions_sent += 1
                        
                        # Log periódico
                        if self.predictions_sent % 10 == 0:
                            elapsed = time.time() - self.start_time
                            rate = self.predictions_sent / elapsed
                            logger.info(f"Predicciones enviadas: {self.predictions_sent} ({rate:.1f} pred/s)")
                
            except Exception as e:
                logger.error(f"Error en el bucle principal: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(1)
    
    def stop(self):
        """Detiene la inferencia"""
        self.running = False
        if self.inlet:
            self.inlet.close_stream()
        if self.outlet:
            self.outlet.close_stream()
        logger.info("Inferencia detenida")

def main():
    try:
        inference = CNNInference()
        inference.run()
    except KeyboardInterrupt:
        logger.info("Detenido por el usuario")
    except Exception as e:
        logger.error(f"Error en main: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        if 'inference' in locals():
            inference.stop()

if __name__ == "__main__":
    main() 