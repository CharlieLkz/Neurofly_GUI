#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cnn_inference.py - Módulo de inferencia para sistema EEG-Dron
Recibe una característica desde FEATURE_STREAM, hace una predicción con CNN, 
calcula la feature más relevante y envía el resultado a PREDICTION_STREAM via LSL
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import threading
from datetime import datetime
import logging
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cnn_inference_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CNN_Inference")

# Configuraciones globales
RESULTS_DIR = "CNNResults"
MODELS_DIR = "CNNModels/models/ensamble/"
INPUT_STREAM_NAME = "FEATURE_STREAM"
OUTPUT_STREAM_NAME = "PREDICTION_STREAM"
SAMPLE_TIMEOUT = 15.0  # segundos (cambiado a 15 según especificación)
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_INTERVAL = 3  # segundos

# Crear directorios si no existen
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Definición de la arquitectura del modelo CNN
class EEGCNNModel(nn.Module):
    def __init__(self, input_size=40, num_classes=6):
        super(EEGCNNModel, self).__init__()
        
        # Capas convolucionales
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calcular tamaño después de convoluciones y pooling
        # input_size -> input_size/2
        conv_output_size = max(1, input_size // 2)
        
        # Capas fully connected
        self.fc1 = nn.Linear(16 * conv_output_size, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # x debe tener forma [batch_size, 1, features]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Aplanar para capa fully connected
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNNInferenceEngine:
    def __init__(self, models_dir=MODELS_DIR, 
                 input_stream_name=INPUT_STREAM_NAME, 
                 output_stream_name=OUTPUT_STREAM_NAME):
        
        self.models_dir = models_dir
        self.input_stream_name = input_stream_name
        self.output_stream_name = output_stream_name
        
        # Estado del sistema
        self.running = False
        self.connected = False
        self.sample_processed = False
        self.model = None
        self.model_path = None
        self.input_stream = None
        self.output_stream = None
        self.reconnect_attempts = 0
        
        # Almacenamiento de datos para análisis
        self.features = None
        self.prediction = None
        self.confidence = None
        self.most_relevant_feature = None
        
        # Clases por defecto
        self.class_names = [
            "RightArmThinking", "LeftArmThinking", "RightFistThinking", 
            "LeftFistThinking", "RightFootThinking", "LeftFootThinking"
        ]
        
        # Inicializar el modelo y los streams
        self._find_best_model()
        self._setup_output_stream()
        
    def _setup_output_stream(self):
        """Configura el stream LSL de salida para las predicciones"""
        # Crear información del stream
        info = StreamInfo(
            name=self.output_stream_name,
            type="Prediction",
            channel_count=2,  # Clase predicha y índice de feature relevante
            nominal_srate=0,  # Irregular (single sample)
            channel_format='string',
            source_id='CNNInference'
        )
        
        # Crear outlet
        self.output_stream = StreamOutlet(info)
        logger.info(f"Stream de salida '{self.output_stream_name}' configurado")
        print(f"Stream de salida '{self.output_stream_name}' configurado")
        
        return True
    
    def _find_input_stream(self):
        """Busca y conecta al stream LSL de características de entrada"""
        logger.info(f"Buscando stream de entrada '{self.input_stream_name}'...")
        print(f"Buscando stream de entrada '{self.input_stream_name}'...")
        
        # Resolver stream por nombre
        streams = resolve_byprop('name', self.input_stream_name, timeout=5.0)
        
        if not streams:
            logger.warning(f"No se encontró el stream '{self.input_stream_name}'")
            print(f"No se encontró el stream '{self.input_stream_name}'")
            return False
        
        # Crear inlet
        self.input_stream = StreamInlet(streams[0])
        logger.info(f"Conectado al stream '{self.input_stream_name}'")
        print(f"Conectado exitosamente al stream '{self.input_stream_name}'")
        
        return True
    
    def _find_best_model(self):
        """Busca el mejor modelo CNN pre-entrenado en el directorio de modelos"""
        logger.info(f"Buscando modelos CNN pre-entrenados en {self.models_dir}")
        print(f"Buscando modelos CNN pre-entrenados en {self.models_dir}")
        
        # Buscar archivos de modelo (.pt o .pth)
        model_files = glob.glob(os.path.join(self.models_dir, "*.pt")) + \
                     glob.glob(os.path.join(self.models_dir, "*.pth"))
        
        if not model_files:
            logger.warning(f"No se encontraron modelos en {self.models_dir}")
            print(f"No se encontraron modelos en {self.models_dir}")
            
            # Usar un modelo dummy para pruebas si no hay modelos
            self.model = EEGCNNModel(input_size=40, num_classes=6)
            self.model_path = "modelo_predeterminado"
            logger.info("Se está utilizando un modelo vacío predeterminado")
            print("Aviso: Se está utilizando un modelo vacío predeterminado")
            return
        
        # Por ahora usamos el primer modelo encontrado
        # En un sistema más avanzado, podríamos implementar criterios de selección
        self.model_path = model_files[0]
        
        try:
            # Cargar modelo
            self.model = EEGCNNModel(input_size=40, num_classes=6)
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()  # Modo evaluación
            
            logger.info(f"Modelo cargado exitosamente desde: {self.model_path}")
            print(f"Modelo cargado exitosamente desde: {self.model_path}")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            print(f"Error al cargar el modelo: {str(e)}")
            
            # Crear un modelo vacío en caso de error
            self.model = EEGCNNModel(input_size=40, num_classes=6)
            logger.warning("Se está utilizando un modelo vacío debido a un error de carga")
            print("Aviso: Se está utilizando un modelo vacío debido a un error de carga")
    
    def preprocess_features(self, features):
        """Preprocesa las características para la inferencia con PyTorch"""
        # Convertir a numpy array si no lo es
        features_np = np.array(features, dtype=np.float32)
        
        # Reshape para CNN [batch, channels, features]
        features_tensor = torch.FloatTensor(features_np).reshape(1, 1, -1)
        
        return features_tensor
    
    def find_most_relevant_feature(self, features_tensor):
        """
        Identifica la feature más relevante basada en la que tiene mayor magnitud.
        
        Args:
            features_tensor: Tensor de características de entrada
            
        Returns:
            int: Índice de la característica más relevante
        """
        # Obtener el vector de características como numpy array
        features_np = features_tensor.detach().numpy().flatten()
        
        # Encontrar el índice del valor con mayor magnitud absoluta
        most_relevant_idx = np.argmax(np.abs(features_np))
        
        return most_relevant_idx
    
    def get_feature_label(self, feature_idx):
        """
        Convierte el índice de feature en una etiqueta legible.
        
        Args:
            feature_idx: Índice de la característica
            
        Returns:
            str: Etiqueta de la característica en formato "Canal X - Banda Y"
        """
        if feature_idx < 0 or feature_idx >= 40:  # 8 canales * 5 bandas = 40 features
            return "Desconocida"
        
        # Calcular canal y banda
        canal = (feature_idx // 5) + 1  # División entera por 5 + 1 (1-based)
        banda_idx = feature_idx % 5      # Módulo 5
        
        # Mapeo de índices de banda a nombres
        bandas = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        banda = bandas[banda_idx] if 0 <= banda_idx < len(bandas) else "Desconocida"
        
        return f"Canal {canal} – Banda {banda}"
    
    def predict(self, features_tensor):
        """
        Realiza una predicción con el modelo CNN y encuentra la feature más relevante
        
        Args:
            features_tensor: Tensor de características preprocesadas
            
        Returns:
            tuple: (clase_predicha, confianza, índice_feature_relevante)
        """
        if self.model is None:
            return "ERROR", 0.0, -1
        
        # Encontrar la feature más relevante
        most_relevant_feature_idx = self.find_most_relevant_feature(features_tensor)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Obtener clase con mayor probabilidad
        confidence, class_idx = torch.max(probabilities, 1)
        
        confidence_value = float(confidence.item())
        class_index = int(class_idx.item())
        
        # Convertir índice a nombre de clase
        if 0 <= class_index < len(self.class_names):
            class_name = self.class_names[class_index]
        else:
            class_name = f"Unknown_Class_{class_index}"
        
        return class_name, confidence_value, most_relevant_feature_idx
    
    def connect(self):
        """Establece conexión con el stream de entrada"""
        connected = False
        self.reconnect_attempts = 0
        
        while not connected and self.reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            self.reconnect_attempts += 1
            logger.info(f"Intento de conexión {self.reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}")
            print(f"Intento de conexión {self.reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}")
            
            if self._find_input_stream():
                connected = True
                self.connected = True
                logger.info("Conexión establecida con éxito")
                print("Conexión establecida con éxito")
                break
            
            if not connected and self.reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
                logger.info(f"Reintentando en {RECONNECT_INTERVAL} segundos...")
                print(f"Reintentando en {RECONNECT_INTERVAL} segundos...")
                time.sleep(RECONNECT_INTERVAL)
        
        if not connected:
            logger.error(f"No se pudo conectar después de {MAX_RECONNECT_ATTEMPTS} intentos")
            print(f"No se pudo conectar después de {MAX_RECONNECT_ATTEMPTS} intentos")
        
        return connected
    
    def process_feature_sample(self):
        """
        Procesa una única muestra de característica del stream de entrada,
        realiza una predicción, y envía el resultado al stream de salida.
        
        Returns:
            bool: True si se procesó correctamente la muestra
        """
        logger.info("Esperando muestra de características...")
        print("Esperando muestra de características...")
        
        # Esperar y recibir muestra
        sample, timestamp = self.input_stream.pull_sample(timeout=SAMPLE_TIMEOUT)
        
        if not sample:
            logger.warning(f"No se recibieron datos después de {SAMPLE_TIMEOUT} segundos")
            print(f"No se recibieron datos después de {SAMPLE_TIMEOUT} segundos")
            return False
        
        # Guardar características en archivo
        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
        features_filename = os.path.join(RESULTS_DIR, f"features_{timestamp_str}.csv")
        
        with open(features_filename, 'w') as f:
            f.write(f"# Características recibidas\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Fecha: {timestamp_str}\n")
            
            # Escribir valores
            f.write("# Valores\n")
            for i, val in enumerate(sample):
                f.write(f"Feature_{i},{val}\n")
        
        logger.info(f"Características guardadas en {features_filename}")
        print(f"Características guardadas en {features_filename}")
        
        # Preprocesar características
        features_tensor = self.preprocess_features(sample)
        
        # Realizar predicción
        class_name, confidence, most_relevant_feature_idx = self.predict(features_tensor)
        
        # Guardar resultados
        self.prediction = class_name
        self.confidence = confidence
        self.most_relevant_feature = most_relevant_feature_idx
        
        # Obtener etiqueta legible para la feature relevante
        feature_label = self.get_feature_label(most_relevant_feature_idx)
        
        logger.info(f"Predicción: {class_name} (Confianza: {confidence:.4f})")
        logger.info(f"Feature más relevante: {feature_label} (índice: {most_relevant_feature_idx})")
        print(f"Predicción: {class_name} (Confianza: {confidence:.4f})")
        print(f"Feature más relevante: {feature_label} (índice: {most_relevant_feature_idx})")
        
        # Guardar predicción en archivo
        prediction_filename = os.path.join(RESULTS_DIR, f"prediction_{timestamp_str}.csv")
        
        with open(prediction_filename, 'w') as f:
            f.write(f"# Predicción generada\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Fecha: {timestamp_str}\n")
            f.write(f"Clase,Confianza,Feature_Relevante\n")
            f.write(f"{class_name},{confidence:.6f},{most_relevant_feature_idx}\n")
        
        logger.info(f"Predicción guardada en {prediction_filename}")
        print(f"Predicción guardada en {prediction_filename}")
        
        # Enviar predicción al stream LSL
        self.output_stream.push_sample([class_name, str(most_relevant_feature_idx)])
        logger.info(f"Predicción enviada al stream '{self.output_stream_name}'")
        print(f"Predicción enviada al stream '{self.output_stream_name}'")
        
        # Marcar como procesado
        self.sample_processed = True
        
        return True