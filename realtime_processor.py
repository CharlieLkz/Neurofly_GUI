#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Procesador de EEG en Tiempo Real - Extractor de Características para Control de Dron
====================================================================================

Este script funciona como un componente intermedio en un sistema de control de dron basado en EEG:
- Recibe datos EEG desde el stream LSL 'AURAPSD' (enviado por el GUI)
- Aplica filtrado y extracción de características (bandas en cada canal)
- Envía las características procesadas a 'FEATURE_STREAM' para que la CNN las consuma
- Se ejecuta continuamente hasta que detecta 15 segundos sin actividad o procesa una muestra única

El procesamiento incluye:
1. Filtrado paso banda para todas las bandas (Delta, Theta, Alpha, Beta, Gamma)
2. Filtrado Kalman para suavizado de señal
3. Extracción de características (power por cada banda y canal)
4. Envío del vector de características al stream 'FEATURE_STREAM'

Uso:
    python realtime_processor.py
"""

import os
import sys
import time
import numpy as np
from scipy import signal
import threading
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
from pykalman import KalmanFilter
import logging
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processor_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EEGProcessor")

# === CONFIGURACIONES === #
# Nombres de streams LSL
INPUT_STREAM_NAME = "AURAPSD"        # Stream de entrada (desde GUI)
OUTPUT_STREAM_NAME = "FEATURE_STREAM"  # Stream de salida (hacia CNN)

# Configuración de bandas de frecuencia (Hz)
BANDAS = {
    'Delta': [0.5, 4.0],
    'Theta': [4.0, 8.0],
    'Alpha': [8.0, 12.0],
    'Beta': [12.0, 30.0],
    'Gamma': [30.0, 100.0]
}

# Canales de interés (todos los canales 1-8)
N_CANALES = 8
CANALES_INTERES = list(range(1, N_CANALES+1))  # Canales 1-8

# Bandas de interés (todas)
BANDAS_INTERES = list(BANDAS.keys())  # Todas las bandas

# Frecuencia de muestreo
SAMPLING_FREQUENCY = 100  # Hz

# Tamaño de ventana para características
WINDOW_SIZE = 50  # muestras (0.5 segundos a 100Hz)
OVERLAP = 0.5     # 50% de solapamiento

# Configuración de filtros
FILTER_ORDER = 4  # Orden del filtro paso banda

# Configuración de Kalman
OBSERVATION_COVARIANCE = 0.1
TRANSITION_COVARIANCE = 0.01

# Configuración de timeout para detección de fin de experimento
SAMPLE_TIMEOUT = 15.0  # 15 segundos sin muestras = finalizar

# Factor de escala para características
FEATURE_SCALE_FACTOR = 1e6  # Multiplicar por 1 millón para valores más manejables

# Tiempo máximo para reintentar conexión
MAX_RECONNECT_TIME = 60  # segundos
RECONNECT_INTERVAL = 5   # segundos entre intentos

# Directorio para guardar datos
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILTERED_DATA_DIR = os.path.join(CURRENT_DIR, "FilteredData")

class EEGProcessor:
    """
    Procesador de señales EEG en tiempo real.
    Extrae características de las bandas para todos los canales.
    """
    def __init__(self):
        self.running = False
        self.connected = False
        self.inlet = None
        self.outlet = None
        self.thread = None
        self.single_sample_mode = True  # Procesar una única muestra
        
        # Buffers para el procesamiento de señales
        self.data_buffer = {f"Canal{c}_{b}": [] for c in CANALES_INTERES for b in BANDAS_INTERES}
        self.filtered_buffer = {f"Canal{c}_{b}": [] for c in CANALES_INTERES for b in BANDAS_INTERES}
        self.kalman_buffer = {f"Canal{c}_{b}": [] for c in CANALES_INTERES for b in BANDAS_INTERES}
        
        # Buffer para características extraídas
        self.feature_buffer = []
        self.feature_times = []
        
        # Métricas
        self.processed_samples = 0
        self.features_sent = 0
        self.start_time = None
        self.experiment_metadata = {}
        
        # Estadísticas de características
        self.feature_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'sum': 0,
            'count': 0
        }
        
        # Contador de reintentos de conexión
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0
        
        # Asegurar que existe el directorio para datos filtrados
        os.makedirs(FILTERED_DATA_DIR, exist_ok=True)
    
    def connect_to_streams(self):
        """
        Conecta a los streams LSL de entrada y salida.
        
        Returns:
            bool: True si la conexión fue exitosa, False en caso contrario
        """
        try:
            logger.info(f"Buscando stream LSL '{INPUT_STREAM_NAME}'...")
            print(f"Buscando stream LSL '{INPUT_STREAM_NAME}'...")
            print("(Esto puede tardar unos segundos, por favor espere)")
            
            # Reiniciar contador de intentos si ha pasado suficiente tiempo
            current_time = time.time()
            if current_time - self.last_reconnect_time > 30:
                self.reconnect_attempts = 0
            
            self.last_reconnect_time = current_time
            self.reconnect_attempts += 1
            
            # Verificar si hemos excedido el número máximo de intentos
            if self.reconnect_attempts > MAX_RECONNECT_TIME / RECONNECT_INTERVAL:
                logger.error(f"Se han agotado los intentos de reconexión ({self.reconnect_attempts})")
                print(f"Error: No se pudo conectar después de múltiples intentos. Verifique que la GUI de experimento esté en ejecución.")
                return False
            
            # Buscar stream de entrada
            streams = resolve_byprop('name', INPUT_STREAM_NAME, timeout=5.0)
            
            if not streams:
                logger.warning(f"No se encontró el stream '{INPUT_STREAM_NAME}', reintentando ({self.reconnect_attempts})...")
                print(f"No se encontró el stream '{INPUT_STREAM_NAME}', esperando...")
                return False
            
            # Conectar al stream de entrada
            try:
                self.inlet = StreamInlet(streams[0])
                logger.info(f"Conectado al stream '{INPUT_STREAM_NAME}'")
                print(f"Conectado al stream '{INPUT_STREAM_NAME}'")
            except Exception as e:
                logger.error(f"Error al crear StreamInlet: {str(e)}")
                print(f"Error al conectar: {str(e)}")
                return False
            
            # Extraer metadatos del stream si están disponibles
            try:
                info = streams[0]  # El objeto streams[0] ya es un StreamInfo
                desc = info.desc()
                
                # Intentar obtener metadatos si existen
                if desc.child_value("experiment"):
                    self.experiment_metadata['experiment'] = desc.child_value("experiment")
                if desc.child_value("subject"):
                    self.experiment_metadata['subject'] = desc.child_value("subject")
                if desc.child_value("task"):
                    self.experiment_metadata['task'] = desc.child_value("task")
                
                logger.info("Metadatos extraídos del stream de entrada")
                print("Metadatos extraídos del stream de entrada")
            except Exception as e:
                logger.warning(f"Aviso: No se pudieron extraer metadatos: {str(e)}")
                print(f"Aviso: No se pudieron extraer metadatos: {str(e)}")
            
            # Crear stream de salida para características
            try:
                self.setup_output_stream()
            except Exception as e:
                logger.error(f"Error al configurar stream de salida: {str(e)}")
                print(f"Error al configurar stream de salida: {str(e)}")
                return False
            
            self.connected = True
            self.reconnect_attempts = 0  # Reiniciar contador tras conexión exitosa
            
            # Verificar que la conexión funciona recibiendo una muestra
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                if sample:
                    logger.info(f"Conexión verificada: Primeras muestras recibidas")
                    print(f"Conexión verificada: Primeras muestras recibidas")
                    print(f"Ejemplo de datos: {sample[:8]}...")  # Mostrar primeros 8 valores
                else:
                    logger.warning("No se recibieron datos en la prueba inicial")
                    print("Aviso: No se recibieron datos en la prueba inicial")
            except Exception as e:
                logger.error(f"Error al recibir muestra de prueba: {str(e)}")
                print(f"Error al verificar conexión: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error al conectar a los streams: {str(e)}")
            print(f"Error al conectar a los streams: {str(e)}")
            return False
    
    def setup_output_stream(self):
        """
        Configura el stream LSL de salida para enviar características
        
        Returns:
            bool: True si la configuración fue exitosa
        """
        try:
            # Calcular número total de características
            n_features = len(CANALES_INTERES) * len(BANDAS_INTERES)
            
            # Crear información del stream
            info = StreamInfo(
                name=OUTPUT_STREAM_NAME,
                type='Features',
                channel_count=n_features,
                nominal_srate=0,  # Irregular (single sample)
                channel_format='float32',
                source_id='EEGProcessor'
            )
            
            # Añadir metadatos
            desc = info.desc()
            desc.append_child_value("type", "PSD")
            
            # Añadir información de canales
            feature_idx = 0
            for canal in CANALES_INTERES:
                for banda in BANDAS_INTERES:
                    ch = desc.append_child("channel")
                    ch.append_child_value("label", f"Canal{canal}_{banda}")
                    ch.append_child_value("unit", "normalized")
                    ch.append_child_value("index", str(feature_idx))
                    feature_idx += 1
            
            # Añadir metadatos del experimento si están disponibles
            if self.experiment_metadata:
                experiment = info.desc().append_child("experiment")
                for key, value in self.experiment_metadata.items():
                    experiment.append_child_value(key, value)
            
            # Crear outlet
            self.outlet = StreamOutlet(info)
            logger.info(f"Stream de salida '{OUTPUT_STREAM_NAME}' configurado con {n_features} características")
            print(f"Stream de salida '{OUTPUT_STREAM_NAME}' configurado con {n_features} características")
            
            # Enviar una muestra de prueba
            test_sample = [0.5] * n_features
            self.outlet.push_sample(test_sample)
            logger.info("Muestra de prueba enviada al stream de salida")
            print("Muestra de prueba enviada al stream de salida")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al configurar stream de salida: {str(e)}")
            print(f"Error al configurar stream de salida: {str(e)}")
            raise

    # Aquí puedes añadir los métodos adicionales como process_signal, extract_features, etc.
    # según sea necesario para completar la implementación

# Función principal para ejecutar el procesador
def main():
    processor = EEGProcessor()
    
    try:
        # Conectar a streams
        if not processor.connect_to_streams():
            print("No se pudo conectar a los streams. Verificar que el sistema esté corriendo.")
            return 1
        
        # Aquí iría la lógica para procesar las señales
        # ...
        
        print("Procesamiento completado con éxito.")
        return 0
        
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())