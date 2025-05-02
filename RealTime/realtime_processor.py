#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Procesador de EEG en Tiempo Real - Extractor de Características para Control de Dron
====================================================================================

Este script funciona como un componente intermedio en un sistema de control de dron basado en EEG:
- Recibe datos EEG desde el stream LSL 'AURA_Power' (enviado por el GUI)
- Acumula las bandas Alpha y Beta para canales 4 y 5
- Envía las características procesadas a 'FEATURE_STREAM' para que la CNN las consuma
- Se ejecuta continuamente hasta que detecta 15 segundos sin actividad

Uso:
    python realtime_processor.py
"""

import os
import sys
import time
import numpy as np
import threading
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
import logging
import traceback
from pathlib import Path
from collections import deque

# === Configuración de rutas y logs === #
BASE_DIR = Path(__file__).parent.resolve()
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "processor_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EEGProcessor")

# === CONFIGURACIONES === #
# Nombres de streams LSL
INPUT_STREAM_NAME = "AURA_Power"        # Stream de entrada (desde GUI)
OUTPUT_STREAM_NAME = "FEATURE_STREAM"    # Stream de salida (hacia CNN)
EMERGENCY_STREAM_NAME = "EMERGENCY_STOP" # Stream para señales de emergencia

# Configuración específica para el formato de salida
CANALES_SALIDA = [4, 5]  # Solo canales 4 y 5
BANDAS_SALIDA = ['Alpha', 'Beta']  # Solo Alpha y Beta
BATCH_SIZE = 55  # Número de muestras a acumular antes de enviar

# Configuración de bandas de frecuencia (Hz)
BANDAS = {
    'Alpha': [8.0, 12.0],
    'Beta': [12.0, 30.0]
}

# Canales de interés (solo 4 y 5)
CANALES_INTERES = CANALES_SALIDA
BANDAS_INTERES = list(BANDAS.keys())

# Nombres de columnas en orden
COLUMNAS_SALIDA = [f"Canal{c}_{b}" for c in CANALES_SALIDA for b in BANDAS_SALIDA]

# Configuración de timeout para detección de fin de experimento
SAMPLE_TIMEOUT = 15.0  # 15 segundos sin muestras = finalizar

# Tiempo máximo para reintentar conexión
MAX_RECONNECT_TIME = 60  # segundos
RECONNECT_INTERVAL = 5   # segundos entre intentos

# Directorio para datos filtrados (relativo al script)
FILTERED_DATA_DIR = BASE_DIR / "FilteredData"
FILTERED_DATA_DIR.mkdir(exist_ok=True)

# Directorio para datos de diagnóstico
DIAGNOSTICS_DIR = BASE_DIR / "Diagnostics"
DIAGNOSTICS_DIR.mkdir(exist_ok=True)

# Modos de operación
DEBUG_MODE = False  # Activa mensajes de depuración adicionales

class EEGProcessor:
    """
    Procesador de señales EEG en tiempo real.
    Extrae características de las bandas Alpha y Beta para los canales 4 y 5.
    """
    def __init__(self):
        self.running = True
        self.connected = False
        self.inlet = None
        self.outlet = None
        self.emergency_outlet = None
        self.thread = None
        self.single_sample_mode = False
        
        # Buffer para acumular muestras (usando deque para mejor rendimiento)
        self.batch_buffer = deque(maxlen=BATCH_SIZE)
        
        # Métricas y monitoreo
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
        
        # Monitoreo de latencia
        self.latency_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0,
            'count': 0
        }
        
        # Contador de reintentos de conexión
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0
        self.last_sample_time = 0
        
        # Validar configuración
        self._validate_config()
    
    def _validate_config(self):
        """Valida la configuración para detectar problemas potenciales temprano"""
        if len(CANALES_SALIDA) * len(BANDAS_SALIDA) != 4:
            logger.warning(f"La configuración actual resultará en {len(CANALES_SALIDA) * len(BANDAS_SALIDA)} canales de salida, pero el modelo espera exactamente 4")
        
        if BATCH_SIZE != 55:
            logger.warning(f"El modelo espera secuencias de longitud 55, pero BATCH_SIZE = {BATCH_SIZE}")
    
    def connect_to_streams(self):
        """Conecta a los streams LSL con manejo robusto de errores y reconexión"""
        while self.running:
            try:
                logger.info(f"Buscando stream LSL '{INPUT_STREAM_NAME}'...")
                
                # Reiniciar contador de intentos si ha pasado suficiente tiempo
                current_time = time.time()
                if current_time - self.last_reconnect_time > 30:
                    self.reconnect_attempts = 0
                
                self.last_reconnect_time = current_time
                self.reconnect_attempts += 1
                
                # Verificar si hemos excedido el número máximo de intentos
                if self.reconnect_attempts > MAX_RECONNECT_TIME / RECONNECT_INTERVAL:
                    logger.error(f"Se han agotado los intentos de reconexión ({self.reconnect_attempts})")
                    return False
                
                # Buscar stream de entrada
                streams = resolve_byprop('name', INPUT_STREAM_NAME, timeout=5.0)
                
                if not streams:
                    logger.warning(f"No se encontró el stream '{INPUT_STREAM_NAME}', reintentando...")
                    time.sleep(RECONNECT_INTERVAL)
                    continue
                
                # Conectar al stream de entrada
                try:
                    self.inlet = StreamInlet(streams[0])
                    logger.info(f"Conectado al stream '{INPUT_STREAM_NAME}'")
                except Exception as e:
                    logger.error(f"Error al crear StreamInlet: {str(e)}")
                    time.sleep(RECONNECT_INTERVAL)
                    continue
                
                # Crear stream de salida
                try:
                    self.setup_output_stream()
                except Exception as e:
                    logger.error(f"Error al configurar stream de salida: {str(e)}")
                    time.sleep(RECONNECT_INTERVAL)
                    continue
                
                self.connected = True
                self.reconnect_attempts = 0
                self.last_sample_time = time.time()
                
                # Verificar que la conexión funciona
                try:
                    sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                    if sample:
                        logger.info("Conexión verificada")
                        return True
                except Exception as e:
                    logger.error(f"Error al verificar conexión: {str(e)}")
                    time.sleep(RECONNECT_INTERVAL)
                    continue
                
            except Exception as e:
                logger.error(f"Error en la conexión: {str(e)}")
                time.sleep(RECONNECT_INTERVAL)
                continue
        
        return False
    
    def setup_output_stream(self):
        """
        Configura el stream LSL de salida para enviar datos formateados para CNN.
        """
        try:
            # Calcular el tamaño total del array aplanado (4 canales x 55 muestras = 220 valores)
            total_values = len(CANALES_SALIDA) * len(BANDAS_SALIDA) * BATCH_SIZE
            
            # Configurar stream de características con el número correcto de canales
            info = StreamInfo(
                name=OUTPUT_STREAM_NAME,
                type='Features',
                channel_count=total_values,  # 220 valores (4x55)
                nominal_srate=0,  # Irregular
                channel_format='float32',
                source_id='EEGProcessor'
            )
            
            # Agregar metadatos para que CNN_Inference sepa cómo interpretar los datos
            desc = info.desc()
            desc.append_child_value("format", "CNN_BATCH")
            desc.append_child_value("original_channels", str(len(CANALES_SALIDA) * len(BANDAS_SALIDA)))
            desc.append_child_value("sequence_length", str(BATCH_SIZE))
            desc.append_child_value("flattened", "true")
            
            # Agregar nombres de canales originales
            channels = desc.append_child("original_channel_names")
            for col in COLUMNAS_SALIDA:
                channels.append_child("channel").append_child_value("label", col)
            
            self.outlet = StreamOutlet(info)
            logger.info(f"Stream de salida '{OUTPUT_STREAM_NAME}' configurado con canales: {COLUMNAS_SALIDA}")
            logger.info(f"Formato de datos para CNN: batch=1, canales={len(CANALES_SALIDA) * len(BANDAS_SALIDA)}, secuencia={BATCH_SIZE}")
            return True
            
        except Exception as e:
            logger.error(f"Error al configurar stream de salida: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _update_stats(self, features):
        """Actualiza las estadísticas de las características procesadas"""
        try:
            # Convertir a numpy array si no lo es
            features = np.array(features, dtype=np.float32)
            
            # Actualizar estadísticas básicas
            self.feature_stats['min'] = min(self.feature_stats['min'], np.min(features))
            self.feature_stats['max'] = max(self.feature_stats['max'], np.max(features))
            self.feature_stats['sum'] += np.sum(features)
            self.feature_stats['count'] += 1
            
            # Calcular media móvil
            if self.feature_stats['count'] > 0:
                self.feature_stats['mean'] = self.feature_stats['sum'] / self.feature_stats['count']
            
            # Log periódico de estadísticas
            if self.feature_stats['count'] % 100 == 0:
                logger.info(f"Estadísticas de características:")
                logger.info(f"  Mínimo: {self.feature_stats['min']:.3f}")
                logger.info(f"  Máximo: {self.feature_stats['max']:.3f}")
                logger.info(f"  Media: {self.feature_stats['mean']:.3f}")
                logger.info(f"  Total muestras: {self.feature_stats['count']}")
                
        except Exception as e:
            logger.error(f"Error al actualizar estadísticas: {str(e)}")
            logger.error(traceback.format_exc())

    def process_signal(self, data, timestamp):
        """Procesa la señal con monitoreo de latencia"""
        try:
            start_time = time.time()
            
            # Verificar timeout
            if time.time() - self.last_sample_time > SAMPLE_TIMEOUT:
                logger.warning("Timeout detectado - finalizando procesamiento")
                self.running = False
                return
            
            self.last_sample_time = time.time()
            
            # Convertir a numpy array
            data = np.array(data, dtype=np.float32)
            
            # Aplicar filtros
            filtered_data = self._apply_filters(data)
            
            # Extraer características
            features = []
            for canal in CANALES_SALIDA:
                for banda in BANDAS_SALIDA:
                    idx = (canal - 1) * len(BANDAS) + list(BANDAS.keys()).index(banda)
                    if idx < len(filtered_data):
                        features.append(filtered_data[idx])
                    else:
                        logger.error(f"Índice fuera de rango: {idx} (max {len(filtered_data)-1})")
                        return
            
            # Añadir al buffer
            self.batch_buffer.append(features)
            
            # Procesar batch completo
            if len(self.batch_buffer) == BATCH_SIZE:
                self.send_batch()
            
            # Calcular y registrar latencia
            latency = time.time() - start_time
            self._update_latency_stats(latency)
            
        except Exception as e:
            logger.error(f"Error en process_signal: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _apply_filters(self, data):
        """Aplica filtros a los datos EEG"""
        try:
            # Filtro bandpass para cada banda
            filtered_data = np.zeros_like(data)
            for i, (banda, freqs) in enumerate(BANDAS.items()):
                # Aquí iría la implementación del filtro bandpass
                # Por ahora solo pasamos los datos
                filtered_data[i::len(BANDAS)] = data[i::len(BANDAS)]
            
            # Filtro Kalman
            # Aquí iría la implementación del filtro Kalman
            # Por ahora solo pasamos los datos filtrados
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error al aplicar filtros: {str(e)}")
            return data
    
    def _update_latency_stats(self, latency):
        """Actualiza las estadísticas de latencia"""
        self.latency_stats['min'] = min(self.latency_stats['min'], latency)
        self.latency_stats['max'] = max(self.latency_stats['max'], latency)
        self.latency_stats['sum'] += latency
        self.latency_stats['count'] += 1
        
        # Log periódico de latencia
        if self.latency_stats['count'] % 100 == 0:
            avg_latency = self.latency_stats['sum'] / self.latency_stats['count']
            logger.info(f"Latencia: min={self.latency_stats['min']:.3f}s, max={self.latency_stats['max']:.3f}s, avg={avg_latency:.3f}s")
    
    def send_batch(self):
        """
        Envía un batch completo de 55 muestras y guarda en CSV.
        Reformatea los datos para que sean compatibles con el modelo CNN.
        """
        if len(self.batch_buffer) >= BATCH_SIZE:
            try:
                # Convertir buffer a numpy array (forma inicial: 55 muestras x 4 características)
                batch_array = np.array(self.batch_buffer[:BATCH_SIZE], dtype=np.float32)
                
                # Guardar en CSV en formato original (55 muestras x 4 características)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                csv_path = FILTERED_DATA_DIR / f"features_batch_{timestamp}.csv"
                np.savetxt(csv_path, batch_array, delimiter=',', header=','.join(COLUMNAS_SALIDA), comments='')
                
                # Reformatear para CNN: cambiar de (55, 4) a (4, 55)
                # Transponer para obtener (4, 55)
                transposed_array = batch_array.T
                
                if DEBUG_MODE:
                    # Diagnóstico de forma de datos
                    logger.info(f"Forma de batch original: {batch_array.shape}")
                    logger.info(f"Forma de batch transpuesto: {transposed_array.shape}")
                
                    # Crear versión para visualizar (1, 4, 55)
                    cnn_ready = np.expand_dims(transposed_array, axis=0)
                    logger.info(f"Forma final para CNN: {cnn_ready.shape}")
                
                    # Guardar diagnóstico
                    diag_path = DIAGNOSTICS_DIR / f"cnn_input_shape_{timestamp}.txt"
                    with open(diag_path, 'w') as f:
                        f.write(f"Forma de batch original: {batch_array.shape}\n")
                        f.write(f"Forma de batch transpuesto: {transposed_array.shape}\n")
                        f.write(f"Forma final para CNN: {cnn_ready.shape}\n")
                        f.write(f"Valores de la primera fila: {batch_array[0]}\n")
                        f.write(f"Valores de primera columna (transpuesto): {transposed_array[0][:5]}...\n")
                
                # Aplanar el array para LSL (convertir a 1D)
                # LSL no maneja arrays multidimensionales, así que aplanamos el array
                flattened_data = transposed_array.flatten()
                
                # Verificar que todos los valores sean números válidos
                for i, val in enumerate(flattened_data):
                    if not np.isfinite(val):  # Detectar NaN o Inf
                        logger.warning(f"Valor no finito detectado en posición {i}: {val}, reemplazado por 0.0")
                        flattened_data[i] = 0.0
                
                # Asegurar que tenemos el número correcto de valores
                expected_values = len(CANALES_SALIDA) * len(BANDAS_SALIDA) * BATCH_SIZE
                if len(flattened_data) != expected_values:
                    logger.warning(f"Número incorrecto de valores: {len(flattened_data)} vs {expected_values} esperados")
                    
                    # Si faltan valores, rellenar con ceros
                    if len(flattened_data) < expected_values:
                        padding = np.zeros(expected_values - len(flattened_data), dtype=np.float32)
                        flattened_data = np.concatenate([flattened_data, padding])
                    # Si hay demasiados valores, truncar
                    else:
                        flattened_data = flattened_data[:expected_values]
                
                # Convertir a lista para compatibilidad con LSL
                flattened_list = flattened_data.tolist()
                
                # Enviar todo el array aplanado de una vez
                self.outlet.push_sample(flattened_list)
                
                # Registrar éxito en log
                logger.info(f"Batch enviado correctamente y guardado: {csv_path}")
                
                # Limpiar buffer (mantener muestras extra si hay)
                self.batch_buffer = self.batch_buffer[BATCH_SIZE:]
                self.features_sent += 1
                
            except Exception as e:
                logger.error(f"Error al enviar batch: {str(e)}")
                logger.error(traceback.format_exc())
    
    def run(self):
        """Ejecuta el bucle principal con manejo robusto de errores"""
        logger.info("Iniciando procesador EEG...")
        
        while self.running:
            try:
                if not self.connected:
                    if not self.connect_to_streams():
                        logger.error("No se pudo establecer conexión, reintentando...")
                        time.sleep(RECONNECT_INTERVAL)
                        continue
                
                # Intentar recibir muestra
                try:
                    features, timestamp = self.inlet.pull_sample(timeout=1.0)
                    if features:
                        self.process_signal(features, timestamp)
                except Exception as e:
                    logger.error(f"Error al recibir muestra: {str(e)}")
                    self.connected = False
                    continue
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error en el bucle principal: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(1)
        
        logger.info("Procesador EEG detenido")

    def stop(self):
        """
        Detiene el procesamiento.
        """
        self.running = False
        logger.info("Procesador detenido")
        print("Procesador detenido")

def main():
    """
    Función principal.
    """
    try:
        # Crear directorios si no existen
        for directory in [LOGS_DIR, FILTERED_DATA_DIR, DIAGNOSTICS_DIR]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directorio creado: {directory}")
        
        # Crear procesador
        processor = EEGProcessor()
        
        # Mostrar información del sistema
        logger.info(f"Versión de Python: {sys.version}")
        logger.info(f"Versión de NumPy: {np.__version__}")
        logger.info(f"Directorio base: {BASE_DIR}")
        
        # Ejecutar procesador
        processor.run()
        
        # Asegurar que el procesador se detenga correctamente
        processor.stop()
        logger.info("Procesador detenido correctamente")
        print("Procesador detenido correctamente")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Procesador interrumpido por el usuario")
        print("Procesador interrumpido por el usuario")
        return 0
    except Exception as e:
        logger.error(f"Error en la función principal: {str(e)}")
        print(f"Error en la función principal: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())