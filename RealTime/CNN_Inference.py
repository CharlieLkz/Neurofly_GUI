#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN_Inference.py
Módulo de inferencia en tiempo real:
  - Busca el .pt con mayor accuracy en CNNModels/models/
  - Carga el modelo de forma robusta
  - Escucha FEATURE_STREAM (LSL) y procesa muestras [1, 20, 17]
  - Realiza inferencia y muestra predicciones en consola
"""

import os
import re
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
from datetime import datetime
import traceback

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cnn_inference_log.txt', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Configuración de streams LSL
INPUT_STREAM_NAME = "FEATURE_STREAM"

# Clases válidas para inferencia
VALID_CLASSES = [
    'RightArmThinking',
    'LeftArmThinking',
    'RightFistThinking',
    'LeftFistThinking'
]

class RobustEEGCNNMulticlass(nn.Module):
    """Versión robusta del modelo EEGCNNMulticlass que maneja diferencias estructurales"""
    def __init__(self, in_channels=4, n_classes=4, config=None):
        super(RobustEEGCNNMulticlass, self).__init__()
        
        # Configuración por defecto
        if config is None:
            config = {
                'hidden_dim': 384,  # Ajustado para coincidir con el modelo entrenado
                'channels': [64, 128, 192],  # Canales ajustados para coincidir
                'dropout_rate': 0.25,
                'attention_mechanism': True
            }
        
        # Parámetros de la arquitectura
        self.in_channels = in_channels  # 4 canales de entrada
        self.n_classes = n_classes
        self.hidden_dim = config.get('hidden_dim', 384)  # Ajustado
        self.channels = config.get('channels', [64, 128, 192])  # Ajustado
        self.dropout_rate = config.get('dropout_rate', 0.25)
        self.use_attention = config.get('attention_mechanism', True)
        
        # Capas convolucionales con BatchNorm
        # Input: (batch_size, 4, 55)
        self.conv1 = nn.Conv1d(self.in_channels, self.channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.channels[0])
        
        self.conv2 = nn.Conv1d(self.channels[0], self.channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.channels[1])
        
        self.conv3 = nn.Conv1d(self.channels[1], self.channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(self.channels[2])
        
        # Mecanismo de atención (opcional)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv1d(self.channels[2], self.channels[2] // 4, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(self.channels[2] // 4, self.channels[2], kernel_size=1),
                nn.Sigmoid()
            )
        
        # Calcular tamaño de entrada para fc1
        # Después de 3 capas de maxpool: 55 -> 27 -> 13 -> 6
        self.fc1_in_features = self.channels[2] * 6  # 192 * 6 = 1152
        
        # Capas fully connected con BatchNorm
        self.fc1 = nn.Linear(self.fc1_in_features, self.hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.n_classes)
        
        # Dropout con tasas diferentes
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate + 0.05)
    
    def forward(self, x):
        # Asegurar que la entrada tiene la forma correcta (batch, channels, sequence)
        if x.shape[1] == 55 and x.shape[2] == 4:
            x = x.transpose(1, 2)  # Cambiar de (batch, 55, 4) a (batch, 4, 55)
        
        # Primera capa convolucional
        x = self.conv1(x)  # (batch, 64, 55)
        x = self.bn1(x)
        x = torch.relu(x)
        x = torch.max_pool1d(x, 2)  # (batch, 64, 27)
        x = self.dropout1(x)
        
        # Segunda capa convolucional
        x = self.conv2(x)  # (batch, 128, 27)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.max_pool1d(x, 2)  # (batch, 128, 13)
        x = self.dropout1(x)
        
        # Tercera capa convolucional
        x = self.conv3(x)  # (batch, 192, 13)
        x = self.bn3(x)
        x = torch.relu(x)
        x = torch.max_pool1d(x, 2)  # (batch, 192, 6)
        x = self.dropout1(x)
        
        # Mecanismo de atención
        if self.use_attention and hasattr(self, 'attention'):
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Aplanar para capas fully connected
        x = x.view(x.size(0), -1)  # (batch, 192 * 6 = 1152)
        
        # Capas fully connected
        x = self.fc1(x)  # (batch, 384)
        x = self.bn_fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)  # (batch, 192)
        x = self.bn_fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # (batch, n_classes)
        
        return x
    
    def set_inference_mode(self):
        """Configura el modelo específicamente para inferencia"""
        self.eval()
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
                module.track_running_stats = False
            elif isinstance(module, nn.Dropout):
                module.eval()

class CNNInference:
    def __init__(self):
        logger.info("=" * 50)
        logger.info("Iniciando CNNInference...")
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Dispositivo seleccionado: {self.device}")
        
        # Configuración por defecto del modelo actual
        self.default_config = {
            'in_channels': 4,
            'n_classes': 4,
            'hidden_dim': 384,
            'channels': [64, 128, 192],
            'dropout_rate': 0.25,
            'attention_mechanism': True
        }
        
        self.class_names = VALID_CLASSES
        self.running = True
        
        # Variables para el control de comandos
        self.last_command = None
        self.last_command_time = 0
        self.command_cooldown = 2.0  # 2 segundos de espera entre comandos
        self.command_outlet = None
        self.status_outlet = None
        
        # Métricas y monitoreo
        self.processed_samples = 0
        self.prediction_stats = {
            'total': 0,
            'correct': 0,
            'confidence_sum': 0.0,
            'latency_sum': 0.0,
            'min_latency': float('inf'),
            'max_latency': float('-inf')
        }
        
        # Configurar directorios
        self.base_dir = Path(__file__).parent.resolve()
        self.models_dir = self.base_dir / 'CNNModels' / 'models'
        
        if not self.models_dir.exists():
            raise FileNotFoundError(f"No se encuentra el directorio de modelos: {self.models_dir}")
        
        # Buscar y cargar el mejor modelo
        self.model_path = self.find_best_model()
        if self.model_path:
            logger.info(f"Modelo seleccionado: {self.model_path.name}")
            self.load_model(self.model_path)
        else:
            raise FileNotFoundError("No se encontró ningún modelo válido")
        
        # Crear outlets para comandos y estado
        self._create_outlets()
    
    def _create_outlets(self):
        """Crea los streams LSL para comandos y estado"""
        try:
            # Stream para comandos
            command_info = StreamInfo(
                name='CNN_COMMANDS',
                type='Commands',
                channel_count=1,
                nominal_srate=0,
                channel_format='string',
                source_id='CNNInference'
            )
            self.command_outlet = StreamOutlet(command_info)
            logger.info("Stream de comandos CNN_COMMANDS creado")
            
            # Stream para estado
            status_info = StreamInfo(
                name='CNN_STATUS',
                type='Status',
                channel_count=2,
                nominal_srate=0,
                channel_format='string',
                source_id='CNNInference'
            )
            self.status_outlet = StreamOutlet(status_info)
            logger.info("Stream de estado CNN_STATUS creado")
        except Exception as e:
            logger.error(f"Error al crear streams LSL: {str(e)}")
            raise
    
    def _update_status(self, status, time_remaining):
        """Actualiza el estado en el stream LSL"""
        try:
            if self.status_outlet:
                self.status_outlet.push_sample([status, str(time_remaining)])
        except Exception as e:
            logger.error(f"Error al actualizar estado: {str(e)}")
    
    def _can_send_command(self, new_command):
        """Verifica si se puede enviar un nuevo comando"""
        current_time = time.time()
        time_since_last = current_time - self.last_command_time
        
        # Si es el mismo comando y no ha pasado el cooldown, no enviar
        if new_command == self.last_command and time_since_last < self.command_cooldown:
            time_remaining = self.command_cooldown - time_since_last
            self._update_status("cooldown", f"{time_remaining:.1f}")
            return False
        
        # Si ha pasado el cooldown o es un comando diferente, se puede enviar
        if time_since_last >= self.command_cooldown or new_command != self.last_command:
            self._update_status("ready", "0.0")
            return True
        
        return False
    
    def find_best_model(self):
        """Encuentra el modelo con mayor accuracy en el directorio de modelos"""
        best_acc = -1.0
        best_path = None
        
        for pt in self.models_dir.glob("*.pt"):
            # Extraer accuracy del nombre del archivo
            nums = re.findall(r"(\d+\.\d+)", pt.name)
            floats = [float(n) for n in nums if "." in n]
            if not floats:
                continue
            acc = max(floats)
            if acc > best_acc:
                best_acc = acc
                best_path = pt
        
        if best_path:
            logger.info(f"Modelo seleccionado: {best_path.name} (accuracy={best_acc:.4f})")
        return best_path
    
    def analyze_checkpoint(self, checkpoint):
        """Analiza el checkpoint y extrae/valida la configuración"""
        config = {}
        
        # Si es solo state_dict, intentar inferir configuración de los pesos
        if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
            state_dict = checkpoint
            logger.info("Checkpoint contiene solo state_dict, inferiendo configuración...")
            
            # Inferir n_classes de la última capa
            if 'fc3.weight' in state_dict:
                n_classes = state_dict['fc3.weight'].shape[0]
                config['n_classes'] = n_classes
                logger.info(f"Inferido n_classes={n_classes} de fc3.weight")
            
            # Inferir in_channels de la primera capa conv
            if 'conv1.weight' in state_dict:
                in_channels = state_dict['conv1.weight'].shape[1]
                config['in_channels'] = in_channels
                logger.info(f"Inferido in_channels={in_channels} de conv1.weight")
            
            return config, state_dict
        
        # Si es un checkpoint completo, extraer configuración
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        saved_config = checkpoint.get('config', {})
        
        # Combinar con valores por defecto
        config = self.default_config.copy()
        config.update(saved_config)
        
        return config, state_dict

    def load_model(self, model_path):
        """Carga el modelo de forma robusta, manejando diferencias estructurales"""
        try:
            logger.info(f"Intentando cargar modelo desde: {model_path}")
            
            # Verificar archivo
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encuentra el archivo del modelo: {model_path}")
            
            # Cargar checkpoint
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                logger.info("Checkpoint cargado exitosamente")
            except Exception as e:
                logger.warning(f"Error al cargar checkpoint completo: {str(e)}")
                logger.info("Intentando cargar solo los pesos...")
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    logger.error("No se pudo cargar el checkpoint ni como pesos")
                    raise
            
            # Analizar checkpoint y extraer configuración
            config, state_dict = self.analyze_checkpoint(checkpoint)
            
            # Validar configuración
            if config.get('in_channels', 4) != 4:
                logger.error(f"in_channels en checkpoint ({config['in_channels']}) != 4 - Modelo incompatible")
                raise ValueError("Modelo incompatible: número de canales de entrada incorrecto")
            
            if config.get('n_classes', 4) != 4:
                logger.error(f"n_classes en checkpoint ({config['n_classes']}) != 4 - Modelo incompatible")
                raise ValueError("Modelo incompatible: número de clases incorrecto")
            
            # Crear modelo con configuración ajustada
            self.model = RobustEEGCNNMulticlass(
                in_channels=config['in_channels'],
                n_classes=config['n_classes'],
                config=config
            ).to(self.device)
            
            # Validar compatibilidad de pesos
            model_dict = self.model.state_dict()
            incompatible_layers = []
            
            for key, value in state_dict.items():
                if key in model_dict:
                    if model_dict[key].shape != value.shape:
                        incompatible_layers.append(f"{key}: checkpoint={value.shape}, modelo={model_dict[key].shape}")
            
            if incompatible_layers:
                logger.error("\nCapas incompatibles detectadas:")
                for layer in incompatible_layers:
                    logger.error(f"  [X] {layer}")
                raise ValueError("Modelo incompatible: dimensiones de capas no coinciden")
            
            # Cargar pesos
            self.model.load_state_dict(state_dict, strict=True)
            logger.info("\nModelo cargado exitosamente")
            
            # Verificar funcionalidad
            self.model.set_inference_mode()
            test_input = torch.randn(1, 4, 55).to(self.device)
            
            try:
                with torch.no_grad():
                    output = self.model(test_input)
                    if output.shape[1] != 4:
                        raise ValueError(f"Salida del modelo tiene forma incorrecta: {output.shape}")
                logger.info("\nModelo verificado y listo para inferencia")
                logger.info(f"Forma de entrada: {test_input.shape}")
                logger.info(f"Forma de salida: {output.shape}")
            except Exception as e:
                logger.error(f"Error al verificar el modelo: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            logger.error("\nSugerencias para resolver el problema:")
            logger.error("1. Verifica que el checkpoint es compatible con la arquitectura actual")
            logger.error("2. Asegúrate de que el modelo fue guardado correctamente")
            logger.error("3. Considera reentrenar el modelo con la arquitectura actual")
            raise
    
    def _validate_input(self, features):
        """Valida la forma y tipo de los datos de entrada"""
        try:
            # Convertir a numpy array si no lo es
            features = np.array(features, dtype=np.float32)
            
            # Verificar forma
            if features.shape != (4, 55):
                logger.error(f"Forma de entrada inválida: {features.shape}, esperado (4, 55)")
                return None
            
            # Verificar valores
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.error("Datos de entrada contienen NaN o Inf")
                return None
            
            # Normalizar si es necesario
            if np.max(np.abs(features)) > 1.0:
                features = features / np.max(np.abs(features))
            
            return features
            
        except Exception as e:
            logger.error(f"Error en validación de entrada: {str(e)}")
            return None
    
    def predict(self, features):
        """Realiza la predicción con validación y monitoreo de rendimiento"""
        try:
            start_time = time.time()
            
            # Validar entrada
            features = self._validate_input(features)
            if features is None:
                return None, 0.0
            
            # Convertir a tensor
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Calcular latencia
            latency = time.time() - start_time
            self._update_prediction_stats(latency)
            
            # Obtener nombre de la clase
            predicted_class = self.class_names[predicted.item()]
            confidence_value = confidence.item()
            
            # Log de predicción
            logger.info(f"Predicción: {predicted_class} ({confidence_value:.2%}) - Latencia: {latency:.3f}s")
            
            return predicted_class, confidence_value
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            logger.error(traceback.format_exc())
            return None, 0.0
    
    def _update_prediction_stats(self, latency):
        """Actualiza las estadísticas de predicción"""
        self.prediction_stats['total'] += 1
        self.prediction_stats['latency_sum'] += latency
        self.prediction_stats['min_latency'] = min(self.prediction_stats['min_latency'], latency)
        self.prediction_stats['max_latency'] = max(self.prediction_stats['max_latency'], latency)
        
        # Log periódico de estadísticas
        if self.prediction_stats['total'] % 100 == 0:
            avg_latency = self.prediction_stats['latency_sum'] / self.prediction_stats['total']
            logger.info(f"Estadísticas de predicción:")
            logger.info(f"  Total: {self.prediction_stats['total']}")
            logger.info(f"  Latencia: min={self.prediction_stats['min_latency']:.3f}s, max={self.prediction_stats['max_latency']:.3f}s, avg={avg_latency:.3f}s")
    
    def run(self):
        """Ejecuta el bucle principal de inferencia con manejo robusto de errores"""
        logger.info("Esperando FEATURE_STREAM...")
        
        while self.running:
            try:
                # Buscar stream
                stream = resolve_byprop('name', INPUT_STREAM_NAME, timeout=5.0)
                if not stream:
                    logger.warning(f"No se encontró el stream '{INPUT_STREAM_NAME}', reintentando...")
                    time.sleep(1)
                    continue
                
                # Conectar al stream
                inlet = StreamInlet(stream[0])
                logger.info("Streaming iniciado")
                
                # Bucle principal
                while self.running:
                    try:
                        features, timestamp = inlet.pull_sample(timeout=1.0)
                        if features:
                            # Realizar predicción
                            predicted_class, confidence = self.predict(features)
                            if predicted_class:
                                # Verificar si se puede enviar el comando
                                if self._can_send_command(predicted_class):
                                    try:
                                        # Enviar comando
                                        self.command_outlet.push_sample([predicted_class])
                                        logger.info(f"Comando enviado: {predicted_class}")
                                        
                                        # Actualizar estado
                                        self.last_command = predicted_class
                                        self.last_command_time = time.time()
                                        self._update_status("sent", str(self.command_cooldown))
                                    except Exception as e:
                                        logger.error(f"Error al enviar comando: {str(e)}")
                        
                    except Exception as e:
                        logger.error(f"Error en el bucle principal: {str(e)}")
                        logger.error(traceback.format_exc())
                        break
                
                # Cerrar stream
                inlet.close_stream()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error en la conexión: {str(e)}")
                time.sleep(1)
        
        logger.info("Streaming detenido")
    
    def stop(self):
        """Detiene el procesamiento"""
        self.running = False
        logger.info("CNN Inference detenido")
        
        # Mostrar estadísticas finales
        if self.prediction_stats['total'] > 0:
            avg_latency = self.prediction_stats['latency_sum'] / self.prediction_stats['total']
            logger.info("\nEstadísticas finales:")
            logger.info(f"  Total de predicciones: {self.prediction_stats['total']}")
            logger.info(f"  Latencia promedio: {avg_latency:.3f}s")
            logger.info(f"  Latencia mínima: {self.prediction_stats['min_latency']:.3f}s")
            logger.info(f"  Latencia máxima: {self.prediction_stats['max_latency']:.3f}s")

def main():
    """Función principal"""
    try:
        logger.info("Iniciando CNN_Inference...")
        inference = CNNInference()
        inference.run()
    except Exception as e:
        logger.error(f"Error general: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Programa terminado")

if __name__ == "__main__":
    main()