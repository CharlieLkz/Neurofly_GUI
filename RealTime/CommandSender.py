#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CommandSender.py
Escucha CNN_COMMANDS (LSL) y envía al dron Tello el comando asociado a cada clase válida.
Guarda logs en logs/command_sender_log.txt
"""

import socket
import time
import logging
from pathlib import Path
import pylsl
import threading
import queue
import sys
from pylsl import StreamInlet, resolve_byprop
from djitellopy import Tello
import traceback

# === Configuración de rutas y logs === #
BASE_DIR = Path(__file__).parent.resolve()
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "command_sender_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

# === Configuración del dron Tello === #
TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)
LOCAL_PORT = 9000

# Timeout para aterrizaje automático (15 segundos)
INACTIVITY_TIMEOUT = 15.0

# === Clases válidas y mapeo a comandos === #
COMMAND_MAP = {
    'RightArmThinking': 'takeoff',
    'LeftArmThinking': 'flip_back',
    'RightFistThinking': 'forward 30',
    'LeftFistThinking': 'land',
    'StartSequence': 'sequence'  # Nueva clase para iniciar la secuencia completa
}

# Estado del drone
drone_state = {
    'is_flying': False,
    'last_command': None,
    'last_command_time': 0,
    'in_sequence': False
}

class DroneController:
    def __init__(self):
        self.drone = Tello()
        self.running = True
        self.max_retries = 3
        self.retry_delay = 2  # segundos entre reintentos
        self.connected = False
        self.last_activity = time.time()
        self.inactivity_monitor = None
        self.command_queue = queue.Queue()
        self.command_thread = None
        
        # Estadísticas
        self.stats = {
            'commands_sent': 0,
            'commands_failed': 0,
            'battery_levels': [],
            'latency_sum': 0.0,
            'min_latency': float('inf'),
            'max_latency': float('-inf')
        }
    
    def connect(self):
        """Intenta conectar con el dron con reintentos y validación"""
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Intento de conexión {attempt + 1}/{self.max_retries}")
                
                # Intentar conectar
                self.drone.connect()
                
                # Verificar batería
                battery = self.drone.get_battery()
                logging.info(f"Nivel de batería: {battery}%")
                if battery < 10:
                    logging.error("Batería demasiado baja para operar")
                    return False
                
                # Activar modo SDK y verificar respuesta
                response = self.drone.send_command_with_return("command")
                if response != 'ok':
                    logging.error("No se pudo activar el modo SDK")
                    if attempt < self.max_retries - 1:
                        logging.info(f"Reintentando en {self.retry_delay} segundos...")
                        time.sleep(self.retry_delay)
                        continue
                    return False
                
                # Verificación adicional de conectividad
                try:
                    # Intentar obtener altura (comando simple que requiere respuesta)
                    height = self.drone.get_height()
                    if height is None:
                        raise Exception("No se pudo obtener altura del dron")
                    logging.info(f"Altura actual: {height}cm")
                except Exception as e:
                    logging.error(f"Error en verificación de conectividad: {str(e)}")
                    if attempt < self.max_retries - 1:
                        logging.info(f"Reintentando en {self.retry_delay} segundos...")
                        time.sleep(self.retry_delay)
                        continue
                    return False
                
                self.connected = True
                logging.info("Conexión exitosa con el dron")
                
                # Iniciar monitoreo de inactividad
                self.inactivity_monitor = threading.Thread(target=self.monitor_inactivity, daemon=True)
                self.inactivity_monitor.start()
                
                # Iniciar thread de procesamiento de comandos
                self.command_thread = threading.Thread(target=self._process_commands, daemon=True)
                self.command_thread.start()
                
                return True
                
            except Exception as e:
                logging.error(f"Error al conectar: {str(e)}")
                if attempt < self.max_retries - 1:
                    logging.info(f"Reintentando en {self.retry_delay} segundos...")
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Se agotaron los intentos de conexión")
                    return False
        
        return False
    
    def _process_commands(self):
        """Procesa los comandos de la cola"""
        while self.running:
            try:
                command = self.command_queue.get(timeout=1.0)
                if command:
                    start_time = time.time()
                    
                    # Verificar batería antes de cada comando
                    battery = self.drone.get_battery()
                    self.stats['battery_levels'].append(battery)
                    
                    if battery < 10:
                        logging.error("Batería demasiado baja para ejecutar comandos")
                        self.stats['commands_failed'] += 1
                        continue
                    
                    # Ejecutar comando
                    response = self.send_command(command)
                    
                    # Actualizar estadísticas
                    latency = time.time() - start_time
                    self.stats['commands_sent'] += 1
                    self.stats['latency_sum'] += latency
                    self.stats['min_latency'] = min(self.stats['min_latency'], latency)
                    self.stats['max_latency'] = max(self.stats['max_latency'], latency)
                    
                    # Log periódico de estadísticas
                    if self.stats['commands_sent'] % 10 == 0:
                        self._log_stats()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error al procesar comando: {str(e)}")
                self.stats['commands_failed'] += 1
    
    def _log_stats(self):
        """Registra las estadísticas actuales"""
        if self.stats['commands_sent'] > 0:
            avg_latency = self.stats['latency_sum'] / self.stats['commands_sent']
            avg_battery = sum(self.stats['battery_levels']) / len(self.stats['battery_levels'])
            
            logging.info("\nEstadísticas de comandos:")
            logging.info(f"  Comandos enviados: {self.stats['commands_sent']}")
            logging.info(f"  Comandos fallidos: {self.stats['commands_failed']}")
            logging.info(f"  Latencia promedio: {avg_latency:.3f}s")
            logging.info(f"  Latencia mínima: {self.stats['min_latency']:.3f}s")
            logging.info(f"  Latencia máxima: {self.stats['max_latency']:.3f}s")
            logging.info(f"  Batería promedio: {avg_battery:.1f}%")
    
    def send_command(self, command, wait=0):
        """Envía un comando al dron con verificación"""
        if not self.connected:
            logging.error("No hay conexión con el dron")
            return 'error'
        
        try:
            # Actualizar última actividad
            self.last_activity = time.time()
            
            # Verificar estado de conexión antes de enviar
            battery = self.drone.get_battery()
            if battery < 10:
                logging.error("Batería demasiado baja para ejecutar comandos")
                return 'error'
            
            # Enviar comando
            response = self.drone.send_command_with_return(command)
            if response != 'ok':
                logging.error(f"Error al ejecutar comando {command}: {response}")
            else:
                logging.info(f"Comando ejecutado: {command}")
                time.sleep(wait)
            return response
            
        except Exception as e:
            logging.error(f"Error al enviar comando: {str(e)}")
            return 'error'
    
    def queue_command(self, command):
        """Añade un comando a la cola de procesamiento"""
        if not self.connected:
            logging.error("No hay conexión con el dron")
            return False
        
        try:
            self.command_queue.put(command)
            return True
        except Exception as e:
            logging.error(f"Error al encolar comando: {str(e)}")
            return False
    
    def monitor_inactivity(self):
        """Monitor inactivity and land if timeout is reached"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_activity > INACTIVITY_TIMEOUT:
                logging.warning("\n¡Timeout de inactividad detectado!")
                logging.warning("Iniciando aterrizaje automático...")
                try:
                    self.send_command('land', wait=4)
                except:
                    pass
                logging.warning("Programa terminado por inactividad")
                sys.exit(0)
            time.sleep(1)
    
    def execute_sequence(self):
        """Ejecuta la secuencia completa de vuelo"""
        if drone_state['in_sequence']:
            logging.warning("Secuencia ya en progreso")
            return
        
        drone_state['in_sequence'] = True
        try:
            # Takeoff
            logging.info("Taking off...")
            response = self.send_command('takeoff', wait=10)
            if response != 'ok':
                raise Exception("Takeoff failed")
            time.sleep(10)
            
            # Ascend 30cm
            logging.info("Ascending 30cm...")
            response = self.send_command('up 30', wait=10)
            if response != 'ok':
                raise Exception("Ascend failed")
            time.sleep(10)
            
            # Move forward
            logging.info("Moving forward...")
            response = self.send_command('forward 50', wait=10)
            if response != 'ok':
                raise Exception("Forward movement failed")
            time.sleep(10)
            
            # Forward flip
            logging.info("Performing forward flip...")
            response = self.send_command('flip f', wait=10)
            if response != 'ok':
                raise Exception("Forward flip failed")
            time.sleep(10)
            
            # Backward flip
            logging.info("Performing backward flip...")
            response = self.send_command('flip b', wait=10)
            if response != 'ok':
                raise Exception("Backward flip failed")
            time.sleep(10)
            
            # Land
            logging.info("Landing...")
            self.send_command('land', wait=10)
            logging.info("Flight sequence completed successfully")
            
        except Exception as e:
            logging.error(f"Error en la secuencia: {e}")
            self.send_command('land', wait=10)
        finally:
            drone_state['in_sequence'] = False
    
    def stop(self):
        """Detiene el controlador y aterriza el dron si está en vuelo"""
        self.running = False
        if self.connected:
            try:
                self.drone.land()
            except:
                pass
            self.connected = False
        
        # Mostrar estadísticas finales
        self._log_stats()

def main():
    """Función principal con manejo mejorado de errores"""
    try:
        # Crear controlador del drone
        drone = DroneController()
        
        # Intentar conectar con el drone
        if not drone.connect():
            logging.error("No se pudo establecer conexión con el drone después de 3 intentos")
            logging.error("Por favor verifica que:")
            logging.error("1. El dron está encendido")
            logging.error("2. El dron está en modo SDK")
            logging.error("3. La batería tiene suficiente carga (>10%)")
            logging.error("4. El dron está conectado a la red WiFi correcta")
            return 1
        
        # Buscar los streams necesarios
        command_stream = None
        emergency_stream = None
        
        # Buscar stream de comandos
        while drone.running:
            try:
                streams = resolve_byprop('name', 'CNN_COMMANDS')
                if streams:
                    command_stream = StreamInlet(streams[0])
                    logging.info("Conectado al stream CNN_COMMANDS")
                    break
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error al buscar stream de comandos: {str(e)}")
                time.sleep(2)
        
        # Buscar stream de emergencia
        try:
            emergency_streams = resolve_byprop('name', 'EMERGENCY_STOP')
            if emergency_streams:
                emergency_stream = StreamInlet(emergency_streams[0])
                logging.info("Conectado al stream de emergencia")
        except Exception as e:
            logging.warning(f"No se pudo conectar al stream de emergencia: {str(e)}")
        
        if not command_stream:
            logging.error("No se encontró el stream CNN_COMMANDS")
            return
        
        # Bucle principal
        while drone.running:
            try:
                # Recibir comando
                command, timestamp = command_stream.pull_sample(timeout=1.0)
                if command:
                    # Verificar si es un comando válido
                    if command[0] in COMMAND_MAP:
                        # Obtener comando del dron
                        drone_command = COMMAND_MAP[command[0]]
                        
                        # Encolar comando
                        if not drone.queue_command(drone_command):
                            logging.error(f"Error al encolar comando: {drone_command}")
                    
                    elif command[0] == 'StartSequence':
                        # Ejecutar secuencia en un thread separado
                        threading.Thread(target=drone.execute_sequence, daemon=True).start()
                    
                    else:
                        logging.warning(f"Comando desconocido: {command[0]}")
                
            except Exception as e:
                logging.error(f"Error en el bucle principal: {str(e)}")
                time.sleep(1)
        
    except KeyboardInterrupt:
        logging.info("Programa interrumpido por el usuario")
        return 0
    except Exception as e:
        logging.error(f"Error fatal: {str(e)}")
        return 1
    finally:
        if 'drone' in locals():
            drone.stop()
        logging.info("Programo terminado")

if __name__ == "__main__":
    main()
