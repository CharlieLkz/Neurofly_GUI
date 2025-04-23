#!/usr/bin/env python3
"""
TestTelloConnection.py - Módulo de conexión entre la salida de CNN y el dron DJI Tello.
Recibe predicciones de clases a través de LSL (PREDICTION_STREAM) y las traduce a comandos del dron.
"""

import socket
import time
import sys
import threading
from pylsl import StreamInlet, resolve_byprop

# Configuración global
PREDICTION_STREAM_NAME = "PREDICTION_STREAM"
TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
LOCAL_PORT = 9000
TIMEOUT_SECONDS = 15.0

class TelloController:
    def __init__(self, tello_ip=TELLO_IP, tello_port=TELLO_PORT):
        # Configuración de conexión con el dron Tello
        self.tello_address = (tello_ip, tello_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('', LOCAL_PORT))  # Puerto local para recibir respuestas
        
        # Mapeo de predicciones a comandos del dron
        self.command_map = {
            "RightArmThinking": "forward 50",
            "LeftArmThinking": "back 50",
            "RightFistThinking": "cw 90",
            "LeftFistThinking": "ccw 90",
            "RightFootThinking": "up 40",
            "LeftFootThinking": "down 40",
            "TakeoffCommand": "takeoff",
            "LandCommand": "land"
        }
        
        # Control de tiempos entre comandos
        self.last_command_time = {}
        for cmd in self.command_map.values():
            self.last_command_time[cmd] = 0
        
        # Variable para control de tiempo de inactividad
        self.last_prediction_time = time.time()
        self.running = True
        self.command_executed = False
        
        # Iniciar el temporizador de inactividad
        self.inactivity_timer = threading.Thread(target=self._check_inactivity)
        self.inactivity_timer.daemon = True
        self.inactivity_timer.start()
    
    def connect_to_drone(self):
        """Establece conexión inicial con el dron Tello."""
        print("Intentando conectar con el dron Tello...")
        
        # Configurar socket para timeout
        self.socket.settimeout(5.0)
        
        # Enviar comando de inicialización
        response = self.send_command("command")
        
        if response != "ok":
            print("ERROR: No se pudo conectar con el dron Tello.")
            print(f"Respuesta recibida: {response}")
            return False
        
        print("Conexión con el dron Tello establecida correctamente.")
        return True
    
    def send_command(self, command):
        """
        Envía un comando al dron y espera respuesta.
        
        Args:
            command (str): Comando a enviar al dron.
            
        Returns:
            str: Respuesta del dron o "timeout" si no hay respuesta.
        """
        # Enviar comando codificado en UTF-8
        print(f"Enviando comando: {command}")
        try:
            self.socket.sendto(command.encode('utf-8'), self.tello_address)
            
            # Esperar respuesta con timeout de 5 segundos
            try:
                response, _ = self.socket.recvfrom(1024)
                response = response.decode('utf-8').strip()
                print(f"Respuesta recibida: {response}")
                return response
            except socket.timeout:
                print("Timeout: No se recibió respuesta del dron.")
                return "timeout"
        except Exception as e:
            print(f"Error al enviar comando: {str(e)}")
            return f"error: {str(e)}"
    
    def execute_flight_sequence(self, command_name):
        """
        Ejecuta una secuencia de vuelo completa basada en el comando recibido.
        
        Args:
            command_name (str): Nombre del comando de predicción
        """
        # Marcar como ejecutado
        self.command_executed = True
        
        # Inicializar el dron
        if self.send_command("command") != "ok":
            print("No se pudo inicializar el dron. Abortando secuencia.")
            return False
        
        # Secuencia de despegue
        print("Ejecutando secuencia de despegue...")
        if self.send_command("takeoff") != "ok":
            print("Error en despegue. Abortando secuencia.")
            return False
        
        # Esperar estabilización
        time.sleep(3)
        
        # Obtener el comando específico del movimiento
        if command_name in self.command_map:
            specific_command = self.command_map[command_name]
            print(f"Ejecutando comando específico: {specific_command}")
            if self.send_command(specific_command) != "ok":
                print(f"Error al ejecutar {specific_command}. Continuando con aterrizaje.")
        else:
            print(f"Comando desconocido: {command_name}. Ejecutando solo despegue y aterrizaje.")
        
        # Esperar antes de aterrizar
        time.sleep(2)
        
        # Aterrizar
        print("Ejecutando aterrizaje...")
        if self.send_command("land") != "ok":
            print("Error en aterrizaje. La secuencia no se completó correctamente.")
            return False
        
        print("Secuencia de vuelo completada exitosamente!")
        return True
    
    def process_prediction(self, prediction, feature_idx):
        """
        Procesa la predicción recibida y ejecuta la secuencia de vuelo.
        
        Args:
            prediction (str): Clase predicha por la CNN
            feature_idx (str): Índice de la feature más relevante
        """
        # Actualizar tiempo de última predicción
        self.last_prediction_time = time.time()
        
        print(f"Predicción recibida: {prediction}")
        print(f"Feature más relevante: {feature_idx}")
        
        # Verificar si ya hemos ejecutado un comando
        if self.command_executed:
            print("Ya se ha ejecutado un comando. Ignorando predicción adicional.")
            return
        
        # Verificar si la predicción está en el mapeo de comandos
        if prediction in self.command_map:
            print(f"Ejecutando secuencia de vuelo para: {prediction}")
            self.execute_flight_sequence(prediction)
            
            # Finalizar procesamiento tras ejecutar un comando
            print("Comando ejecutado. Finalizando el programa...")
            self.running = False
        else:
            print(f"Predicción no reconocida: {prediction}")
    
    def _check_inactivity(self):
        """
        Verifica si ha pasado demasiado tiempo sin recibir predicciones.
        Termina el programa después de TIMEOUT_SECONDS segundos de inactividad.
        """
        while self.running and not self.command_executed:
            time.sleep(1)
            if time.time() - self.last_prediction_time > TIMEOUT_SECONDS:
                print(f"\nNo se han recibido predicciones en {TIMEOUT_SECONDS} segundos.")
                print("Cerrando el programa...")
                self.running = False
                sys.exit(0)
    
    def close(self):
        """Cierra la conexión con el dron."""
        self.running = False
        self.socket.close()
        print("Conexión con el dron cerrada.")


def main():
    # Inicializar controlador del dron
    tello = TelloController()
    
    # Intentar conectar con el dron
    if not tello.connect_to_drone():
        print("No se pudo establecer conexión con el dron. Saliendo...")
        sys.exit(1)
    
    # Buscar el stream LSL de predicciones
    print(f"Buscando stream LSL '{PREDICTION_STREAM_NAME}'...")
    
    # Variable para controlar intentos de conexión
    connection_attempts = 0
    max_connection_attempts = 30  # 30 segundos
    
    # Buscar stream con reintentos
    while connection_attempts < max_connection_attempts:
        try:
            streams = resolve_byprop('name', PREDICTION_STREAM_NAME, timeout=1.0)
            if streams:
                break
            
            print(f"Stream '{PREDICTION_STREAM_NAME}' no encontrado. Reintentando ({connection_attempts+1}/{max_connection_attempts})...")
            connection_attempts += 1
            time.sleep(1)
        except Exception as e:
            print(f"Error al buscar streams: {str(e)}")
            connection_attempts += 1
            time.sleep(1)
    
    if not streams:
        print(f"ERROR: No se encontró el stream '{PREDICTION_STREAM_NAME}' después de {max_connection_attempts} intentos.")
        print("Asegúrate que cnn_inference.py esté ejecutándose.")
        tello.close()
        sys.exit(1)
    
    # Crear inlet para recibir datos del stream
    inlet = StreamInlet(streams[0])
    print(f"Conectado al stream: {streams[0].name()}")
    
    try:
        print("Esperando predicciones (presiona Ctrl+C para salir)...")
        start_time = time.time()
        
        while tello.running:
            # Verificar timeout
            if time.time() - start_time > TIMEOUT_SECONDS and not tello.command_executed:
                print(f"No se han recibido predicciones en {TIMEOUT_SECONDS} segundos.")
                print("Cerrando el programa...")
                break
            
            # Obtener la siguiente predicción con timeout (para permitir interrupciones)
            sample, timestamp = inlet.pull_sample(timeout=0.5)
            
            if sample:
                start_time = time.time()  # Reiniciar timer al recibir datos
                
                # Procesar la predicción (será un string en el primer elemento)
                prediction = sample[0]  # Clase predicha
                
                # El segundo valor es el índice de la feature más relevante
                feature_idx = "-1"
                if len(sample) > 1 and sample[1]:
                    feature_idx = sample[1]
                
                # Procesar la predicción
                tello.process_prediction(prediction, feature_idx)
    
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario.")
    finally:
        # Cerrar conexiones
        tello.close()
        print("Programa finalizado.")


if __name__ == "__main__":
    main()