#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CommandSender para Control de ESP32 basado en Clasificación EEG
===============================================================

Este script implementa una interfaz gráfica que:
1. Se conecta al stream LSL "CNN_COMMANDS" para recibir predicciones
2. Permite al usuario seleccionar una clase objetivo (ej. "RightFistThinking")
3. Cuando la clase detectada coincide con la objetivo, envía un comando 
   a un ESP32 para controlar su LED integrado.

Uso:
    python CommandSender.py
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, StringVar
import json
import logging
import serial
import serial.tools.list_ports
from pylsl import StreamInlet, resolve_byprop

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("command_sender_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CommandSender")

# === CONFIGURACIONES === #
# Nombre del stream LSL de entrada (comandos desde CNN)
INPUT_STREAM_NAME = "CNN_COMMANDS"

# Configuración para puertos seriales
DEFAULT_PORT_WINDOWS = "COM3"
DEFAULT_PORT_UNIX = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 9600

# Tiempo máximo de espera para la búsqueda de streams LSL
LSL_SEARCH_TIMEOUT = 10  # segundos

# Tiempo entre comandos para evitar saturación
COMMAND_COOLDOWN = 1.0  # segundos

# Colores para la interfaz
COLOR_NORMAL = "#f0f0f0"      # Gris claro para estado normal
COLOR_CONNECTED = "#d0f0d0"   # Verde claro para estado conectado
COLOR_DISCONNECTED = "#f0d0d0"  # Rojo claro para desconectado
COLOR_MATCH = "#90ee90"       # Verde intenso para coincidencia
COLOR_ACTIVE = "#add8e6"      # Azul claro para activo

class ESPController:
    """
    Clase para manejar la comunicación con el ESP32 vía puerto serial.
    """
    def __init__(self):
        self.serial = None
        self.port = None
        self.baudrate = DEFAULT_BAUDRATE
        self.connected = False
        self.last_command_time = 0
    
    def list_ports(self):
        """
        Lista los puertos seriales disponibles.
        
        Returns:
            list: Lista de puertos disponibles
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(port.device)
        return ports
    
    def connect(self, port=None, baudrate=DEFAULT_BAUDRATE):
        """
        Conecta al ESP32 a través del puerto serial.
        
        Args:
            port: Puerto serial a utilizar
            baudrate: Velocidad de comunicación
            
        Returns:
            bool: True si la conexión fue exitosa
        """
        if self.connected and self.serial:
            self.disconnect()
        
        # Si no se especifica puerto, detectar automáticamente
        if port is None:
            if sys.platform.startswith('win'):
                port = DEFAULT_PORT_WINDOWS
            else:
                port = DEFAULT_PORT_UNIX
        
        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            self.port = port
            self.baudrate = baudrate
            self.connected = True
            
            # Esperar a que el ESP32 se reinicie (típico tras conexión serial)
            time.sleep(2)
            
            # Limpiar buffer
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Verificar conexión enviando un comando de prueba
            self.send_command("TEST")
            response = self.read_response(timeout=2)
            
            logger.info(f"Conectado a ESP32 en {port} a {baudrate} baudios")
            return True
            
        except Exception as e:
            logger.error(f"Error al conectar con ESP32: {str(e)}")
            if self.serial:
                try:
                    self.serial.close()
                except:
                    pass
                self.serial = None
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Desconecta del ESP32.
        
        Returns:
            bool: True si la desconexión fue exitosa
        """
        try:
            if self.serial:
                # Enviar señal de apagado antes de desconectar
                try:
                    self.send_command("OFF")
                except:
                    pass
                    
                self.serial.close()
                self.serial = None
            self.connected = False
            logger.info("Desconectado de ESP32")
            return True
        except Exception as e:
            logger.error(f"Error al desconectar de ESP32: {str(e)}")
            return False
    
    def send_command(self, command):
        """
        Envía un comando al ESP32.
        
        Args:
            command: Comando a enviar (ON, OFF, TOGGLE)
            
        Returns:
            bool: True si el comando se envió correctamente
        """
        if not self.connected or not self.serial:
            logger.warning("No hay conexión con ESP32")
            return False
        
        # Aplicar cooldown para evitar saturación
        current_time = time.time()
        if current_time - self.last_command_time < COMMAND_COOLDOWN:
            # No ha pasado suficiente tiempo desde el último comando
            return False
        
        try:
            # Asegurar que el comando termina con salto de línea
            if not command.endswith('\n'):
                command += '\n'
            
            # Enviar comando
            self.serial.write(command.encode('utf-8'))
            self.serial.flush()
            
            self.last_command_time = current_time
            logger.info(f"Comando enviado: {command.strip()}")
            return True
        except Exception as e:
            logger.error(f"Error al enviar comando: {str(e)}")
            return False
    
    def read_response(self, timeout=1.0):
        """
        Lee la respuesta del ESP32.
        
        Args:
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            str: Respuesta recibida o None si no hay respuesta
        """
        if not self.connected or not self.serial:
            return None
        
        try:
            # Guardar timeout original
            original_timeout = self.serial.timeout
            
            # Establecer nuevo timeout
            self.serial.timeout = timeout
            
            # Leer respuesta
            response = self.serial.readline().decode('utf-8').strip()
            
            # Restaurar timeout original
            self.serial.timeout = original_timeout
            
            if response:
                logger.info(f"Respuesta recibida: {response}")
            
            return response
        except Exception as e:
            logger.error(f"Error al leer respuesta: {str(e)}")
            return None

class LSLCommandReceiver:
    """
    Clase para recibir comandos desde el stream LSL.
    """
    def __init__(self):
        self.inlet = None
        self.connected = False
        self.running = False
        self.thread = None
        
        # Callback para cuando se recibe un comando
        self.on_command_received = None
        
        # Contador de intentos de reconexión
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.last_reconnect_time = 0
        self.reconnect_interval = 5  # segundos
    
    def connect(self):
        """
        Conecta al stream LSL de comandos de forma no bloqueante.
        
        Returns:
            bool: True si la conexión fue exitosa
        """
        try:
            # Si ya estamos conectados, no hacer nada
            if self.connected and self.inlet:
                return True
                
            # Si ha pasado poco tiempo desde el último intento, no reintentar
            current_time = time.time()
            if current_time - self.last_reconnect_time < self.reconnect_interval:
                return False
            
            self.last_reconnect_time = current_time
            logger.info(f"Buscando stream LSL '{INPUT_STREAM_NAME}'...")
            
            # Buscar streams con un timeout corto
            streams = resolve_byprop('name', INPUT_STREAM_NAME, timeout=1.0)
            
            if not streams:
                logger.warning(f"No se encontró el stream '{INPUT_STREAM_NAME}'")
                return False
            
            self.inlet = StreamInlet(streams[0])
            self.connected = True
            logger.info("Conectado al stream LSL")
            return True
            
        except Exception as e:
            logger.error(f"Error al conectar al stream LSL: {str(e)}")
            self.connected = False
            return False
    
    def start_listening(self):
        """
        Inicia la escucha de comandos en un hilo separado.
        No bloquea si no hay conexión.
        """
        if self.running:
            return True
        
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Iniciada escucha de comandos LSL")
        return True
    
    def stop_listening(self):
        """
        Detiene la escucha de comandos.
        """
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        self.connected = False
        self.inlet = None
        logger.info("Detenida escucha de comandos LSL")
        return True
    
    def _listen_loop(self):
        """
        Bucle principal de escucha de comandos.
        Maneja reconexiones de forma no bloqueante.
        """
        while self.running:
            try:
                # Si no estamos conectados, intentar conectar
                if not self.connected or not self.inlet:
                    if self.connect():
                        self.reconnect_attempts = 0
                    else:
                        self.reconnect_attempts += 1
                        time.sleep(0.5)  # Pequeña pausa antes de reintentar
                        continue
                
                # Intentar recibir un comando con timeout corto
                sample, timestamp = self.inlet.pull_sample(timeout=0.1)
                
                if sample:
                    # Si se recibió una muestra y hay un callback registrado, notificar
                    if self.on_command_received:
                        self.on_command_received(sample[0], timestamp)
                
            except Exception as e:
                logger.error(f"Error en bucle de escucha: {str(e)}")
                self.connected = False
                self.inlet = None
                time.sleep(0.5)
            
            # Pequeña pausa para no saturar la CPU
            time.sleep(0.01)

class CommandSenderApp:
    """
    Aplicación principal para enviar comandos al ESP32 basados en clasificación EEG.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Command Sender")
        
        # Configurar ventana
        self.root.geometry("600x500")
        self.root.configure(padx=20, pady=20)
        self.root.resizable(True, True)
        
        # Interceptar cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Controlador ESP32
        self.esp = ESPController()
        
        # Receptor de comandos LSL
        self.lsl_receiver = LSLCommandReceiver()
        self.lsl_receiver.on_command_received = self.on_command_received
        
        # Variables para la interfaz
        self.target_class = StringVar(value="")
        self.led_state = StringVar(value="OFF")
        self.status_text = StringVar(value="Desconectado")
        self.detect_status = StringVar(value="Esperando conexión LSL...")
        self.port_var = StringVar(value="")
        self.lsl_status = StringVar(value="Desconectado")
        
        # Variables de control
        self.last_detected_class = ""
        self.last_detection_time = 0
        self.detection_active = False
        self.available_classes = []
        
        # Crear interfaz
        self.create_widgets()
        
        # Configuración inicial
        self.update_port_list()
        self.update_status()
        
        # Iniciar escucha de comandos LSL en segundo plano
        self.lsl_receiver.start_listening()
        
        # Actualizar interfaz periódicamente
        self.update_ui()
    
    def create_widgets(self):
        """
        Crea los widgets de la interfaz.
        """
        # === SECCIÓN DE ESTADO LSL === #
        lsl_frame = ttk.LabelFrame(self.root, text="Estado LSL", padding=10)
        lsl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(lsl_frame, text="Stream LSL:").pack(side=tk.LEFT, padx=5)
        self.lsl_status_label = ttk.Label(lsl_frame, textvariable=self.lsl_status)
        self.lsl_status_label.pack(side=tk.LEFT, padx=5)
        
        # === SECCIÓN DE CONEXIÓN ESP32 === #
        conn_frame = ttk.LabelFrame(self.root, text="Conexión ESP32", padding=10)
        conn_frame.pack(fill=tk.X, padx=10, pady=10, ipady=5)
        
        # Fila 1: Puerto y velocidad
        ttk.Label(conn_frame, text="Puerto:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        port_frame = ttk.Frame(conn_frame)
        port_frame.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=15)
        self.port_combo.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(port_frame, text="⟳", width=3, command=self.update_port_list).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(conn_frame, text="Baudrate:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        self.baudrate_combo = ttk.Combobox(conn_frame, values=["9600", "115200"], width=10)
        self.baudrate_combo.current(0)
        self.baudrate_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Fila 2: Botones de conexión
        btn_frame = ttk.Frame(conn_frame)
        btn_frame.grid(row=1, column=0, columnspan=4, pady=5)
        
        self.connect_btn = ttk.Button(btn_frame, text="Conectar", command=self.connect_esp)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_btn = ttk.Button(btn_frame, text="Desconectar", command=self.disconnect_esp, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        # Fila 3: Estado de conexión
        ttk.Label(conn_frame, text="Estado:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.status_label = ttk.Label(conn_frame, textvariable=self.status_text)
        self.status_label.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # === SECCIÓN DE CONTROL DE LED === #
        led_frame = ttk.LabelFrame(self.root, text="Control de LED", padding=10)
        led_frame.pack(fill=tk.X, padx=10, pady=10, ipady=5)
        
        # Fila 1: Botones de control manual
        manual_frame = ttk.Frame(led_frame)
        manual_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(manual_frame, text="Control Manual:").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(manual_frame, text="Encender", command=lambda: self.send_esp_command("ON")).pack(side=tk.LEFT, padx=5)
        ttk.Button(manual_frame, text="Apagar", command=lambda: self.send_esp_command("OFF")).pack(side=tk.LEFT, padx=5)
        ttk.Button(manual_frame, text="Alternar", command=lambda: self.send_esp_command("TOGGLE")).pack(side=tk.LEFT, padx=5)
        
        # Fila 2: Estado actual del LED
        state_frame = ttk.Frame(led_frame)
        state_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(state_frame, text="Estado del LED:").pack(side=tk.LEFT, padx=5)
        
        self.led_state_label = ttk.Label(state_frame, textvariable=self.led_state, width=6)
        self.led_state_label.pack(side=tk.LEFT, padx=5)
        
        # Indicador visual (círculo coloreado)
        self.led_indicator = tk.Canvas(state_frame, width=20, height=20, bg=COLOR_NORMAL, bd=0, highlightthickness=0)
        self.led_indicator.pack(side=tk.LEFT, padx=5)
        self.led_circle = self.led_indicator.create_oval(2, 2, 18, 18, fill="gray", outline="")
        
        # === SECCIÓN DE DETECCIÓN DE CLASES === #
        class_frame = ttk.LabelFrame(self.root, text="Detección de Clases EEG", padding=10)
        class_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Fila 1: Selección de clase objetivo
        target_frame = ttk.Frame(class_frame)
        target_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_frame, text="Clase objetivo:").pack(side=tk.LEFT, padx=5)
        
        self.class_combo = ttk.Combobox(target_frame, textvariable=self.target_class, width=25)
        self.class_combo.pack(side=tk.LEFT, padx=5)
        
        # Fila 2: Activar/desactivar detección
        activation_frame = ttk.Frame(class_frame)
        activation_frame.pack(fill=tk.X, pady=5)
        
        self.activate_btn = ttk.Button(activation_frame, text="Activar detección", command=self.toggle_detection)
        self.activate_btn.pack(side=tk.LEFT, padx=5)
        
        # Fila 3: Estado de detección
        detection_frame = ttk.Frame(class_frame)
        detection_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(detection_frame, text="Estado:").pack(side=tk.LEFT, padx=5)
        
        self.detect_label = ttk.Label(detection_frame, textvariable=self.detect_status, width=30)
        self.detect_label.pack(side=tk.LEFT, padx=5)
        
        # Fila 4: Log de detecciones
        log_frame = ttk.Frame(class_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(log_frame, text="Registro de detecciones:").pack(anchor=tk.W, padx=5)
        
        # Crear área de texto con scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(text_frame, height=10, width=50, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text.config(yscrollcommand=scrollbar.set)
        self.log_text.config(state=tk.DISABLED)  # Solo lectura
        
        # === BOTONES DE ACCIÓN GENERAL === #
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(bottom_frame, text="Salir", command=self.on_closing).pack(side=tk.RIGHT, padx=5)
    
    def update_port_list(self):
        """
        Actualiza la lista de puertos seriales disponibles.
        """
        ports = self.esp.list_ports()
        if ports:
            self.port_combo['values'] = ports
            
            # Seleccionar un puerto por defecto
            if not self.port_var.get() and len(ports) > 0:
                if sys.platform.startswith('win'):
                    # Buscar puertos COM
                    for port in ports:
                        if 'COM' in port:
                            self.port_var.set(port)
                            break
                    # Si no hay puertos COM, seleccionar el primero
                    if not self.port_var.get():
                        self.port_var.set(ports[0])
                else:
                    # En Unix, buscar ttyUSB o ttyACM
                    for port in ports:
                        if 'ttyUSB' in port or 'ttyACM' in port:
                            self.port_var.set(port)
                            break
                    # Si no hay puertos tty, seleccionar el primero
                    if not self.port_var.get():
                        self.port_var.set(ports[0])
        else:
            self.port_combo['values'] = ["No hay puertos disponibles"]
    
    def connect_esp(self):
        """
        Conecta al ESP32.
        """
        port = self.port_var.get()
        baudrate = int(self.baudrate_combo.get())
        
        if not port:
            messagebox.showerror("Error", "Selecciona un puerto serial")
            return
        
        # Mostrar mensaje de espera
        self.status_text.set("Conectando...")
        self.root.update_idletasks()
        
        # Intentar conectar
        if self.esp.connect(port, baudrate):
            messagebox.showinfo("Conexión", f"Conexión exitosa a ESP32 en {port}")
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.update_status()
        else:
            messagebox.showerror("Error", f"No se pudo conectar al ESP32 en {port}")
            self.status_text.set("Error de conexión")
    
    def disconnect_esp(self):
        """
        Desconecta del ESP32.
        """
        if self.esp.disconnect():
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            self.update_status()
        else:
            messagebox.showerror("Error", "Error al desconectar")
    
    def send_esp_command(self, command):
        """
        Envía un comando al ESP32.
        
        Args:
            command: Comando a enviar (ON, OFF, TOGGLE)
        """
        if not self.esp.connected:
            messagebox.showerror("Error", "No hay conexión con ESP32")
            return
        
        if self.esp.send_command(command):
            # Actualizar estado del LED según el comando
            if command == "ON":
                self.led_state.set("ON")
                self.led_indicator.itemconfig(self.led_circle, fill="#00FF00")
            elif command == "OFF":
                self.led_state.set("OFF")
                self.led_indicator.itemconfig(self.led_circle, fill="gray")
            elif command == "TOGGLE":
                # Alternar estado
                if self.led_state.get() == "ON":
                    self.led_state.set("OFF")
                    self.led_indicator.itemconfig(self.led_circle, fill="gray")
                else:
                    self.led_state.set("ON")
                    self.led_indicator.itemconfig(self.led_circle, fill="#00FF00")
            
            # Leer respuesta del ESP32
            response = self.esp.read_response()
            if response:
                self.log_message(f"ESP32 responde: {response}")
        else:
            self.log_message("Error al enviar comando")
    
    def toggle_detection(self):
        """
        Activa o desactiva la detección de clases.
        """
        self.detection_active = not self.detection_active
        
        if self.detection_active:
            target_class = self.target_class.get()
            if not target_class:
                messagebox.showerror("Error", "Debes seleccionar una clase objetivo")
                self.detection_active = False
                return
            
            self.activate_btn.config(text="Desactivar detección")
            self.log_message(f"Detección activada para clase: {target_class}")
            self.detect_status.set("Esperando detección...")
            self.detect_label.configure(background=COLOR_ACTIVE)
        else:
            self.activate_btn.config(text="Activar detección")
            self.log_message("Detección desactivada")
            self.detect_status.set("Detección desactivada")
            self.detect_label.configure(background=COLOR_NORMAL)
    
    def on_command_received(self, command, timestamp):
        """
        Callback llamado cuando se recibe un comando del stream LSL.
        
        Args:
            command: Clase detectada
            timestamp: Timestamp del comando
        """
        # Si es el primer comando, actualizar la lista de clases disponibles
        if not self.available_classes:
            # Suponemos que el comando es una clase y la añadimos a la lista
            self.available_classes.append(command)
            self.class_combo['values'] = self.available_classes
            
            # Si no hay clase seleccionada, seleccionar la primera
            if not self.target_class.get():
                self.target_class.set(command)
        elif command not in self.available_classes:
            self.available_classes.append(command)
            self.class_combo['values'] = self.available_classes
        
        # Si la detección está activada, procesar el comando
        if self.detection_active:
            target_class = self.target_class.get()
            
            # Registrar que se recibió un comando
            self.log_message(f"Clase detectada: {command}")
            
            # Verificar si coincide con la clase objetivo
            if command == target_class:
                current_time = time.time()
                
                # Actualizar estado de detección
                self.detect_status.set(f"¡Coincidencia! Clase: {command}")
                self.detect_label.configure(background=COLOR_MATCH)
                
                # Evitar enviar comandos demasiado seguidos
                if current_time - self.last_detection_time > COMMAND_COOLDOWN:
                    self.last_detection_time = current_time
                    
                    # Enviar comando para alternar el LED
                    self.log_message(f"¡Coincidencia con clase objetivo! Alternando LED")
                    self.send_esp_command("TOGGLE")
            else:
                # Actualizar estado de detección
                self.detect_status.set(f"Clase detectada: {command} (no coincide)")
                self.detect_label.configure(background=COLOR_NORMAL)
    
    def log_message(self, message):
        """
        Agrega un mensaje al log.
        
        Args:
            message: Mensaje a agregar
        """
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def update_status(self):
        """
        Actualiza el estado de conexión en la interfaz.
        """
        if self.esp.connected:
            self.status_text.set(f"Conectado a {self.esp.port} a {self.esp.baudrate} baudios")
            self.status_label.configure(background=COLOR_CONNECTED)
        else:
            self.status_text.set("Desconectado")
            self.status_label.configure(background=COLOR_DISCONNECTED)
    
    def update_ui(self):
        """
        Actualiza la interfaz periódicamente.
        """
        try:
            # Verificar conexión ESP32
            if self.esp.connected and not self.esp.serial:
                self.esp.connected = False
                self.update_status()
                self.connect_btn.config(state=tk.NORMAL)
                self.disconnect_btn.config(state=tk.DISABLED)
                self.log_message("Conexión con ESP32 perdida")
            
            # Actualizar estado LSL
            if self.lsl_receiver.connected:
                self.lsl_status.set("Conectado")
                self.lsl_status_label.configure(foreground="green")
            else:
                self.lsl_status.set("Esperando stream...")
                self.lsl_status_label.configure(foreground="orange")
            
            # Si la detección está activa pero no hay conexión LSL
            if self.detection_active and not self.lsl_receiver.connected:
                self.detect_status.set("Esperando conexión LSL...")
                self.detect_label.configure(background="#fff3cd")  # Amarillo suave
        
        except Exception as e:
            logger.error(f"Error al actualizar UI: {str(e)}")
        
        # Programar la próxima actualización
        self.root.after(1000, self.update_ui)
    
    def on_closing(self):
        """
        Maneja el cierre de la aplicación.
        """
        if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres salir?"):
            # Detener componentes
            self.lsl_receiver.stop_listening()
            self.esp.disconnect()
            
            # Cerrar ventana
            self.root.destroy()

def main():
    """
    Función principal del script.
    """
    print("="*70)
    print("ENVIADOR DE COMANDOS PARA ESP32 BASADO EN EEG")
    print("="*70)
    
    try:
        # Crear ventana principal
        root = tk.Tk()
        app = CommandSenderApp(root)
        
        # Iniciar bucle principal
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("Programa finalizado correctamente")
    return 0

if __name__ == "__main__":
    sys.exit(main())