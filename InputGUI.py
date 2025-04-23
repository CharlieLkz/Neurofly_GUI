#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InputGUI.py - Interfaz principal para sistema de control EEG-Dron
Envía una sola muestra EEG desde AURA vía LSL para procesamiento
y recibe predicciones para control del dron
"""

import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import time
import threading
import os
from datetime import datetime
import pylsl
import numpy as np
import socket
import sys

# === CONFIGURATION CONSTANTS === #
INPUT_STREAM_NAME = "AURA_Power"  # Nombre del stream LSL de AURA
OUTPUT_STREAM_NAME = "AURAPSD"  # Nombre del stream LSL de salida
PREDICTION_STREAM_NAME = "PREDICTION_STREAM"  # Stream de predicciones
SCALE_FACTOR_EEG = (4500000)/24/(2**23-1)
N_CANALES = 8  # Número de canales físicos
BANDAS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]  # Bandas de frecuencia
TOTAL_COLUMNAS = N_CANALES * len(BANDAS)  # Total columnas de datos (8 canales x 5 bandas = 40)
TIMEOUT_SECONDS = 15  # Timeout en segundos para espera de datos

# Mapeo de bandas para mostrar feature relevante
BANDA_MAP = {0: "Delta", 1: "Theta", 2: "Alpha", 3: "Beta", 4: "Gamma"}

class InputGUIController:
    """Controlador principal de la interfaz gráfica para el sistema EEG-Dron"""
    
    def __init__(self, master):
        """
        Inicializa la interfaz gráfica y configura los componentes
        
        Args:
            master: Ventana principal de tkinter
        """
        # Configuración de la ventana principal
        self.master = master
        self.master.title("Sistema de Control EEG-Dron - Envío Único")
        self.master.geometry("700x400")
        self.master.config(bg="#002147")  # Fondo azul oscuro
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables de estado
        self.aura_stream = None
        self.prediction_stream = None
        self.connected_to_aura = False
        self.connected_to_prediction = False
        self.sample_sent = False
        self.timeout_timer = None
        self.prediction_received = False
        self.tello_socket = None
        
        # Variable para controlar el cierre de la aplicación
        self.exit_flag = False
        
        # Crear directorios si no existen
        os.makedirs("AURAData", exist_ok=True)
        
        # Configurar el layout principal
        self.setup_ui()
    
    def setup_ui(self):
        """Configura todos los elementos de la interfaz de usuario"""
        # Estilos y fuentes
        title_font = ('Helvetica', 18, 'bold')
        label_font = ('Helvetica', 12)
        button_font = ('Helvetica', 12, 'bold')
        console_font = ('Courier', 10)
        
        # Frame principal
        main_frame = tk.Frame(self.master, bg="#002147")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        tk.Label(main_frame, text="Envío Único de Datos EEG", 
                font=title_font, bg="#002147", fg="white").pack(pady=20)
        
        # Frame para controles
        control_frame = tk.Frame(main_frame, bg="#002147")
        control_frame.pack(pady=10, fill=tk.X)
        
        # Estado AURA
        tk.Label(control_frame, text="Estado AURA:", 
                font=label_font, bg="#002147", fg="white").grid(row=0, column=0, sticky="w", pady=5)
        self.aura_status = tk.Label(control_frame, text="Desconectado", 
                                  font=label_font, bg="#002147", fg="#FF5555")
        self.aura_status.grid(row=0, column=1, pady=5, sticky="w")
        
        # Estado Predicción
        tk.Label(control_frame, text="Estado Predicción:", 
                font=label_font, bg="#002147", fg="white").grid(row=1, column=0, sticky="w", pady=5)
        self.prediction_status = tk.Label(control_frame, text="Esperando muestra", 
                                       font=label_font, bg="#002147", fg="#FFAA00")
        self.prediction_status.grid(row=1, column=1, pady=5, sticky="w")
        
        # Botones
        button_frame = tk.Frame(main_frame, bg="#002147")
        button_frame.pack(pady=20, fill=tk.X)
        
        self.connect_aura_button = tk.Button(button_frame, text="Conectar con AURA", 
                                         command=self.connect_to_aura, bg="#006699", fg="white", 
                                         font=button_font, width=18)
        self.connect_aura_button.pack(side=tk.LEFT, padx=5)
        
        self.send_button = tk.Button(button_frame, text="Enviar una muestra", 
                                  command=self.send_single_sample, bg="#006699", fg="white", 
                                  font=button_font, width=18, state=tk.DISABLED)
        self.send_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = tk.Button(button_frame, text="Salir", 
                                 command=self.on_closing, bg="#990000", fg="white", 
                                 font=button_font, width=10)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Área de log
        tk.Label(main_frame, text="Log:", 
                font=label_font, bg="#002147", fg="white").pack(anchor="w", pady=(20, 5))
        
        self.log_console = scrolledtext.ScrolledText(main_frame, height=10, 
                                                  width=70, font=console_font, 
                                                  bg="#000033", fg="#00FF00")
        self.log_console.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Añadir mensaje inicial al log
        self.log_message("Sistema iniciado. Conecte con AURA para enviar una muestra.")
    
    def log_message(self, message):
        """Añade un mensaje al área de log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_console.config(state=tk.NORMAL)
        self.log_console.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_console.see(tk.END)  # Scroll al final
        self.log_console.config(state=tk.DISABLED)
        print(f"[{timestamp}] {message}")  # También mostrar en consola
    
    def connect_to_aura(self):
        """Conecta con el dispositivo AURA a través de LSL"""
        self.log_message("Buscando dispositivo AURA...")
        self.aura_status.config(text="Buscando...", fg="#FFA500")
        
        # Deshabilitar botón mientras se busca
        self.connect_aura_button.config(state=tk.DISABLED)
        
        def search_aura_thread():
            try:
                # Buscar el stream de AURA
                timeout = 10.0
                start_time = time.time()
                streams = None
                
                while time.time() - start_time < timeout and not streams:
                    try:
                        streams = pylsl.resolve_byprop('name', INPUT_STREAM_NAME, timeout=1.0)
                    except Exception:
                        time.sleep(0.5)
                
                if streams:
                    # Conectar al stream de AURA
                    self.aura_stream = pylsl.StreamInlet(streams[0])
                    self.connected_to_aura = True
                    
                    # Actualizar UI en el hilo principal
                    self.master.after(0, lambda: self.aura_status.config(text="Conectado ✓", fg="#55FF55"))
                    self.master.after(0, lambda: self.log_message(f"Conectado al stream {INPUT_STREAM_NAME}"))
                    self.master.after(0, lambda: self.send_button.config(state=tk.NORMAL))
                    
                    # Probar recibir una muestra
                    sample, timestamp = self.aura_stream.pull_sample(timeout=1.0)
                    if sample:
                        self.master.after(0, lambda: self.log_message(f"Muestra recibida de AURA con {len(sample)} valores"))
                    else:
                        self.master.after(0, lambda: self.log_message("Conectado pero no se recibieron datos iniciales"))
                else:
                    # No se encontró AURA, actualizar UI
                    self.master.after(0, lambda: self.aura_status.config(text="No encontrado", fg="#FF5555"))
                    self.master.after(0, lambda: self.log_message(f"No se encontró el stream {INPUT_STREAM_NAME}"))
                    self.master.after(0, lambda: self.connect_aura_button.config(state=tk.NORMAL))
            except Exception as e:
                self.master.after(0, lambda err=str(e): self.log_message(f"Error al conectar: {err}"))
                self.master.after(0, lambda: self.aura_status.config(text="Error", fg="#FF5555"))
                self.master.after(0, lambda: self.connect_aura_button.config(state=tk.NORMAL))
        
        # Iniciar búsqueda en hilo separado
        threading.Thread(target=search_aura_thread, daemon=True).start()
    
    def send_single_sample(self):
        """Envía una sola muestra de EEG y espera por predicción"""
        if not self.connected_to_aura or not self.aura_stream:
            self.log_message("Error: No hay conexión con AURA")
            return
        
        self.log_message("Esperando muestra de AURA...")
        
        # Deshabilitar botones
        self.send_button.config(state=tk.DISABLED)
        self.connect_aura_button.config(state=tk.DISABLED)
        
        def send_sample_thread():
            try:
                # Recibir una muestra de AURA
                sample, timestamp = self.aura_stream.pull_sample(timeout=3.0)
                
                if sample:
                    # Crear stream LSL para enviar la muestra a realtime_processor
                    info = pylsl.StreamInfo(
                        name=OUTPUT_STREAM_NAME,
                        type='EEG',
                        channel_count=len(sample),
                        nominal_srate=0,  # Irregular rate (single sample)
                        channel_format='float32',
                        source_id='InputGUI'
                    )
                    outlet = pylsl.StreamOutlet(info)
                    
                    # Guardar muestra en archivo
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"AURAData/single_sample_{timestamp_str}.csv"
                    
                    with open(filename, 'w') as f:
                        f.write(f"# Muestra única AURA\n")
                        f.write(f"# Timestamp: {timestamp}\n")
                        f.write(f"# Fecha: {timestamp_str}\n")
                        f.write("Valor\n")
                        for val in sample:
                            f.write(f"{val}\n")
                    
                    # Enviar muestra al stream LSL
                    outlet.push_sample(sample)
                    
                    self.master.after(0, lambda: self.log_message(f"Muestra enviada con {len(sample)} valores"))
                    self.master.after(0, lambda: self.log_message(f"Datos guardados en {filename}"))
                    self.master.after(0, lambda: self.prediction_status.config(text="Esperando predicción...", fg="#FFAA00"))
                    
                    # Marcar como enviada
                    self.sample_sent = True
                    
                    # Conectarse al stream de predicciones y esperar resultado
                    self.master.after(0, lambda: self.connect_to_prediction_stream())
                    
                else:
                    self.master.after(0, lambda: self.log_message("Error: No se recibió muestra de AURA"))
                    # Reactivar botones
                    self.master.after(0, lambda: self.send_button.config(state=tk.NORMAL))
                    self.master.after(0, lambda: self.connect_aura_button.config(state=tk.NORMAL))
            except Exception as e:
                self.master.after(0, lambda err=str(e): self.log_message(f"Error al enviar muestra: {err}"))
                # Reactivar botones
                self.master.after(0, lambda: self.send_button.config(state=tk.NORMAL))
                self.master.after(0, lambda: self.connect_aura_button.config(state=tk.NORMAL))
        
        # Iniciar envío en hilo separado
        threading.Thread(target=send_sample_thread, daemon=True).start()
    
    def connect_to_prediction_stream(self):
        """Conecta al stream de predicciones para recibir el resultado"""
        self.log_message(f"Conectando al stream de predicciones {PREDICTION_STREAM_NAME}...")
        
        def prediction_thread():
            try:
                # Buscar el stream de predicciones
                timeout_find = 10.0
                start_time = time.time()
                streams = None
                
                while time.time() - start_time < timeout_find and not streams:
                    try:
                        streams = pylsl.resolve_byprop('name', PREDICTION_STREAM_NAME, timeout=1.0)
                    except Exception:
                        time.sleep(0.5)
                
                if streams:
                    # Conectar al stream de predicciones
                    self.prediction_stream = pylsl.StreamInlet(streams[0])
                    self.connected_to_prediction = True
                    
                    # Actualizar UI
                    self.master.after(0, lambda: self.log_message(f"Conectado al stream {PREDICTION_STREAM_NAME}"))
                    
                    # Iniciar temporizador de timeout
                    self.timeout_timer = threading.Timer(TIMEOUT_SECONDS, self.on_prediction_timeout)
                    self.timeout_timer.daemon = True
                    self.timeout_timer.start()
                    
                    # Esperar predicción
                    while not self.exit_flag and not self.prediction_received:
                        # Recibir predicción con timeout corto para permitir interrupciones
                        sample, timestamp = self.prediction_stream.pull_sample(timeout=0.5)
                        
                        if sample:
                            # Cancelar el temporizador de timeout
                            if self.timeout_timer:
                                self.timeout_timer.cancel()
                                self.timeout_timer = None
                            
                            # Procesar la predicción recibida
                            self.prediction_received = True
                            prediction = sample[0]  # Clase predicha (primer elemento)
                            
                            # El segundo elemento es el índice de la feature más relevante
                            feature_idx = -1
                            if len(sample) > 1 and sample[1]:
                                try:
                                    feature_idx = int(sample[1])
                                except ValueError:
                                    feature_idx = -1
                            
                            # Calcular canal y banda desde el índice
                            canal = None
                            banda = None
                            if feature_idx >= 0:
                                canal = (feature_idx // 5) + 1  # Índice // 5 → canal (1-based)
                                banda = BANDA_MAP.get(feature_idx % 5, "Desconocida")  # Índice % 5 → banda
                            
                            feature_text = f"Canal {canal} – Banda {banda}" if canal and banda else "No disponible"
                            
                            # Mostrar predicción en la UI principal
                            self.master.after(0, lambda p=prediction, f=feature_text: 
                                              self.show_prediction_popup(p, f))
                    
                else:
                    # No se encontró el stream de predicciones
                    self.master.after(0, lambda: self.log_message(f"No se encontró el stream {PREDICTION_STREAM_NAME}"))
                    self.master.after(0, lambda: self.prediction_status.config(text="Error: No disponible", fg="#FF5555"))
                    
                    # Iniciar temporizador de timeout
                    self.timeout_timer = threading.Timer(TIMEOUT_SECONDS, self.on_prediction_timeout)
                    self.timeout_timer.daemon = True
                    self.timeout_timer.start()
            except Exception as e:
                self.master.after(0, lambda err=str(e): self.log_message(f"Error al conectar a predicciones: {err}"))
                # Iniciar temporizador de timeout
                self.timeout_timer = threading.Timer(TIMEOUT_SECONDS, self.on_prediction_timeout)
                self.timeout_timer.daemon = True
                self.timeout_timer.start()
        
        # Iniciar hilo para esperarpredición
        threading.Thread(target=prediction_thread, daemon=True).start()
    
    def on_prediction_timeout(self):
        """Maneja el caso de timeout al esperar predicción"""
        self.log_message(f"Timeout: No se recibió predicción después de {TIMEOUT_SECONDS} segundos")
        self.prediction_status.config(text="Timeout: Sin predicción", fg="#FF5555")
        
        # Mostrar mensaje y cerrar aplicación
        messagebox.showwarning("Timeout", f"No se recibió predicción después de {TIMEOUT_SECONDS} segundos. El programa se cerrará.")
        self.exit_flag = True
        self.master.after(1000, self.master.destroy)
    
    def show_prediction_popup(self, prediction, feature_relevante):
        """
        Muestra un popup con la predicción recibida y la feature más relevante
        
        Args:
            prediction (str): Nombre de la clase predicha
            feature_relevante (str): Etiqueta de la feature más relevante (Canal X - Banda Y)
        """
        self.log_message(f"Predicción recibida: {prediction}, Feature relevante: {feature_relevante}")
        self.prediction_status.config(text=f"Recibida: {prediction}", fg="#55FF55")
        
        # Verificar si la predicción es válida
        valid_predictions = [
            "RightArmThinking", "LeftArmThinking", 
            "RightFistThinking", "LeftFistThinking",
            "RightFootThinking", "LeftFootThinking",
            "TakeoffCommand", "LandCommand"
        ]
        
        if prediction not in valid_predictions:
            self.log_message(f"Predicción no reconocida: {prediction}")
            messagebox.showwarning("Predicción No Válida", 
                                  "Predicción no reconocida, inténtalo de nuevo.")
            
            # Reactivar botones para permitir otro intento
            self.send_button.config(state=tk.NORMAL)
            self.connect_aura_button.config(state=tk.NORMAL)
            self.prediction_received = False
            return
        
        # Crear popup para confirmación
        popup = tk.Toplevel(self.master)
        popup.title("Confirmar Predicción")
        popup.geometry("400x250")
        popup.config(bg="#002147")  # Mismo fondo azul oscuro
        popup.grab_set()  # Modal
        
        # Añadir contenido al popup
        title_font = ('Helvetica', 16, 'bold')
        text_font = ('Helvetica', 12)
        button_font = ('Helvetica', 12, 'bold')
        
        tk.Label(popup, text="Predicción Detectada", 
               font=title_font, bg="#002147", fg="white").pack(pady=(20, 10))
        
        tk.Label(popup, text=f"Clase: {prediction}", 
               font=text_font, bg="#002147", fg="white").pack(pady=5)
        
        tk.Label(popup, text=f"Feature más relevante:", 
               font=text_font, bg="#002147", fg="white").pack(pady=5)
        
        tk.Label(popup, text=feature_relevante, 
               font=text_font, bg="#002147", fg="#55FF55").pack(pady=5)
        
        # Botón de confirmación
        confirm_button = tk.Button(
            popup, text="Confirmar", font=button_font,
            bg="#006699", fg="white", width=15,
            command=lambda p=prediction: self.on_confirm_prediction(p, popup)
        )
        confirm_button.pack(pady=20)
    
    def on_confirm_prediction(self, prediction, popup):
        """
        Maneja la confirmación de predicción y envía el comando al dron
        
        Args:
            prediction (str): Nombre de la clase predicha
            popup (tk.Toplevel): Ventana de popup a cerrar
        """
        self.log_message(f"Predicción confirmada: {prediction}. Enviando comando al dron...")
        
        # Cerrar popup
        popup.destroy()
        
        # Enviar comando al dron
        def send_command_thread(comando):
            try:
                self.log_message(f"Enviando '{comando}' al PREDICTION_STREAM para el dron...")
                
                # Enviar comando a través de LSL como alternativa
                # Crear outlet para PREDICTION_STREAM
                info = pylsl.StreamInfo(
                    name=PREDICTION_STREAM_NAME,
                    type='Command',
                    channel_count=1,
                    nominal_srate=0,  # Irregular rate (single sample)
                    channel_format='string',
                    source_id='InputGUI'
                )
                outlet = pylsl.StreamOutlet(info)
                
                # Enviar comando
                outlet.push_sample([comando])
                self.log_message(f"Comando enviado a PREDICTION_STREAM: {comando}")
                
                # Dar tiempo para que el comando se procese
                time.sleep(1)
                
                # Cerrar la aplicación
                self.log_message("Operación completada. Cerrando aplicación...")
                self.exit_flag = True
                self.master.after(1000, self.master.destroy)
                
            except Exception as e:
                self.log_message(f"Error al enviar comando: {str(e)}")
                # Cerrar de todas formas
                self.exit_flag = True
                self.master.after(1000, self.master.destroy)
        
        # Iniciar hilo para enviar comando
        threading.Thread(target=send_command_thread, args=(prediction,), daemon=True).start()
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        if messagebox.askokcancel("Salir", "¿Desea cerrar la aplicación?"):
            # Marcar para salir
            self.exit_flag = True
            
            # Cancelar timer si existe
            if self.timeout_timer:
                self.timeout_timer.cancel()
                self.timeout_timer = None
            
            # Cerrar streams si existen
            if self.aura_stream:
                del self.aura_stream
            
            if self.prediction_stream:
                del self.prediction_stream
            
            # Cerrar socket si existe
            if self.tello_socket:
                self.tello_socket.close()
            
            self.master.destroy()


def main():
    """Función principal para iniciar la aplicación"""
    root = tk.Tk()
    app = InputGUIController(root)
    root.mainloop()


if __name__ == "__main__":
    main()