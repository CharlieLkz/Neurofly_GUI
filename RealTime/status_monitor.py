import tkinter as tk
from tkinter import ttk
import threading
import time
from pylsl import resolve_byprop
import queue

# Definir colores profesionales (igual que en run_sequence_gui.py)
COLORS = {
    "bg_dark": "#1e1e2e",       # Fondo principal (azul oscuro)
    "bg_light": "#2b2b3a",      # Fondo secundario (azul grisáceo)
    "accent": "#4a4a8f",        # Color de acento (azul medio)
    "text": "#e0e0e0",          # Texto principal (gris claro)
    "text_secondary": "#a0a0a0", # Texto secundario (gris)
    "success": "#4caf50",       # Verde para éxito
    "warning": "#ff9800",       # Naranja para advertencias
    "error": "#f44336",         # Rojo para errores
    "info": "#2196f3"           # Azul para información
}

class StatusMonitor:
    def __init__(self, parent):
        self.parent = parent
        self.status_frame = ttk.LabelFrame(parent, text="Estado del Sistema", padding="10")
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Configurar estilos
        self.setup_styles()
        
        # Crear etiquetas de estado
        self.status_labels = {}
        self.status_values = {}
        
        # Definir los componentes a monitorear
        self.components = [
            {"name": "AURA_Power", "label": "AURA Power"},
            {"name": "CNN_Inference", "label": "CNN Inference"},  # Cambiado de CNN_Interference a CNN_Inference
            {"name": "CommandSender", "label": "Command Sender"},
            {"name": "Drone", "label": "Drone"}
        ]
        
        # Crear etiquetas para cada componente
        for i, component in enumerate(self.components):
            # Frame para cada fila de estado
            row_frame = ttk.Frame(self.status_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            # Etiqueta del nombre
            name_label = ttk.Label(
                row_frame, 
                text=component["label"] + ":", 
                width=15,
                style="StatusLabel.TLabel"
            )
            name_label.pack(side=tk.LEFT, padx=5)
            
            # Etiqueta del valor
            value_label = ttk.Label(
                row_frame, 
                text="Desconocido", 
                width=20,
                style="StatusValue.TLabel"
            )
            value_label.pack(side=tk.LEFT, padx=5)
            
            # Guardar referencias
            self.status_labels[component["name"]] = name_label
            self.status_values[component["name"]] = value_label
        
        # Cola para actualizaciones de estado
        self.status_queue = queue.Queue()
        
        # Iniciar thread de monitoreo
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_status, daemon=True)
        self.monitor_thread.start()
        
        # Iniciar actualizador de UI
        self._update_ui()
    
    def setup_styles(self):
        """Configura los estilos de la interfaz"""
        style = ttk.Style()
        
        # Estilo para etiquetas de estado
        style.configure("StatusLabel.TLabel", 
                        font=("Segoe UI", 10, "bold"),
                        background=COLORS["bg_light"],
                        foreground=COLORS["text"])
        
        # Estilo para valores de estado
        style.configure("StatusValue.TLabel", 
                        font=("Segoe UI", 10),
                        background=COLORS["bg_light"],
                        foreground=COLORS["text"])
        
        # Estilo para el frame de estado
        style.configure("TLabelframe", 
                        background=COLORS["bg_light"],
                        foreground=COLORS["text"])
        style.configure("TLabelframe.Label", 
                        font=("Segoe UI", 11, "bold"),
                        background=COLORS["bg_light"],
                        foreground=COLORS["text"])
    
    def _monitor_status(self):
        """Monitorea el estado de los streams LSL"""
        while self.running:
            try:
                # Buscar streams LSL
                streams = resolve_byprop("type", "EEG", timeout=1.0)
                
                # Actualizar estado de AURA Power
                if streams:
                    self.status_queue.put(("AURA_Power", "Activo", "success"))
                else:
                    self.status_queue.put(("AURA_Power", "Inactivo", "error"))
                
                # Buscar streams de predicción
                pred_streams = resolve_byprop("name", "PREDICTION_STREAM", timeout=1.0)
                if pred_streams:
                    self.status_queue.put(("CNN_Inference", "Activo", "success"))
                else:
                    self.status_queue.put(("CNN_Inference", "Inactivo", "error"))
                
                # Buscar streams de comandos
                cmd_streams = resolve_byprop("name", "COMMAND_STREAM", timeout=1.0)
                if cmd_streams:
                    self.status_queue.put(("CommandSender", "Activo", "success"))
                else:
                    self.status_queue.put(("CommandSender", "Inactivo", "error"))
                
                # El estado del dron se actualiza desde fuera
                
            except Exception as e:
                print(f"Error en monitoreo: {str(e)}")
            
            time.sleep(1)
    
    def _update_ui(self):
        """Actualiza la interfaz de usuario con los estados más recientes"""
        try:
            while not self.status_queue.empty():
                component, status, color_type = self.status_queue.get_nowait()
                if component in self.status_values:
                    self.status_values[component].configure(
                        text=status,
                        foreground=COLORS[color_type]
                    )
        except queue.Empty:
            pass
        finally:
            if self.running:
                self.parent.after(100, self._update_ui)
    
    def update_drone_status(self, status):
        """Actualiza el estado del dron"""
        if "error" in status.lower() or "fallo" in status.lower():
            color = "error"
        elif "advertencia" in status.lower() or "warning" in status.lower():
            color = "warning"
        else:
            color = "success"
        
        self.status_queue.put(("Drone", status, color))
    
    def update_command_status(self, status):
        """Actualiza el estado del comando"""
        if "error" in status.lower() or "fallo" in status.lower():
            color = "error"
        elif "advertencia" in status.lower() or "warning" in status.lower():
            color = "warning"
        else:
            color = "info"
        
        self.status_queue.put(("CommandSender", status, color))
    
    def stop(self):
        """Detiene el monitoreo"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join() 