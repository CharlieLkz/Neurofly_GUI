import tkinter as tk
from tkinter import ttk, messagebox, font
import subprocess
import sys
import os
import queue
import threading
from pathlib import Path
from status_monitor import StatusMonitor
import time
import torch
import torch.nn as nn

# Definir colores profesionales
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

class ProcessOutputFrame(ttk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, padding="10")
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Crear área de texto con scrollbar
        self.text_area = tk.Text(
            self, 
            wrap=tk.WORD, 
            height=10,
            bg=COLORS["bg_dark"],
            fg=COLORS["text"],
            font=("Consolas", 9),
            insertbackground=COLORS["text"]  # Color del cursor
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=scrollbar.set)
        
        # Empaquetar widgets
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Cola para mensajes
        self.message_queue = queue.Queue()
        self.running = True
        self.update_thread = threading.Thread(target=self._update_text, daemon=True)
        self.update_thread.start()
    
    def _update_text(self):
        """Actualiza el área de texto con mensajes de la cola"""
        while self.running:
            try:
                message = self.message_queue.get_nowait()
                self.text_area.insert(tk.END, message + "\n")
                self.text_area.see(tk.END)
            except queue.Empty:
                pass
            finally:
                if self.running:
                    self.after(100, self._update_text)
                    break
    
    def add_message(self, message):
        """Añade un mensaje a la cola"""
        self.message_queue.put(message)
    
    def clear(self):
        """Limpia el área de texto"""
        self.text_area.delete(1.0, tk.END)
    
    def stop(self):
        """Detiene la actualización de texto"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join()

class ProgramController:
    def __init__(self, root):
        self.root = root
        self.root.title("Neurofly Control Panel")
        self.root.geometry("900x700")  # Tamaño inicial de la ventana
        
        # Configurar tema y estilos
        self.setup_styles()
        
        # Obtener directorio actual
        self.current_dir = Path(__file__).parent.resolve()
        
        # Configurar procesos
        self.processes = {
            "realtime_processor.py": {
                "path": self.current_dir / "realtime_processor.py",
                "process": None,
                "output_frame": None
            },
            "CNN_Inference.py": {  # Cambiado de CNN_Interference.py a CNN_Inference.py
                "path": self.current_dir / "CNN_Inference.py",
                "process": None,
                "output_frame": None
            },
            "CommandSender.py": {
                "path": self.current_dir / "CommandSender.py",
                "process": None,
                "output_frame": None
            }
        }
        
        # Crear monitor de estado
        self.status_monitor = StatusMonitor(root)
        
        # Crear frames de salida
        for name in self.processes:
            self.processes[name]["output_frame"] = ProcessOutputFrame(root, f"Salida de {name}")
        
        # Crear botones de control
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_button = ttk.Button(
            self.control_frame, 
            text="Iniciar Sistema", 
            command=self.start_all,
            style="Accent.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            self.control_frame, 
            text="Detener Sistema", 
            command=self.stop_all,
            style="Danger.TButton"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Configurar cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configura los estilos de la interfaz"""
        style = ttk.Style()
        
        # Configurar tema general
        style.configure("TFrame", background=COLORS["bg_dark"])
        style.configure("TLabel", background=COLORS["bg_dark"], foreground=COLORS["text"])
        
        # Estilo para botones normales
        style.configure("TButton", 
                        font=("Segoe UI", 10, "bold"),
                        padding=10)
        
        # Estilo para el botón de inicio (verde)
        style.configure("Accent.TButton", 
                        background=COLORS["success"],
                        foreground="black",
                        font=("Segoe UI", 10, "bold"),
                        padding=10)
        style.map("Accent.TButton",
                background=[('active', '#45a049'), ('pressed', '#3d8b40')],
                foreground=[('active', 'black'), ('pressed', 'black')])
        
        # Estilo para el botón de detener (rojo)
        style.configure("Danger.TButton", 
                        background=COLORS["error"],
                        foreground="black",
                        font=("Segoe UI", 10, "bold"),
                        padding=10)
        style.map("Danger.TButton",
                background=[('active', '#d32f2f'), ('pressed', '#c62828')],
                foreground=[('active', 'black'), ('pressed', 'black')])
        
        # Estilo para el título
        style.configure("Title.TLabel", 
                        font=("Segoe UI", 16, "bold"), 
                        foreground=COLORS["text"],
                        background=COLORS["bg_dark"])
        
        # Estilo para los frames
        style.configure("TLabelframe", 
                        background=COLORS["bg_light"], 
                        foreground=COLORS["text"])
        style.configure("TLabelframe.Label", 
                        background=COLORS["bg_light"], 
                        foreground=COLORS["text"],
                        font=("Segoe UI", 10, "bold"))
        
        # Configurar color de fondo de la ventana principal
        self.root.configure(bg=COLORS["bg_dark"])
    
    def start_program(self, program_name):
        """Inicia un programa específico"""
        program = self.processes[program_name]
        
        if program["process"] is not None:
            return
        
        if not program["path"].exists():
            error_msg = f"Error: No se encuentra el archivo {program_name}"
            program["output_frame"].add_message(error_msg)
            messagebox.showerror("Error", error_msg)
            return
        
        try:
            # Iniciar proceso sin ventana de terminal
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            process = subprocess.Popen(
                [sys.executable, str(program["path"])],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=True,
                bufsize=1
            )
            
            program["process"] = process
            program["output_frame"].add_message(f"Iniciando {program_name}...")
            
            # Iniciar thread para leer salida
            threading.Thread(
                target=self._read_output,
                args=(program_name, process),
                daemon=True
            ).start()
            
        except Exception as e:
            error_msg = f"Error al iniciar {program_name}: {str(e)}"
            program["output_frame"].add_message(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def _read_output(self, program_name, process):
        """Lee la salida del proceso y actualiza el frame correspondiente"""
        program = self.processes[program_name]
        output_frame = program["output_frame"]
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                output_frame.add_message(line.strip())
                
                # Actualizar estado según el contenido de la línea
                if "recibiendo datos" in line.lower():
                    self.status_monitor.update_command_status("Recibiendo datos EEG")
                elif "procesando" in line.lower():
                    self.status_monitor.update_command_status("Procesando características")
                elif "comando" in line.lower():
                    self.status_monitor.update_command_status(line.strip())
                elif "drone" in line.lower():
                    self.status_monitor.update_drone_status(line.strip())
    
    def stop_program(self, program_name):
        """Detiene un programa específico"""
        program = self.processes[program_name]
        
        if program["process"] is not None:
            try:
                # Intentar terminar el proceso principal
                program["process"].terminate()
                
                # Esperar hasta 3 segundos para que termine
                for _ in range(30):  # 30 * 0.1 = 3 segundos
                    if program["process"].poll() is not None:
                        break
                    time.sleep(0.1)
                
                # Si aún está corriendo, forzar el cierre
                if program["process"].poll() is None:
                    program["process"].kill()
                    program["process"].wait()
                
                program["process"] = None
                program["output_frame"].add_message(f"{program_name} detenido.")
                
            except Exception as e:
                program["output_frame"].add_message(f"Error al detener {program_name}: {str(e)}")
                # Intentar forzar el cierre si hubo error
                try:
                    program["process"].kill()
                except:
                    pass
    
    def start_all(self):
        """Inicia todos los programas en el orden correcto"""
        # 1. Iniciar realtime_processor.py
        self.start_program("realtime_processor.py")
        
        # 2. Esperar 2 segundos para que se establezca el stream
        self.root.after(2000, lambda: self._start_cnn())
    
    def _start_cnn(self):
        """Inicia CNN_Inference.py después del delay"""
        self.start_program("CNN_Inference.py")
        
        # 3. Esperar 2 segundos más antes de iniciar CommandSender
        self.root.after(2000, lambda: self.start_program("CommandSender.py"))
    
    def stop_all(self):
        """Detiene todos los programas en orden inverso"""
        # Primero, deshabilitar los botones para evitar clics múltiples
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        
        try:
            # Detener en orden inverso
            programs_to_stop = ["CommandSender.py", "CNN_Inference.py", "realtime_processor.py"]
            
            for program_name in programs_to_stop:
                self.stop_program(program_name)
                # Asegurar que la interfaz se actualice
                self.root.update()
                time.sleep(0.5)
            
            # Verificar que todos los procesos estén detenidos
            for program_name, program in self.processes.items():
                if program["process"] is not None and program["process"].poll() is None:
                    # Forzar cierre si algún proceso sigue vivo
                    try:
                        program["process"].kill()
                        program["process"].wait()
                        program["process"] = None
                    except:
                        pass
        
        finally:
            # Reactivar los botones
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="normal")
    
    def on_closing(self):
        """Maneja el cierre de la ventana"""
        self.stop_all()
        self.status_monitor.stop()
        for program in self.processes.values():
            if program["output_frame"]:
                program["output_frame"].stop()
        self.root.destroy()

class EEGCNNMulticlass(nn.Module):
    def __init__(self, in_channels, n_classes, config):
        super().__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.hidden_dim = config.get('hidden_dim', 384)
        self.dropout_rate = config.get('dropout_rate', 0.25)
        self.channels = config.get('channels', [64, 128, 192])
        self.use_attention = config.get('attention_mechanism', True)
        
        # Construir la arquitectura
        self.build_architecture()
    
    def build_architecture(self):
        """Construye la arquitectura del modelo basada en la configuración"""
        # Capas convolucionales
        self.conv_layers = nn.ModuleList()
        current_channels = self.in_channels
        
        for out_channels in self.channels:
            conv_block = nn.Sequential(
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(self.dropout_rate)
            )
            self.conv_layers.append(conv_block)
            current_channels = out_channels
        
        # Calcular dimensión de entrada para fc1
        self.fc1_in_features = self.channels[-1] * 2  # Ajustar según tu arquitectura
        
        # Capas fully connected
        self.fc1 = nn.Linear(self.fc1_in_features, self.hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout_fc1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)
        
        # Mecanismo de atención (opcional)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.channels[-1], 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        # Aplicar capas convolucionales
        for conv_block in self.conv_layers:
            x = conv_block(x)
        
        # Aplicar atención si está habilitada
        if self.use_attention:
            attention_weights = self.attention(x.transpose(1, 2))
            x = x * attention_weights.transpose(1, 2)
        
        # Aplanar para capas fully connected
        x = x.view(x.size(0), -1)
        
        # Capas fully connected con BatchNorm y Dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    root = tk.Tk()
    app = ProgramController(root)
    root.mainloop() 