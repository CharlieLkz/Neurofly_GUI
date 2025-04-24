import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import time
import threading

class ProgramController:
    def __init__(self, root):
        self.root = root
        self.root.title("Control de Programas EEG")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Diccionario para almacenar los procesos y su información
        self.processes = {
            "InputGUI.py": {
                "process": None,
                "order": 1,
                "description": "Interfaz de entrada EEG",
                "stream_out": "AURA_Power",
                "status": "Detenido"
            },
            "realtime_processor.py": {
                "process": None,
                "order": 2,
                "description": "Procesador de señales",
                "stream_in": "AURA_Power",
                "stream_out": "FEATURE_STREAM",
                "status": "Detenido"
            },
            "cnn_interference.py": {
                "process": None,
                "order": 3,
                "description": "Inferencia CNN",
                "stream_in": "FEATURE_STREAM",
                "stream_out": "CNN_COMMANDS",
                "status": "Detenido"
            },
            "CommandSender.py": {
                "process": None,
                "order": 4,
                "description": "Control de comandos",
                "stream_in": "CNN_COMMANDS",
                "status": "Detenido"
            }
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        title_label = ttk.Label(main_frame, 
                               text="Control de Programas EEG", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Frame para información
        info_frame = ttk.LabelFrame(main_frame, text="Información", padding="10")
        info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        info_text = """Secuencia de ejecución recomendada:
1. InputGUI.py → AURA_Power stream
2. realtime_processor.py → FEATURE_STREAM
3. cnn_interference.py → CNN_COMMANDS
4. CommandSender.py

Nota: Esperar unos segundos entre cada inicio para permitir que los streams LSL se establezcan."""
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(pady=5)
        
        # Frame para los programas
        programs_frame = ttk.LabelFrame(main_frame, text="Programas", padding="10")
        programs_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Crear controles para cada programa
        self.program_controls = {}
        sorted_programs = sorted(self.processes.items(), key=lambda x: x[1]['order'])
        
        for program_name, info in sorted_programs:
            # Frame para cada programa
            program_frame = ttk.Frame(programs_frame)
            program_frame.pack(fill=tk.X, pady=5)
            
            # Número de orden
            order_label = ttk.Label(program_frame, 
                                  text=f"{info['order']}.", 
                                  width=3)
            order_label.pack(side=tk.LEFT, padx=5)
            
            # Nombre y descripción
            desc_frame = ttk.Frame(program_frame)
            desc_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            name_label = ttk.Label(desc_frame, 
                                 text=program_name, 
                                 font=('Helvetica', 10, 'bold'))
            name_label.pack(anchor=tk.W)
            
            # Descripción con streams
            stream_info = []
            if 'stream_in' in info:
                stream_info.append(f"In: {info['stream_in']}")
            if 'stream_out' in info:
                stream_info.append(f"Out: {info['stream_out']}")
            
            stream_text = f"{info['description']} ({' | '.join(stream_info)})"
            desc_label = ttk.Label(desc_frame, text=stream_text)
            desc_label.pack(anchor=tk.W)
            
            # Estado
            status_label = ttk.Label(program_frame, 
                                   text=info['status'],
                                   width=15)
            status_label.pack(side=tk.LEFT, padx=5)
            
            # Botones
            btn_frame = ttk.Frame(program_frame)
            btn_frame.pack(side=tk.RIGHT)
            
            start_btn = ttk.Button(btn_frame, 
                                 text="Iniciar",
                                 command=lambda p=program_name: self.start_program(p))
            start_btn.pack(side=tk.LEFT, padx=2)
            
            stop_btn = ttk.Button(btn_frame, 
                                text="Detener",
                                command=lambda p=program_name: self.stop_program(p),
                                state=tk.DISABLED)
            stop_btn.pack(side=tk.LEFT, padx=2)
            
            # Guardar referencias
            self.program_controls[program_name] = {
                'status_label': status_label,
                'start_btn': start_btn,
                'stop_btn': stop_btn
            }
        
        # Frame para controles globales
        control_frame = ttk.LabelFrame(main_frame, text="Control Global", padding="10")
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(control_frame, 
                  text="Iniciar Todos en Secuencia",
                  command=self.start_all_programs).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, 
                  text="Detener Todos",
                  command=self.stop_all_programs).pack(side=tk.LEFT, padx=5)
        
        # Área de log
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.log_text = tk.Text(log_frame, height=10, width=70)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = scrollbar.set
        
        # Configurar expansión
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
    
    def log_message(self, message):
        """Añade un mensaje al área de log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def start_program(self, program_name):
        """Inicia un programa específico"""
        try:
            if sys.platform.startswith('win'):
                process = subprocess.Popen(['start', 'cmd', '/k', 'python', program_name], 
                                        shell=True)
            else:
                process = subprocess.Popen(['gnome-terminal', '--', 'python', program_name])
            
            self.processes[program_name]["process"] = process
            self.program_controls[program_name]['start_btn'].state(['disabled'])
            self.program_controls[program_name]['stop_btn'].state(['!disabled'])
            self.program_controls[program_name]['status_label'].configure(
                text="Ejecutando", foreground="green")
            
            self.log_message(f"Programa {program_name} iniciado")
        except Exception as e:
            self.log_message(f"Error al iniciar {program_name}: {str(e)}")
    
    def stop_program(self, program_name):
        """Detiene un programa específico"""
        try:
            if self.processes[program_name]["process"]:
                if sys.platform.startswith('win'):
                    subprocess.run(['taskkill', '/F', '/T', '/PID', 
                                 str(self.processes[program_name]["process"].pid)])
                else:
                    self.processes[program_name]["process"].terminate()
                
                self.processes[program_name]["process"] = None
                self.program_controls[program_name]['start_btn'].state(['!disabled'])
                self.program_controls[program_name]['stop_btn'].state(['disabled'])
                self.program_controls[program_name]['status_label'].configure(
                    text="Detenido", foreground="red")
                
                self.log_message(f"Programa {program_name} detenido")
        except Exception as e:
            self.log_message(f"Error al detener {program_name}: {str(e)}")
    
    def start_all_programs(self):
        """Inicia todos los programas en secuencia"""
        def run_sequence():
            for program in self.processes.keys():
                if not self.processes[program]["process"]:
                    self.start_program(program)
                    time.sleep(2)  # Esperar 2 segundos entre programas
            
            self.log_message("Todos los programas han sido iniciados")
        
        # Ejecutar en un hilo separado para no bloquear la interfaz
        threading.Thread(target=run_sequence, daemon=True).start()
    
    def stop_all_programs(self):
        """Detiene todos los programas"""
        for program in self.processes.keys():
            if self.processes[program]["process"]:
                self.stop_program(program)
        
        self.log_message("Todos los programas han sido detenidos")
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        self.stop_all_programs()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ProgramController(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 