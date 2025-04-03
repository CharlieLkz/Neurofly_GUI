import tkinter as tk
from tkinter import messagebox, ttk
import os
import csv
import time
import threading
import queue
from datetime import datetime
from pylsl import StreamInlet, resolve_stream
from PIL import Image, ImageTk, ImageSequence

# === CONFIGURACIONES === #
STREAM_NAME = "AURA_Power"
SCALE_FACTOR_EEG = (4500000)/24/(2**23-1)
N_CANALES = 40
DURACION_REST = 20
DURACION_TAREA = 20
SUBTAREAS = [
    ("RightArm", "Thinking"),
    ("RightArm", "Moving"),
    ("LeftArm", "Thinking"),
    ("LeftArm", "Moving"),
    ("RightFist", "Thinking"),
    ("RightFist", "Moving"),
    ("LeftFist", "Thinking"),
    ("LeftFist", "Moving"),
    ("RightFoot", "Thinking"),
    ("RightFoot", "Moving"),
    ("LeftFoot", "Thinking"),
    ("LeftFoot", "Moving")
]

# === VARIABLES DE CONTROL === #
gif_jobs = []
is_rest_phase = False
ui_queue = queue.Queue()

# Clase para manejar el experimento
class ExperimentController:
    def __init__(self):
        self.running = False
        self.current_thread = None
        self.data_thread = None
        self.paused = False
        self._lock = threading.Lock()
        self.connected = False
        self.inlet = None

    def stop(self):
        with self._lock:
            self.running = False
        current_thread = threading.current_thread()
        if self.current_thread and self.current_thread.is_alive() and self.current_thread != current_thread:
            self.current_thread.join(timeout=1.0)
        if self.data_thread and self.data_thread.is_alive() and self.data_thread != current_thread:
            self.data_thread.join(timeout=1.0)
        self.connected = False
        self.inlet = None

    def reset_ui(self):
        ui_queue.put(('reset_ui', None))

    def conectar_stream(self):
        def connect_thread():
            try:
                ui_queue.put(('update_status', "Buscando dispositivo AURA...\nPor favor, espere."))
                timeout = 10.0
                start_time = time.time()
                while time.time() - start_time < timeout and not self.connected:
                    try:
                        streams = resolve_stream('name', STREAM_NAME)
                        if streams:
                            self.inlet = StreamInlet(streams[0])
                            with self._lock:
                                self.connected = True
                            ui_queue.put(('connection_success', None))
                            return True
                    except Exception:
                        time.sleep(0.2)
                if not self.connected:
                    ui_queue.put(('connection_error', "No se pudo encontrar el dispositivo AURA"))
                return False
            except Exception as e:
                ui_queue.put(('connection_error', f"Error al conectar con AURA: {str(e)}"))
                return False
        threading.Thread(target=connect_thread, daemon=True).start()

controller = ExperimentController()

# === FUNCIONES DE UI === #
def procesar_cola_ui():
    try:
        while not ui_queue.empty():
            action, data = ui_queue.get_nowait()
            if action == 'update_status':
                lbl.config(text=data)
            elif action == 'update_timer':
                if data == "Preparando...":
                    temporizador_label.config(text="")
                    preparacion_label.config(text=data)
                else:
                    temporizador_label.config(text=f"Tiempo restante: {data}s")
                    preparacion_label.config(text="")
            elif action == 'connection_success':
                lbl.config(text="Conexión establecida con AURA ✓\nComenzando experimento...")
                estado_conexion.config(text="Conectado ✓", fg="green")
                btn_comenzar.config(state='normal')
            elif action == 'connection_error':
                lbl.config(text="Error de conexión con AURA")
                estado_conexion.config(text="Desconectado ✗", fg="red")
                messagebox.showerror("Error de conexión", data)
                controller.stop()
            elif action == 'reset_ui':
                entry_nombre.config(state='normal')
                entry_nombre.delete(0, tk.END)
                btn_comenzar.config(state='normal')
                btn_pausar.config(state='disabled', text="Pausar")
                limpiar_animaciones()
                temporizador_label.config(text="Tiempo restante:")
                preparacion_label.config(text="")
                estado_conexion.config(text="Desconectado", fg="gray")
                lbl.config(text="Bienvenido/a al experimento de Neurofly\nIngresa tu nombre:")
            elif action == 'show_instruction':
                lbl.config(text=data)
            elif action == 'update_pause_button':
                btn_pausar.config(text="Continuar" if controller.paused else "Pausar")
    except Exception as e:
        print(f"Error en procesamiento de cola UI: {str(e)}")
    ventana.after(50, procesar_cola_ui)

def actualizar_temporizador(segundos_restantes):
    ui_queue.put(('update_timer', segundos_restantes))

def limpiar_animaciones():
    for job in gif_jobs:
        ventana.after_cancel(job)
    gif_jobs.clear()
    gif_label1.config(image='')
    gif_label2.config(image='')
    gif_label1.image = None
    gif_label2.image = None

def animar_gif(label, gif_frames, delay, index=0):
    if gif_frames:
        frame = gif_frames[index % len(gif_frames)]
        label.configure(image=frame)
        label.image = frame
        job = ventana.after(delay, lambda: animar_gif(label, gif_frames, delay, index + 1))
        gif_jobs.append(job)

def mostrar_gif_dual(tipo_accion, parte_cuerpo):
    limpiar_animaciones()
    path1 = os.path.join("gif", f"{tipo_accion}.gif")
    path2 = os.path.join("gif", f"{parte_cuerpo}.gif")
    if not os.path.exists(path1) or not os.path.exists(path2):
        return
    gif1_frames = [ImageTk.PhotoImage(img.resize((250, 250))) for img in ImageSequence.Iterator(Image.open(path1))]
    gif2_frames = [ImageTk.PhotoImage(img.resize((250, 250))) for img in ImageSequence.Iterator(Image.open(path2))]
    gif_label1.grid(row=0, column=0, padx=(200, 10), pady=(20, 20), sticky= "e")
    gif_label2.grid(row=0, column=1, padx=(10, 200), pady=(20, 20), sticky= "w")
    animar_gif(gif_label1, gif1_frames, 100)
    animar_gif(gif_label2, gif2_frames, 100)

def mostrar_gif_rest():
    limpiar_animaciones()
    path = os.path.join("gif", "Breathe.gif")
    if not os.path.exists(path):
        return
    gif_frames = [ImageTk.PhotoImage(img.resize((250, 250))) for img in ImageSequence.Iterator(Image.open(path))]
    gif_label1.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    animar_gif(gif_label1, gif_frames, 100)

def capturar_datos(nombre, tarea, subtarea, duracion):
    carpeta = os.path.join(os.path.dirname(__file__), "data", nombre)
    os.makedirs(carpeta, exist_ok=True)
    archivo = os.path.join(carpeta, f"{nombre}_{tarea}_{subtarea}.csv")
    def data_thread():
        with open(archivo, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp"] + [f"Canal{i+1}" for i in range(N_CANALES)])
            if not controller.connected or not controller.inlet:
                ui_queue.put(('connection_error', "No hay conexión con AURA"))
                return
            start = time.time()
            while controller.running and (elapsed := time.time() - start) < duracion:
                sample, timestamp = controller.inlet.pull_sample(timeout=0.1)
                if sample:
                    muestra = [float(x) * SCALE_FACTOR_EEG for x in sample[:N_CANALES]]
                    writer.writerow([timestamp] + muestra)
                    actualizar_temporizador(int(duracion - elapsed))
                time.sleep(0.01)
    controller.data_thread = threading.Thread(target=data_thread, daemon=True)
    controller.data_thread.start()

def iniciar_experimento():
    if controller.running:
        return
    nombre = entry_nombre.get().strip()
    if not nombre:
        messagebox.showerror("Error", "Debes ingresar un nombre")
        return
    if not controller.connected:
        ui_queue.put(('update_status', "Intentando conectar con AURA..."))
        btn_comenzar.config(state='disabled')
        controller.conectar_stream()
        return
    controller.running = True
    entry_nombre.config(state='disabled')
    btn_comenzar.config(state='disabled')
    btn_pausar.config(state='normal')
    def secuencia():
        try:
            ui_queue.put(('show_instruction', f"Capturando REST para: {nombre}\nRelájate y respira..."))
            mostrar_gif_rest()
            archivo_rest = os.path.join(os.path.dirname(__file__), "data", nombre, f"{nombre}_Rest_Base.csv")
            os.makedirs(os.path.dirname(archivo_rest), exist_ok=True)
            with open(archivo_rest, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp"] + [f"Canal{i+1}" for i in range(N_CANALES)])
                start = time.time()
                while controller.running and (elapsed := time.time() - start) < DURACION_REST:
                    sample, timestamp = controller.inlet.pull_sample(timeout=0.1)
                    if sample:
                        muestra = [float(x) * SCALE_FACTOR_EEG for x in sample[:N_CANALES]]
                        writer.writerow([timestamp] + muestra)
                        actualizar_temporizador(int(DURACION_REST - elapsed))
                    time.sleep(0.01)
            for parte, tipo in SUBTAREAS:
                instrucciones_titulo = f"{tipo.upper()}\n{parte.upper()}"
                if tipo == "Moving":
                    if "Fist" in parte:
                        instrucciones_texto = "Cierra el puño durante 1 segundo\ny luego suéltalo durante 1 segundo.\nRepite esto hasta que se acabe el tiempo."
                    elif "Foot" in parte:
                        instrucciones_texto = "Despega la planta del pie del suelo,\nmantenla en el aire durante 1 segundo\ny vuelve a bajarla.\nRepite esto hasta que finalice el tiempo."
                    else:
                        instrucciones_texto = "Levanta tu brazo hacia adelante en 1 segundo,\nmantenlo arriba 1 segundo y bájalo en 1 segundo.\nRepite esto hasta que termine."
                else:
                    instrucciones_texto = "Concéntrate en imaginar el movimiento\nsin realizarlo."
                instruccion = f"{instrucciones_titulo}\n\n{instrucciones_texto}"
                ui_queue.put(('show_instruction', instruccion))
                ui_queue.put(('update_timer', "Preparando..."))
                time.sleep(3)
                mostrar_gif_dual(tipo, parte)
                time.sleep(1)
                capturar_datos(nombre, parte, tipo, DURACION_TAREA)
                while controller.data_thread.is_alive():
                    time.sleep(0.1)
            limpiar_animaciones()
            ui_queue.put(('show_instruction', "¡Gracias por participar!"))
            ventana.after(0, lambda: messagebox.showinfo("Finalizado", "Captura completa para este participante."))
            controller.stop()
            controller.reset_ui()
        except Exception as e:
            ui_queue.put(('connection_error', f"Error en la secuencia: {str(e)}"))
            controller.stop()
            controller.reset_ui()
    controller.current_thread = threading.Thread(target=secuencia, daemon=True)
    controller.current_thread.start()

def toggle_pausa():
    controller.paused = not controller.paused
    ui_queue.put(('update_pause_button', None))

def salir_aplicacion():
    controller.stop()
    ventana.quit()

ventana = tk.Tk()
ventana.title("Neurofly GUI Experiment")
ventana.configure(bg="#002147")
ventana.geometry("1920x1080")
ventana.resizable(True, True)

main_frame = ttk.Frame(ventana, padding=20)
main_frame.pack(fill=tk.BOTH, expand=True)

style = ttk.Style()
style.configure('TFrame', background='#002147')
style.configure('TLabel', background='#002147', foreground='white', font=('Arial', 16))
style.configure('TButton', font=('Arial', 16))

lbl = tk.Label(main_frame, text="Bienvenido/a al experimento de Neurofly\nIngresa tu nombre:",
              font=("Arial", 36, "bold"), bg="#002147", fg="white", wraplength=1800)
lbl.pack(pady=20)

entry_frame = tk.Frame(main_frame, bg="#002147")
entry_frame.pack(pady=10)

entry_nombre = tk.Entry(entry_frame, width=30, font=("Arial", 28))
entry_nombre.pack(side=tk.LEFT, padx=10)

estado_conexion = tk.Label(entry_frame, text="Desconectado", font=("Arial", 16),
                          bg="#002147", fg="gray", padx=10)
estado_conexion.pack(side=tk.LEFT, padx=10)

button_frame = tk.Frame(main_frame, bg="#002147")
button_frame.pack(pady=20)

btn_comenzar = tk.Button(button_frame, text="Comenzar Experimento", font=("Arial", 28),
                        command=iniciar_experimento, bg="#006699", fg="white")
btn_comenzar.pack(side=tk.LEFT, padx=10)

btn_pausar = tk.Button(button_frame, text="Pausar", font=("Arial", 20),
                      command=toggle_pausa, bg="#FFA500", fg="white", state='disabled')
btn_pausar.pack(side=tk.LEFT, padx=10)

btn_salir = tk.Button(button_frame, text="Salir", font=("Arial", 20),
                    command=salir_aplicacion, bg="#990000", fg="white")
btn_salir.pack(side=tk.LEFT, padx=10)

temporizador_label = tk.Label(main_frame, text="Tiempo restante:", font=("Arial", 28),
                             bg="#002147", fg="white")
temporizador_label.pack(pady=10)

preparacion_label = tk.Label(main_frame, text="", font=("Arial", 24, "italic"),
                           bg="#002147", fg="#FFA500")
preparacion_label.pack(pady=5)

frame_gifs = tk.Frame(main_frame, bg="#002147", height=300)
frame_gifs.pack(pady=20, fill=tk.X)
frame_gifs.pack_propagate(False)

frame_gifs.columnconfigure(0, weight=1)
frame_gifs.columnconfigure(1, weight=1)

gif_label1 = tk.Label(frame_gifs, bg="#002147")
gif_label1.grid(row=0, column=0)

gif_label2 = tk.Label(frame_gifs, bg="#002147")
gif_label2.grid(row=0, column=1)

ventana.protocol("WM_DELETE_WINDOW", salir_aplicacion)
ventana.after(100, procesar_cola_ui)
ventana.mainloop()
