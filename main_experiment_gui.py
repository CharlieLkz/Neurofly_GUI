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
N_CANALES = 8  # Número de canales físicos
BANDAS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]  # Bandas de frecuencia
TOTAL_COLUMNAS = N_CANALES * len(BANDAS)  # Total de columnas de datos (8 canales x 5 bandas = 40)
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
    ("LeftFoot", "Moving"),
    ("Flecha Izquierda", "Thinking"),
    ("Flecha Derecha", "Thinking"),
    ("Flecha Arriba", "Thinking"),
    ("Flecha Abajo", "Thinking")
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
    if hasattr(gif_label1, 'image'):
        gif_label1.config(image='')
        gif_label1.image = None
    if hasattr(gif_label2, 'image'):
        gif_label2.config(image='')
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
    gif1_frames = [ImageTk.PhotoImage(img.resize((200, 200))) for img in ImageSequence.Iterator(Image.open(path1))]
    gif2_frames = [ImageTk.PhotoImage(img.resize((200, 200))) for img in ImageSequence.Iterator(Image.open(path2))]
    
    # Animar los GIFs en sus respectivas etiquetas
    animar_gif(gif_label1, gif1_frames, 100)
    animar_gif(gif_label2, gif2_frames, 100)

def mostrar_gif_rest():
    limpiar_animaciones()
    path = os.path.join("gif", "Breathe.gif")
    if not os.path.exists(path):
        return
    gif_frames = [ImageTk.PhotoImage(img.resize((200, 200))) for img in ImageSequence.Iterator(Image.open(path))]
    
    # Centrar el GIF de descanso en el contenedor central
    animar_gif(gif_label1, gif_frames, 100)

def capturar_datos(nombre, tarea, subtarea, duracion):
    carpeta = os.path.join(os.path.dirname(__file__), "data", nombre)
    os.makedirs(carpeta, exist_ok=True)
    
    # Cambiar el sufijo a "VI" para las flechas, manteniendo el original para el resto
    if tarea in ["Flecha Arriba", "Flecha Abajo", "Flecha Izquierda", "Flecha Derecha"]:
        # Eliminar el espacio para el nombre del archivo
        tarea_sin_espacio = tarea.replace(" ", "")
        archivo = os.path.join(carpeta, f"{nombre}_{tarea_sin_espacio}_VI.csv")
        tipo_tarea = "Visual Imagery"
    else:
        archivo = os.path.join(carpeta, f"{nombre}_{tarea}_{subtarea}.csv")
        tipo_tarea = subtarea
    
    # Crear encabezados para las bandas de frecuencia para cada canal
    encabezados = ["Timestamp"]
    for i in range(N_CANALES):
        for banda in BANDAS:
            encabezados.append(f"Canal{i+1}_{banda}")
    
    # Añadir metadatos para ayudar en el análisis
    metadatos = {
        "Nombre": nombre,
        "Tarea": tarea,
        "Tipo": tipo_tarea,
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Duración": duracion
    }
    def data_thread():
        with open(archivo, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Escribir metadatos como comentarios
            for key, value in metadatos.items():
                f.write(f"# {key}: {value}\n")
            writer.writerow(encabezados)
            if not controller.connected or not controller.inlet:
                ui_queue.put(('connection_error', "No hay conexión con AURA"))
                return
            start = time.time()
            while controller.running and (elapsed := time.time() - start) < duracion:
                sample, timestamp = controller.inlet.pull_sample(timeout=0.1)
                if sample:
                    # Aplicar factor de escala a los datos
                    muestra = [float(x) * SCALE_FACTOR_EEG for x in sample[:TOTAL_COLUMNAS]]
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
            
            # Crear encabezados para las bandas de frecuencia para cada canal
            encabezados = ["Timestamp"]
            for i in range(N_CANALES):
                for banda in BANDAS:
                    encabezados.append(f"Canal{i+1}_{banda}")
            
            # Añadir metadatos para REST
            metadatos_rest = {
                "Nombre": nombre,
                "Tarea": "Rest",
                "Tipo": "Base",
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Duración": DURACION_REST
            }
            
            with open(archivo_rest, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Escribir metadatos como comentarios
                for key, value in metadatos_rest.items():
                    f.write(f"# {key}: {value}\n")
                writer.writerow(encabezados)
                start = time.time()
                while controller.running and (elapsed := time.time() - start) < DURACION_REST:
                    sample, timestamp = controller.inlet.pull_sample(timeout=0.1)
                    if sample:
                        muestra = [float(x) * SCALE_FACTOR_EEG for x in sample[:TOTAL_COLUMNAS]]
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
                    if "Flecha Arriba" in parte:
                        instrucciones_titulo = "VISUAL IMAGERY\nFLECHA ARRIBA"
                        instrucciones_texto = "Concéntrate en la dirección de la flecha hacia arriba\nsin realizar ningún movimiento."
                    elif "Flecha Abajo" in parte:
                        instrucciones_titulo = "VISUAL IMAGERY\nFLECHA ABAJO"
                        instrucciones_texto = "Concéntrate en la dirección de la flecha hacia abajo\nsin realizar ningún movimiento."
                    elif "Flecha Izquierda" in parte:
                        instrucciones_titulo = "VISUAL IMAGERY\nFLECHA IZQUIERDA"
                        instrucciones_texto = "Concéntrate en la dirección de la flecha hacia la izquierda\nsin realizar ningún movimiento."
                    elif "Flecha Derecha" in parte:
                        instrucciones_titulo = "VISUAL IMAGERY\nFLECHA DERECHA"
                        instrucciones_texto = "Concéntrate en la dirección de la flecha hacia la derecha\nsin realizar ningún movimiento."
                    else:
                        instrucciones_titulo = "MOTOR IMAGERY\n" + parte.upper()
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
              font=("Arial", 32, "bold"), bg="#002147", fg="white", wraplength=1800)
lbl.pack(pady=15)

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

temporizador_label = tk.Label(main_frame, text="Tiempo restante:", font=("Arial", 24),
                             bg="#002147", fg="white")
temporizador_label.pack(pady=8)

preparacion_label = tk.Label(main_frame, text="", font=("Arial", 22, "italic"),
                           bg="#002147", fg="#FFA500")
preparacion_label.pack(pady=4)

frame_gifs = tk.Frame(main_frame, bg="#002147", height=350, width=800)
frame_gifs.pack(pady=(0, 15), fill=tk.X)  # Reducir padding inferior para subir los gifs
frame_gifs.pack_propagate(False)
frame_gifs.grid_propagate(False)

# Contenedor central para mejorar la alineación
center_container = tk.Frame(frame_gifs, bg="#002147")
center_container.place(relx=0.5, rely=0.4, anchor="center")  # Alinear al centro pero más arriba

# Configurar los GIFs dentro del contenedor central
gif_label1 = tk.Label(center_container, bg="#002147")
gif_label1.grid(row=0, column=0, padx=10)

gif_label2 = tk.Label(center_container, bg="#002147")
gif_label2.grid(row=0, column=1, padx=10)

ventana.protocol("WM_DELETE_WINDOW", salir_aplicacion)
ventana.after(100, procesar_cola_ui)
ventana.mainloop()