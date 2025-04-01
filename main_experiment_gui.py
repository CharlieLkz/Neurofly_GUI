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
STREAM_NAME = "AURA"
SCALE_FACTOR_EEG = (4500000)/24/(2**23-1)
N_CANALES = 8
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
ui_queue = queue.Queue()  # Cola para comunicación segura con la UI

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
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.join(timeout=1.0)
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)
        self.connected = False
        self.inlet = None

    def reset_ui(self):
        ui_queue.put(('reset_ui', None))

    def conectar_stream(self):
        """Intenta conectar al stream AURA en un hilo separado"""
        def connect_thread():
            try:
                ui_queue.put(('update_status', "Buscando dispositivo AURA...\nPor favor, espere."))
                
                # Intentar encontrar el stream con timeout
                timeout = 10.0  # segundos
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
                    except Exception as e:
                        time.sleep(0.2)  # Pequeña pausa entre intentos
                
                if not self.connected:
                    ui_queue.put(('connection_error', "No se pudo encontrar el dispositivo AURA"))
                return False
                
            except Exception as e:
                ui_queue.put(('connection_error', f"Error al conectar con AURA: {str(e)}"))
                return False

        threading.Thread(target=connect_thread, daemon=True).start()

# Crear la instancia del controlador
controller = ExperimentController()

# === FUNCIONES DE UI === #
def procesar_cola_ui():
    """Procesa los mensajes en la cola de UI de forma segura"""
    try:
        while not ui_queue.empty():
            action, data = ui_queue.get_nowait()
            
            if action == 'update_status':
                lbl.config(text=data)
                ventana.update_idletasks()
            
            elif action == 'update_timer':
                if data == "Preparando...":
                    temporizador_label.config(text="")
                    preparacion_label.config(text=data)
                else:
                    temporizador_label.config(text=f"Tiempo restante: {data}s")
                    preparacion_label.config(text="")
                ventana.update_idletasks()
            
            elif action == 'connection_success':
                lbl.config(text="Conexión establecida con AURA ✓\nComenzando experimento...")
                estado_conexion.config(text="Conectado ✓", fg="green")
                btn_comenzar.config(state='normal')
                ventana.update_idletasks()
            
            elif action == 'connection_error':
                lbl.config(text="Error de conexión con AURA")
                estado_conexion.config(text="Desconectado ✗", fg="red")
                messagebox.showerror("Error de conexión", data)
                controller.stop()
                ventana.update_idletasks()
            
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
                ventana.update_idletasks()
            
            elif action == 'show_instruction':
                lbl.config(text=data)
                ventana.update_idletasks()
            
            elif action == 'update_pause_button':
                if controller.paused:
                    btn_pausar.config(text="Continuar")
                else:
                    btn_pausar.config(text="Pausar")
                ventana.update_idletasks()
    
    except Exception as e:
        print(f"Error en procesamiento de cola UI: {str(e)}")
    
    # Programar la próxima verificación de la cola
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
    if index < len(gif_frames):
        frame = gif_frames[index]
        label.configure(image=frame)
        label.image = frame
        job = ventana.after(delay, lambda: animar_gif(label, gif_frames, delay, (index + 1) % len(gif_frames)))
        gif_jobs.append(job)

def mostrar_gif_dual(tipo_accion, parte_cuerpo):
    try:
        limpiar_animaciones()
        path1 = os.path.join("gif", f"{tipo_accion}.gif")
        path2 = os.path.join("gif", f"{parte_cuerpo}.gif")
        
        if not (os.path.exists(path1) and os.path.exists(path2)):
            raise FileNotFoundError(f"No se encontraron los archivos GIF necesarios")
        
        # Usar threading para cargar las imágenes y no bloquear la UI
        def load_gifs():
            try:
                gif1 = Image.open(path1)
                gif2 = Image.open(path2)
                
                gif1_frames = [ImageTk.PhotoImage(img.resize((250, 250))) for img in ImageSequence.Iterator(gif1)]
                gif2_frames = [ImageTk.PhotoImage(img.resize((250, 250))) for img in ImageSequence.Iterator(gif2)]
                
                # Calcular posiciones
                frame_width = frame_gifs.winfo_width() or 1000  # Valor por defecto si aún no está calculado
                gif_width = 250
                padding = max(50, (frame_width - (2 * gif_width)) // 3)
                
                ventana.after(0, lambda: gif_label1.grid(row=0, column=0, padx=(padding, padding//2), pady=(20, 20)))
                ventana.after(0, lambda: gif_label2.grid(row=0, column=1, padx=(padding//2, padding), pady=(20, 20)))
                ventana.after(0, lambda: animar_gif(gif_label1, gif1_frames, 100))
                ventana.after(0, lambda: animar_gif(gif_label2, gif2_frames, 100))
            
            except Exception as e:
                ui_queue.put(('connection_error', f"Error al cargar GIFs: {str(e)}"))
        
        threading.Thread(target=load_gifs, daemon=True).start()
    
    except Exception as e:
        ui_queue.put(('connection_error', f"Error al iniciar carga de GIFs: {str(e)}"))

def capturar_datos(nombre_participante, tarea, subtarea, duracion):
    try:
        carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", nombre_participante)
        os.makedirs(carpeta, exist_ok=True)
        archivo = os.path.join(carpeta, f"{nombre_participante}_{tarea}_{subtarea}.csv")

        def data_collection():
            try:
                with open(archivo, mode='w', newline='') as f:
                    escritor = csv.writer(f)
                    escritor.writerow(["Timestamp"] + [f"Canal{i+1}" for i in range(N_CANALES)])

                    # Usar la conexión ya establecida
                    if not controller.connected or not controller.inlet:
                        ui_queue.put(('connection_error', "No hay conexión con AURA"))
                        return

                    tiempo_inicio = time.time()
                    tiempo_pausado = 0
                    ultima_pausa = 0
                    
                    while controller.running and (tiempo_actual := time.time() - tiempo_inicio - tiempo_pausado) < duracion:
                        if controller.paused:
                            if ultima_pausa == 0:  # Primera vez en pausa
                                ultima_pausa = time.time()
                            time.sleep(0.1)
                            continue
                        else:
                            if ultima_pausa > 0:  # Salimos de pausa
                                tiempo_pausado += time.time() - ultima_pausa
                                ultima_pausa = 0
                                
                        try:
                            sample, timestamp = controller.inlet.pull_sample(timeout=0.1)
                            if sample:
                                muestra = [float(x) * SCALE_FACTOR_EEG for x in sample[:N_CANALES]]
                                escritor.writerow([timestamp] + muestra)
                                segundos_restantes = int(duracion - tiempo_actual)
                                actualizar_temporizador(segundos_restantes)
                        except Exception as e:
                            print(f"Error en captura: {str(e)}")
                            time.sleep(0.01)  # Evitar bucle muy rápido en caso de error

            except Exception as e:
                ui_queue.put(('connection_error', f"Error en la captura de datos: {str(e)}"))

        controller.data_thread = threading.Thread(target=data_collection, daemon=True)
        controller.data_thread.start()
        return True
        
    except Exception as e:
        ui_queue.put(('connection_error', f"Error al iniciar la captura: {str(e)}"))
        return False

def toggle_pausa():
    controller.paused = not controller.paused
    ui_queue.put(('update_pause_button', None))

def iniciar_experimento():
    if controller.running:
        return  # Evitar múltiples inicios

    nombre = entry_nombre.get().strip()
    if not nombre:
        messagebox.showerror("Error", "Debes ingresar un nombre")
        return

    # Iniciar la conexión al stream
    if not controller.connected:
        ui_queue.put(('update_status', "Intentando conectar con AURA...\nPor favor, espere."))
        btn_comenzar.config(state='disabled')
        controller.conectar_stream()
        # El resto se manejará cuando la conexión sea exitosa en procesar_cola_ui
        return

    controller.running = True
    entry_nombre.config(state='disabled')
    btn_comenzar.config(state='disabled')
    btn_pausar.config(state='normal')
    
    def secuencia():
        try:
            # Primero realizar la fase de REST
            ui_queue.put(('show_instruction', f"Capturando REST para: {nombre}\nRelájate y respira..."))
            
            # Capturar datos REST
            carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", nombre)
            os.makedirs(carpeta, exist_ok=True)
            archivo_rest = os.path.join(carpeta, f"{nombre}_Rest_Base.csv")
            
            def capturar_rest():
                try:
                    with open(archivo_rest, mode='w', newline='') as f:
                        escritor = csv.writer(f)
                        escritor.writerow(["Timestamp"] + [f"Canal{i+1}" for i in range(N_CANALES)])
                        
                        if not controller.connected or not controller.inlet:
                            ui_queue.put(('connection_error', "No hay conexión con AURA durante REST"))
                            return
                            
                        tiempo_inicio = time.time()
                        while controller.running and (time.time() - tiempo_inicio) < DURACION_REST:
                            if controller.paused:
                                time.sleep(0.1)
                                continue
                                
                            try:
                                sample, timestamp = controller.inlet.pull_sample(timeout=0.1)
                                if sample:
                                    muestra = [float(x) * SCALE_FACTOR_EEG for x in sample[:N_CANALES]]
                                    escritor.writerow([timestamp] + muestra)
                                    segundos_restantes = int(DURACION_REST - (time.time() - tiempo_inicio))
                                    actualizar_temporizador(segundos_restantes)
                            except Exception as e:
                                print(f"Error en captura REST: {str(e)}")
                                time.sleep(0.01)
                except Exception as e:
                    ui_queue.put(('connection_error', f"Error en la captura REST: {str(e)}"))
            
            # Ejecutar captura REST en hilo separado
            rest_thread = threading.Thread(target=capturar_rest, daemon=True)
            rest_thread.start()
            
            # Esperar a que termine la fase REST
            tiempo_inicio_rest = time.time()
            while rest_thread.is_alive() and time.time() - tiempo_inicio_rest < DURACION_REST + 5:
                time.sleep(0.1)
            
            if not controller.running:
                return
            
            # Definir una función específica para la preparación
            def mostrar_preparacion():
                ui_queue.put(('update_timer', "Preparando..."))
            
            # Continuar con las tareas regulares
            def ejecutar_tarea(parte, tipo):
                if not controller.running:
                    return False
                
                # Preparar instrucciones
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
                
                instruccion_completa = f"{instrucciones_titulo}\n\n{instrucciones_texto}"
                ui_queue.put(('show_instruction', instruccion_completa))
                
                # Mostrar mensaje de preparación en área separada
                mostrar_preparacion()
                time.sleep(3)  # Pausa entre tareas
                
                # Mostrar GIFs
                mostrar_gif_dual(tipo, parte)
                time.sleep(1)  # Pausa para asegurar que los GIFs se carguen
                
                # Limpiar el mensaje de preparación antes de empezar la captura
                ui_queue.put(('update_timer', ""))
                
                # Capturar datos
                if not controller.running:
                    return False
                    
                if not capturar_datos(nombre, parte, tipo, DURACION_TAREA):
                    return False
                    
                # Esperar a que termine la captura
                tiempo_inicio = time.time()
                while controller.running and controller.data_thread.is_alive() and time.time() - tiempo_inicio < DURACION_TAREA + 5:
                    time.sleep(0.1)
                    
                return True

            # Ejecutar todas las subtareas
            for parte, tipo in SUBTAREAS:
                if not ejecutar_tarea(parte, tipo):
                    break

            # Finalizar si todo salió bien
            if controller.running:
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

def salir_aplicacion():
    controller.stop()
    ventana.quit()

# === GUI === #
ventana = tk.Tk()
ventana.title("Neurofly GUI Experiment")
ventana.configure(bg="#002147")
ventana.geometry("1920x1080")
ventana.resizable(True, True)

# Frame principal con diseño responsivo
main_frame = ttk.Frame(ventana, padding=20)
main_frame.pack(fill=tk.BOTH, expand=True)

# Estilo para los widgets
style = ttk.Style()
style.configure('TFrame', background='#002147')
style.configure('TLabel', background='#002147', foreground='white', font=('Arial', 16))
style.configure('TButton', font=('Arial', 16))

# Título principal
lbl = tk.Label(main_frame, text="Bienvenido/a al experimento de Neurofly\nIngresa tu nombre:", 
              font=("Arial", 36, "bold"), bg="#002147", fg="white", wraplength=1800)
lbl.pack(pady=20)

# Frame para entrada de nombre
entry_frame = tk.Frame(main_frame, bg="#002147")
entry_frame.pack(pady=10)

entry_nombre = tk.Entry(entry_frame, width=30, font=("Arial", 28))
entry_nombre.pack(side=tk.LEFT, padx=10)

# Estado de conexión
estado_conexion = tk.Label(entry_frame, text="Desconectado", font=("Arial", 16), 
                          bg="#002147", fg="gray", padx=10)
estado_conexion.pack(side=tk.LEFT, padx=10)

# Frame para botones
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

# Temporizador
temporizador_label = tk.Label(main_frame, text="Tiempo restante:", font=("Arial", 28), 
                             bg="#002147", fg="white")
temporizador_label.pack(pady=10)

# Etiqueta para mensajes de preparación
preparacion_label = tk.Label(main_frame, text="", font=("Arial", 24, "italic"), 
                           bg="#002147", fg="#FFA500")  # Color naranja para destacar
preparacion_label.pack(pady=5)

# Frame para GIFs con tamaño fijo
frame_gifs = tk.Frame(main_frame, bg="#002147", height=300)
frame_gifs.pack(pady=20, fill=tk.X)
frame_gifs.pack_propagate(False)  # Mantener tamaño fijo

frame_gifs.columnconfigure(0, weight=1)
frame_gifs.columnconfigure(1, weight=1)

gif_label1 = tk.Label(frame_gifs, bg="#002147")
gif_label1.grid(row=0, column=0)

gif_label2 = tk.Label(frame_gifs, bg="#002147")
gif_label2.grid(row=0, column=1)

# Configurar para cerrar correctamente
ventana.protocol("WM_DELETE_WINDOW", salir_aplicacion)

# Iniciar el procesamiento de la cola de UI
ventana.after(100, procesar_cola_ui)

ventana.mainloop()