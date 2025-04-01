import tkinter as tk
from tkinter import messagebox
import os
import csv
import time
import threading
from datetime import datetime
from pylsl import StreamInlet, resolve_stream
from PIL import Image, ImageTk, ImageSequence

# === CONFIGURACIONES === #
STREAM_NAME = "AURA"
SCALE_FACTOR_EEG = (4500000)/24/(2**23-1)  # µV/count
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

# === FUNCIONES === #
def capturar_datos(nombre_participante, tarea, subtarea, duracion):
    carpeta = os.path.join(os.path.dirname(__file__), nombre_participante)
    os.makedirs(carpeta, exist_ok=True)
    archivo = os.path.join(carpeta, f"{nombre_participante}_{tarea}_{subtarea}.csv")

    with open(archivo, mode='w', newline='') as f:
        escritor = csv.writer(f)
        escritor.writerow(["Timestamp"] + [f"Canal{i+1}" for i in range(N_CANALES)])

        streams = resolve_stream('name', STREAM_NAME)
        inlet = StreamInlet(streams[0])

        tiempo_inicio = time.time()
        while (time.time() - tiempo_inicio) < duracion:
            sample, timestamp = inlet.pull_sample()
            if sample:
                muestra = [float(x) * SCALE_FACTOR_EEG for x in sample[:N_CANALES]]
                escritor.writerow([timestamp] + muestra)
            segundos_restantes = int(duracion - (time.time() - tiempo_inicio))
            actualizar_temporizador(segundos_restantes)
            time.sleep(0.01)

def actualizar_temporizador(segundos_restantes):
    temporizador_label.config(text=f"Tiempo restante: {segundos_restantes}s")
    ventana.update_idletasks()

def limpiar_animaciones():
    for job in gif_jobs:
        ventana.after_cancel(job)
    gif_jobs.clear()
    gif_label1.config(image='')
    gif_label2.config(image='')
    gif_label1.image = None
    gif_label2.image = None

def animar_gif(label, gif_frames, delay, index=0):
    if not gif_frames:
        return
    frame = gif_frames[index]
    label.configure(image=frame)
    label.image = frame
    job = ventana.after(delay, lambda: animar_gif(label, gif_frames, delay, (index + 1) % len(gif_frames)))
    gif_jobs.append(job)

def mostrar_gif_dual(tipo_accion, parte_cuerpo):
    limpiar_animaciones()
    gif_dir = os.path.join(os.path.dirname(__file__), "gif")
    path1 = os.path.join(gif_dir, f"{tipo_accion}.gif")
    path2 = os.path.join(gif_dir, f"{parte_cuerpo}.gif")

    gif1_frames = [ImageTk.PhotoImage(img.resize((250, 250))) for img in ImageSequence.Iterator(Image.open(path1))]
    gif2_frames = [ImageTk.PhotoImage(img.resize((250, 250))) for img in ImageSequence.Iterator(Image.open(path2))]

    animar_gif(gif_label1, gif1_frames, 100)
    animar_gif(gif_label2, gif2_frames, 100)

def mostrar_gif_rest():
    limpiar_animaciones()
    gif_dir = os.path.join(os.path.dirname(__file__), "gif")
    path = os.path.join(gif_dir, "Breathe.gif")
    gif_frames = [ImageTk.PhotoImage(img.resize((250, 250))) for img in ImageSequence.Iterator(Image.open(path))]
    animar_gif(gif_label1, gif_frames, 100)

def iniciar_experimento():
    global is_rest_phase
    nombre = entry_nombre.get().strip()
    if not nombre:
        messagebox.showerror("Error", "Debes ingresar un nombre")
        return

    lbl.config(text=f"Capturando REST para: {nombre}\nRelájate y respira...", font=("Arial", 36, "bold"))
    entry_nombre.config(state='disabled')
    btn_comenzar.config(state='disabled')

    def secuencia():
        global is_rest_phase
        is_rest_phase = True
        mostrar_gif_rest()
        capturar_datos(nombre, "Rest", "Base", DURACION_REST)
        is_rest_phase = False

        for parte, tipo in SUBTAREAS:
            limpiar_animaciones()
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

            lbl.config(text=f"{instrucciones_titulo}\n\n{instrucciones_texto}", font=("Arial", 36, "bold"))
            temporizador_label.config(text="Descansa 3 segundos...")
            time.sleep(3)
            mostrar_gif_dual(tipo, parte)
            capturar_datos(nombre, parte, tipo, DURACION_TAREA)

        limpiar_animaciones()
        lbl.config(text="¡Gracias por participar!", font=("Arial", 48, "bold"))
        messagebox.showinfo("Finalizado", "Captura completa para este participante.")
        ventana.quit()

    threading.Thread(target=secuencia, daemon=True).start()

def salir_aplicacion():
    ventana.quit()

# === GUI === #
ventana = tk.Tk()
ventana.title("Neurofly GUI Experiment")
ventana.configure(bg="#002147")
ventana.geometry("1920x1080")
ventana.resizable(True, True)

lbl = tk.Label(ventana, text="Bienvenido/a al experimento de Neurofly\nIngresa tu nombre:", font=("Arial", 36, "bold"), bg="#002147", fg="white")
lbl.pack(pady=30)

entry_nombre = tk.Entry(ventana, width=30, font=("Arial", 28))
entry_nombre.pack(pady=10)

btn_comenzar = tk.Button(ventana, text="Comenzar Experimento", font=("Arial", 28), command=iniciar_experimento, bg="#006699", fg="white")
btn_comenzar.pack(pady=20)

btn_salir = tk.Button(ventana, text="Salir", font=("Arial", 20), command=salir_aplicacion, bg="#990000", fg="white")
btn_salir.pack(pady=10)

temporizador_label = tk.Label(ventana, text="Tiempo restante:", font=("Arial", 28), bg="#002147", fg="white")
temporizador_label.pack(pady=10)

frame_gifs = tk.Frame(ventana, bg="#002147")
frame_gifs.pack(pady=20, expand=True, fill=tk.BOTH)

frame_gifs.columnconfigure(0, weight=1)
frame_gifs.columnconfigure(1, weight=1)

gif_label1 = tk.Label(frame_gifs, bg="#002147")
gif_label1.grid(row=0, column=0, sticky="nw", padx=(66, 0), pady=(0, 300))

gif_label2 = tk.Label(frame_gifs, bg="#002147")
gif_label2.grid(row=0, column=1, sticky="ne", padx=(0, 66), pady=(0, 180))

ventana.mainloop()
