#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para analizar, estandarizar y unificar archivos CSV de datos EEG
del proyecto Neurofly, detectando inconsistencias y aplicando bandpower
donde sea necesario.

Este script:
1. Detecta diferencias en número de columnas (8 vs 40)
2. Identifica presencia o ausencia de metadatos
3. Corrige problemas de codificación
4. Estandariza nombres de columnas
5. Aplica bandpower a archivos raw si es necesario
6. Genera un reporte de inconsistencias
7. Genera gráficos para visualización de datos
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import json
import chardet 
import re
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import traceback

# Constantes y configuración global
# Definiciones de bandas de frecuencia cerebral
ALPHA = [8.0, 12.0]
BETA = [12.0, 30.0]
GAMMA = [30.0, 60.0]
DELTA = [0.0, 4.0]
THETA = [4.0, 8.0]

# Frecuencia de muestreo y parámetros
SAMPLING_FREQUENCY = 250  # Hz
EPOCH_SIZE = 250
WINDOW_SIZE = 50
CHANNELS = 8

# Colores para cada banda de frecuencia en gráficos
COLOR_MAP = {
    'Alpha': 'blue',
    'Beta': 'red',
    'Gamma': 'green',
    'Delta': 'purple',
    'Theta': 'orange'
}

# Canales de interés (C3 y C4, que corresponden a los electrodos 4 y 5)
CANALES_INTERES = [4, 5]
NOMBRES_CANALES = {4: 'C3', 5: 'C4'}

# Bandas de interés para el análisis específico
BANDAS_INTERES = ['Alpha', 'Beta']

# Patrones para detección de metadatos
METADATOS_PATRONES = {
    'nombre': re.compile(r'#\s*Nombre\s*:\s*(.*)', re.IGNORECASE),
    'tarea': re.compile(r'#\s*Tarea\s*:\s*(.*)', re.IGNORECASE),
    'tipo': re.compile(r'#\s*Tipo\s*:\s*(.*)', re.IGNORECASE),
    'fecha': re.compile(r'#\s*Fecha\s*:\s*(.*)', re.IGNORECASE),
    'duracion': re.compile(r'#\s*Duraci[oóòn]n\s*:\s*(.*)', re.IGNORECASE)
}

# Clase para registro y reporte de problemas encontrados
class ReporteProblemas:
    def __init__(self):
        self.problemas = []
        self.estadisticas = {
            'total_archivos': 0,
            'archivos_raw': 0,
            'archivos_bandpower': 0,
            'archivos_con_metadatos': 0,
            'archivos_sin_metadatos': 0,
            'problemas_codificacion': 0,
            'nombres_columnas_inconsistentes': 0,
            'archivos_procesados': 0,
            'errores': 0
        }
    
    def agregar_problema(self, archivo, tipo, detalle):
        self.problemas.append({
            'archivo': archivo,
            'tipo': tipo,
            'detalle': detalle,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Actualizar contador de estadísticas
        if tipo in self.estadisticas:
            self.estadisticas[tipo] += 1
        else:
            self.estadisticas[tipo] = 1
    
    def generar_reporte(self, ruta_salida):
        """Genera un archivo JSON y un HTML con el reporte de problemas"""
        # Guardar reporte como JSON
        reporte = {
            'fecha_generacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'estadisticas': self.estadisticas,
            'problemas': self.problemas
        }
        
        with open(os.path.join(ruta_salida, 'reporte_problemas.json'), 'w', encoding='utf-8') as f:
            json.dump(reporte, f, ensure_ascii=False, indent=4)
        
        # Generar reporte HTML para mejor visualización
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Reporte de Inconsistencias en Datos EEG</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #333366; }
                .stats { display: flex; flex-wrap: wrap; margin-bottom: 20px; }
                .stat-box { 
                    background-color: #f0f0f0; padding: 15px; margin: 10px; 
                    border-radius: 5px; min-width: 200px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
                th { background-color: #333366; color: white; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .problem-raw { background-color: #ffe6e6; }
                .problem-bandpower { background-color: #e6ffe6; }
                .problem-encoding { background-color: #e6e6ff; }
                .problem-metadata { background-color: #fff5e6; }
                .problem-columns { background-color: #f2e6ff; }
                .footer { margin-top: 30px; font-size: 0.8em; color: #666; }
            </style>
        </head>
        <body>
            <h1>Reporte de Inconsistencias en Datos EEG Neurofly</h1>
            <p>Fecha de generación: """ + reporte['fecha_generacion'] + """</p>
            
            <h2>Estadísticas</h2>
            <div class="stats">
        """
        
        # Añadir estadísticas
        for key, value in self.estadisticas.items():
            html += f"""
                <div class="stat-box">
                    <h3>{key.replace('_', ' ').title()}</h3>
                    <p>{value}</p>
                </div>
            """
        
        html += """
            </div>
            
            <h2>Problemas Detectados</h2>
            <table>
                <tr>
                    <th>Archivo</th>
                    <th>Tipo de Problema</th>
                    <th>Detalle</th>
                    <th>Timestamp</th>
                </tr>
        """
        
        # Añadir filas de problemas
        for problema in self.problemas:
            tipo_class = f"problem-{problema['tipo'].lower().replace(' ', '-')}"
            html += f"""
                <tr class="{tipo_class}">
                    <td>{problema['archivo']}</td>
                    <td>{problema['tipo']}</td>
                    <td>{problema['detalle']}</td>
                    <td>{problema['timestamp']}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <div class="footer">
                <p>Generado automáticamente por el script de unificación y análisis de datos EEG Neurofly.</p>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(ruta_salida, 'reporte_problemas.html'), 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Reporte generado en: {ruta_salida}")

# Funciones para el procesamiento de bandpower
def nextpow2(n):
    """Calcula la siguiente potencia de 2"""
    n_log = np.log2(n)
    N_log = np.ceil(n_log)
    return int(N_log)

def myfft(x, fs):
    """Calcula la Transformada Rápida de Fourier para obtener el espectro de potencia"""
    m = len(x)
    n = 2 ** nextpow2(m)
    y = scipy.fft(x, n)
    f = np.arange(n) * fs / n
    power = y * np.conj(y) / n
    return f, np.real(power)

def bandpower(x, fs):
    """Calcula el espectro de potencia"""
    f, Pxx = myfft(x, fs)
    return f, Pxx

def SplitDataToFreqBand(f, power, fmin, fmax):
    """Divide los datos en bandas de frecuencia específicas"""
    lowidx = np.argmax(f >= fmin)
    highidx = np.argmax(f > fmax) - 1
    if highidx < lowidx:  # Manejar caso donde fmax está fuera del rango
        highidx = len(f) - 1
    band_power = np.mean(power[lowidx:highidx + 1])
    return band_power

def normalization(band):
    """Normaliza los valores de potencia de banda"""
    a = np.min(np.abs(band), axis=0)
    b = np.max(np.abs(band), axis=0)
    if b - a == 0:
        band = np.zeros_like(band)
    else:
        band = (band - a) / (b - a)
    return band.tolist()

def ComputeBandPowerAll(EEGdata, fs):
    """Calcula todas las bandas de potencia para todos los canales"""
    num_channels = len(EEGdata)
    
    alpha_power = np.empty(num_channels)
    beta_power = np.empty(num_channels)
    gamma_power = np.empty(num_channels)
    delta_power = np.empty(num_channels)
    theta_power = np.empty(num_channels)
    
    for i in range(num_channels):
        f, power = bandpower(EEGdata[i], fs)
        alpha_power[i] = SplitDataToFreqBand(f, power, ALPHA[0], ALPHA[1])
        beta_power[i] = SplitDataToFreqBand(f, power, BETA[0], BETA[1])
        gamma_power[i] = SplitDataToFreqBand(f, power, GAMMA[0], GAMMA[1])
        delta_power[i] = SplitDataToFreqBand(f, power, DELTA[0], DELTA[1])
        theta_power[i] = SplitDataToFreqBand(f, power, THETA[0], THETA[1])
    
    alpha_power = normalization(alpha_power)
    beta_power = normalization(beta_power)
    gamma_power = normalization(gamma_power)
    delta_power = normalization(delta_power)
    theta_power = normalization(theta_power)
    
    # Retorna un diccionario con las potencias normalizadas
    return {
        'Delta': delta_power,
        'Theta': theta_power,
        'Alpha': alpha_power,
        'Beta': beta_power,
        'Gamma': gamma_power
    }

def aplicar_bandpower_a_raw(datos_raw, fs=SAMPLING_FREQUENCY):
    """
    Aplica bandpower a datos raw (8 canales).
    
    Args:
        datos_raw (DataFrame): DataFrame con datos raw (timestamp + 8 canales)
        fs (int): Frecuencia de muestreo
        
    Returns:
        DataFrame: DataFrame con datos de bandpower calculados
    """
    # Extraer timestamp
    timestamp_col = datos_raw.columns[0]
    timestamp = datos_raw[timestamp_col].values
    
    # Extraer datos EEG (8 canales)
    eeg_data = datos_raw.iloc[:, 1:9].values.T  # Transpose para tener canales como filas
    
    # Verificar que tenemos datos suficientes
    if eeg_data.shape[1] < WINDOW_SIZE:
        raise ValueError(f"Se necesitan al menos {WINDOW_SIZE} muestras para calcular bandpower")
    
    # Preparar DataFrame para datos de bandpower
    bp_columns = ["Timestamp"]
    for i in range(CHANNELS):
        for banda in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
            bp_columns.append(f"Canal{i+1}_{banda}")
    
    bandpower_df = pd.DataFrame(columns=bp_columns)
    
    # Calcular bandpower para cada ventana de datos
    num_windows = len(timestamp) - WINDOW_SIZE + 1
    for i in range(0, num_windows, WINDOW_SIZE // 2):  # Avanzar con solapamiento del 50%
        window_data = eeg_data[:, i:i+WINDOW_SIZE]
        bp_result = ComputeBandPowerAll(window_data, fs)
        
        # Crear nueva fila para este cálculo de bandpower
        row = [timestamp[i+WINDOW_SIZE-1]]  # Usar último timestamp de la ventana
        
        # Añadir datos de bandpower
        for canal in range(CHANNELS):
            for banda in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
                row.append(bp_result[banda][canal])
        
        # Añadir al DataFrame
        bandpower_df.loc[len(bandpower_df)] = row
    
    return bandpower_df

def detectar_formato_archivo(ruta_archivo, reporte):
    """
    Detecta el formato y las características del archivo CSV.
    
    Args:
        ruta_archivo (str): Ruta al archivo CSV
        reporte (ReporteProblemas): Objeto para registrar problemas
        
    Returns:
        dict: Diccionario con información del formato del archivo
    """
    info = {
        'ruta': ruta_archivo,
        'nombre': os.path.basename(ruta_archivo),
        'sujeto': os.path.basename(os.path.dirname(ruta_archivo)),
        'tipo_datos': None,  # 'raw' o 'bandpower'
        'metadatos': {},
        'codificacion': None,
        'num_columnas': 0,
        'tiene_timestamp': False,
        'columnas': []
    }
    
    # Detectar codificación
    with open(ruta_archivo, 'rb') as f:
        result = chardet.detect(f.read(10000))
    
    info['codificacion'] = result['encoding']
    
    # Si la codificación no es UTF-8 o es desconocida, intentar con diferentes codificaciones
    codificaciones = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
    if result['encoding'] not in codificaciones:
        reporte.agregar_problema(info['nombre'], 'problemas_codificacion', 
                                f"Codificación detectada: {result['encoding']}")
    
    # Leer primeras líneas para detectar metadatos
    for encoding in codificaciones:
        try:
            with open(ruta_archivo, 'r', encoding=encoding) as f:
                primeras_lineas = [next(f) for _ in range(10) if f]
            break
        except (UnicodeDecodeError, StopIteration):
            continue
    
    # Buscar metadatos en las primeras líneas
    for linea in primeras_lineas:
        for clave, patron in METADATOS_PATRONES.items():
            match = patron.search(linea)
            if match:
                info['metadatos'][clave] = match.group(1).strip()
    
    if info['metadatos']:
        info['tiene_metadatos'] = True
        reporte.estadisticas['archivos_con_metadatos'] += 1
    else:
        info['tiene_metadatos'] = False
        reporte.estadisticas['archivos_sin_metadatos'] += 1
    
    # Identificar formato de datos (primero usando pandas para evitar problemas de parsing)
    try:
        # Intentar leer con pandas
        df = pd.read_csv(ruta_archivo, encoding=info['codificacion'])
        info['num_columnas'] = len(df.columns)
        info['columnas'] = df.columns.tolist()
        
        # Verificar si tiene timestamp (primera columna)
        if 'Timestamp' in df.columns or 'timestamp' in df.columns or df.columns[0].lower() == 'timestamp':
            info['tiene_timestamp'] = True
        
        # Determinar tipo de datos (raw o bandpower)
        if info['num_columnas'] >= 40:  # 8 canales x 5 bandas + timestamp
            info['tipo_datos'] = 'bandpower'
            reporte.estadisticas['archivos_bandpower'] += 1
        elif info['num_columnas'] >= 8:  # Al menos 8 canales
            info['tipo_datos'] = 'raw'
            reporte.estadisticas['archivos_raw'] += 1
        else:
            reporte.agregar_problema(info['nombre'], 'formato_desconocido', 
                                    f"Número de columnas insuficiente: {info['num_columnas']}")
        
        # Verificar consistencia de nombres de columnas
        if info['tipo_datos'] == 'bandpower':
            # Verificar patrones esperados para bandpower
            patron_banda = re.compile(r'Canal\d+_(Alpha|Beta|Gamma|Delta|Theta)', re.IGNORECASE)
            columnas_validas = sum(1 for col in info['columnas'] if patron_banda.match(col))
            
            if columnas_validas < 40:  # Esperamos 8 canales x 5 bandas
                reporte.agregar_problema(info['nombre'], 'nombres_columnas_inconsistentes',
                                        f"Columnas de bandas inconsistentes, solo {columnas_validas}/40 coinciden")
        
    except Exception as e:
        reporte.agregar_problema(info['nombre'], 'error_lectura', f"Error al leer archivo: {str(e)}")
    
    reporte.estadisticas['total_archivos'] += 1
    return info

def estandarizar_archivo(info_archivo, ruta_archivo, dir_backup, reporte):
    """
    Estandariza un archivo CSV, aplicando bandpower si es necesario y corrigiendo metadatos.
    
    Args:
        info_archivo (dict): Información del archivo detectada
        ruta_archivo (str): Ruta al archivo CSV
        dir_backup (str): Directorio para backup
        reporte (ReporteProblemas): Objeto para registrar problemas
        
    Returns:
        bool: True si el archivo fue procesado correctamente
    """
    try:
        # Crear backup del archivo original
        nombre_backup = os.path.join(dir_backup, f"{info_archivo['sujeto']}_{info_archivo['nombre']}")
        shutil.copy2(ruta_archivo, nombre_backup)
        
        # Si el archivo ya tiene formato bandpower, verificar consistencia
        if info_archivo['tipo_datos'] == 'bandpower':
            # Leer el archivo
            df = pd.read_csv(ruta_archivo, encoding=info_archivo['codificacion'])
            
            # Verificar que tiene todas las columnas necesarias
            necesita_correccion = False
            
            # Si tiene menos de 40 columnas de bandpower, registrar problema
            columnas_bandpower = sum(1 for col in df.columns if '_' in col and any(banda in col for banda in ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']))
            if columnas_bandpower < 40:
                reporte.agregar_problema(info_archivo['nombre'], 'columnas_bandpower_incompletas',
                                        f"Tiene {columnas_bandpower}/40 columnas de bandpower")
                necesita_correccion = True
            
            if not necesita_correccion:
                # Guardar con formato estándar
                guardar_archivo_estandarizado(df, info_archivo, ruta_archivo, reporte)
                return True
            
        # Si es raw o necesita corrección, aplicar bandpower
        if info_archivo['tipo_datos'] == 'raw':
            print(f"⚙️ Aplicando bandpower a archivo raw: {info_archivo['nombre']}")
            
            # Leer datos raw
            df = pd.read_csv(ruta_archivo, encoding=info_archivo['codificacion'])
            
            # Aplicar bandpower
            df_bandpower = aplicar_bandpower_a_raw(df)
            
            # Transferir o añadir metadatos
            if info_archivo['metadatos']:
                guardar_archivo_estandarizado(df_bandpower, info_archivo, ruta_archivo, reporte)
            else:
                # Si no tiene metadatos, extraer del nombre del archivo
                extraer_metadata_de_nombre(info_archivo)
                guardar_archivo_estandarizado(df_bandpower, info_archivo, ruta_archivo, reporte)
            
            return True
            
    except Exception as e:
        reporte.agregar_problema(info_archivo['nombre'], 'error_procesamiento', 
                                f"Error al estandarizar: {str(e)}")
        traceback.print_exc()
        return False

def extraer_metadata_de_nombre(info_archivo):
    """
    Intenta extraer metadatos del nombre del archivo si no están presentes.
    
    Args:
        info_archivo (dict): Información del archivo
    """
    # Formato esperado: Nombre_Tarea_Tipo.csv o similar
    nombre_sin_ext = os.path.splitext(info_archivo['nombre'])[0]
    partes = nombre_sin_ext.split('_')
    
    if len(partes) >= 3:
        if 'nombre' not in info_archivo['metadatos']:
            info_archivo['metadatos']['nombre'] = info_archivo['sujeto']
        
        if 'tarea' not in info_archivo['metadatos'] and len(partes) >= 1:
            info_archivo['metadatos']['tarea'] = partes[1]
        
        if 'tipo' not in info_archivo['metadatos'] and len(partes) >= 2:
            info_archivo['metadatos']['tipo'] = partes[2]
    else:
        # Si no podemos extraer del nombre, usar valores predeterminados
        if 'nombre' not in info_archivo['metadatos']:
            info_archivo['metadatos']['nombre'] = info_archivo['sujeto']
        
        if 'tarea' not in info_archivo['metadatos']:
            info_archivo['metadatos']['tarea'] = 'Desconocida'
        
        if 'tipo' not in info_archivo['metadatos']:
            info_archivo['metadatos']['tipo'] = 'Desconocido'
    
    # Fecha actual si no existe
    if 'fecha' not in info_archivo['metadatos']:
        info_archivo['metadatos']['fecha'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Duración predeterminada si no existe
    if 'duracion' not in info_archivo['metadatos']:
        info_archivo['metadatos']['duracion'] = '20'

def guardar_archivo_estandarizado(df, info_archivo, ruta_archivo, reporte):
    """
    Guarda un DataFrame como archivo CSV con formato estandarizado.
    
    Args:
        df (DataFrame): DataFrame a guardar
        info_archivo (dict): Información del archivo
        ruta_archivo (str): Ruta donde guardar el archivo
        reporte (ReporteProblemas): Objeto para registrar problemas
    """
    try:
        # Abrir archivo para escritura
        with open(ruta_archivo, 'w', encoding='utf-8', newline='') as f:
            # Escribir metadatos como comentarios
            for clave, valor in info_archivo['metadatos'].items():
                f.write(f"# {clave.capitalize()}: {valor}\n")
            
            # Escribir el DataFrame
            df.to_csv(f, index=False)
        
        reporte.estadisticas['archivos_procesados'] += 1
        print(f"✅ Archivo estandarizado: {info_archivo['nombre']}")
    except Exception as e:
        reporte.agregar_problema(info_archivo['nombre'], 'error_guardado', 
                                f"Error al guardar archivo estandarizado: {str(e)}")

def generar_grafico_general(df, info_archivo, output_path):
    """
    Genera un gráfico general con todas las bandas de frecuencia.
    
    Args:
        df (DataFrame): DataFrame con datos de bandpower
        info_archivo (dict): Información del archivo
        output_path (str): Ruta donde guardar el gráfico
    """
    # Detectar columnas por banda
    columnas_por_banda = {}
    for banda in ["Alpha", "Beta", "Gamma", "Delta", "Theta"]:
        patron = re.compile(f".*{banda}.*", re.IGNORECASE)
        columnas_por_banda[banda] = [col for col in df.columns if patron.match(col)]
    
    # Verificar que hay columnas para cada banda
    if not any(columnas_por_banda.values()):
        print(f"⚠️ No se encontraron columnas de bandas en {info_archivo['nombre']}")
        return
    
    # Calcular promedios por banda
    promedios_por_banda = {}
    for banda, columnas in columnas_por_banda.items():
        if columnas:
            promedios_por_banda[banda] = df[columnas].mean(axis=1).values
    
    # Configurar el gráfico
    plt.figure(figsize=(12, 6))
    
    # Eje X (tiempo o muestras)
    x = range(len(df))
    
    # Graficar cada banda
    for banda, valores in promedios_por_banda.items():
        plt.plot(x, valores, label=banda, color=COLOR_MAP.get(banda, 'black'), linewidth=2)
    
    # Extraer título del metadato o nombre de archivo
    if 'tarea' in info_archivo['metadatos'] and 'tipo' in info_archivo['metadatos']:
        titulo = f"{info_archivo['metadatos']['tarea']} - {info_archivo['metadatos']['tipo']}"
    else:
        titulo = os.path.splitext(info_archivo['nombre'])[0]
    
    # Configurar título y etiquetas
    plt.title(f"Bandas de frecuencia EEG - {titulo}", fontsize=16)
    plt.xlabel("Muestras", fontsize=12)
    plt.ylabel("Amplitud (normalizada)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generar_grafico_c3c4(df, info_archivo, output_path):
    """
    Genera un gráfico específico para C3 y C4 (canales 4 y 5) con Alpha y Beta.
    
    Args:
        df (DataFrame): DataFrame con datos de bandpower
        info_archivo (dict): Información del archivo
        output_path (str): Ruta donde guardar el gráfico
    """
    # Detectar columnas específicas para C3/C4 y Alpha/Beta
    columnas_especificas = {}
    for canal in CANALES_INTERES:
        columnas_especificas[canal] = {}
        for banda in BANDAS_INTERES:
            patron = re.compile(f".*Canal{canal}.*{banda}.*", re.IGNORECASE)
            columnas_especificas[canal][banda] = [col for col in df.columns if patron.match(col)]
    
    # Verificar que hay columnas
    hay_columnas = False
    for canal in CANALES_INTERES:
        for banda in BANDAS_INTERES:
            if columnas_especificas[canal][banda]:
                hay_columnas = True
                break
    
    if not hay_columnas:
        print(f"⚠️ No se encontraron columnas para C3/C4 Alpha/Beta en {info_archivo['nombre']}")
        return
    
    # Configurar el gráfico
    plt.figure(figsize=(12, 6))
    
    # Eje X (tiempo o muestras)
    x = range(len(df))
    
    # Graficar cada combinación canal/banda
    for canal in CANALES_INTERES:
        for banda in BANDAS_INTERES:
            columnas = columnas_especificas[canal][banda]
            if columnas:
                valores = df[columnas].mean(axis=1).values
                nombre_canal = NOMBRES_CANALES.get(canal, f"Canal{canal}")
                label = f"{nombre_canal} - {banda}"
                plt.plot(x, valores, label=label, linewidth=2.5)
    
    # Extraer título del metadato o nombre de archivo
    if 'tarea' in info_archivo['metadatos'] and 'tipo' in info_archivo['metadatos']:
        titulo = f"{info_archivo['metadatos']['tarea']} - {info_archivo['metadatos']['tipo']}"
    else:
        titulo = os.path.splitext(info_archivo['nombre'])[0]
    
    # Configurar título y etiquetas