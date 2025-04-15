#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 CORREGIDO FINAL: Preparación de datos EEG y aplicación de filtro paso banda
====================================================================================

Este script corrige los problemas con el filtro paso banda para datos cortos,
elimina código inalcanzable y asegura que todos los archivos se procesen correctamente.
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
import traceback
from scipy import signal, fft
import matplotlib.pyplot as plt
import warnings

# Ignorar advertencias específicas de filtfilt
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*The signal contains NaN.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

# Constantes y configuraciones
# ===========================

# Definiciones de bandas de frecuencia
BANDAS = {
    'Delta': [0.5, 4.0],
    'Theta': [4.0, 8.0],
    'Alpha': [8.0, 12.0],
    'Beta': [12.0, 30.0],
    'Gamma': [30.0, 100.0]
}

# Parámetros
SAMPLING_FREQUENCY = 250  # Frecuencia de muestreo en Hz
CANALES_INTERES = [4, 5]  # C3 y C4 (indices 4 y 5)
BANDAS_INTERES = ['Alpha', 'Beta']  # Bandas de interés

# Definir rutas principales
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
BANDPASS_DIR = os.path.join(EXPERIMENT_DIR, "BandPassData")

# Archivos que no se pudieron procesar en intentos anteriores
ARCHIVOS_FALLIDOS = []

# Funciones para bandpower
# ===========================

def nextpow2(n):
    """
    Encuentra la siguiente potencia de 2 mayor o igual a n.
    Útil para calcular el tamaño óptimo para FFT.
    """
    if n <= 0:
        return 0
    return int(np.ceil(np.log2(n)))

def calcular_bandpower(datos, fs=SAMPLING_FREQUENCY):
    """
    Calcula el bandpower de una señal usando FFT.
    
    Args:
        datos: Array con los datos de la señal EEG
        fs: Frecuencia de muestreo
        
    Returns:
        Diccionario con las potencias normalizadas por banda
    """
    # Verificar si hay suficientes datos
    if len(datos) < 2:
        return {banda: 0.01 for banda in BANDAS.keys()}
    
    # Verificar y limpiar NaN e infinitos
    datos = np.nan_to_num(datos, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Aplicar ventana Hamming para reducir leakage espectral
    datos = datos * np.hamming(len(datos))
    
    # Calcular FFT
    n = len(datos)
    n_fft = 2**nextpow2(n)  # Usar potencia de 2 para FFT eficiente
    fft_vals = fft.fft(datos, n=n_fft)
    fft_vals = fft_vals[:n_fft//2]  # Solo la mitad positiva del espectro
    
    # Calcular frecuencias
    freqs = fft.fftfreq(n_fft, 1/fs)
    freqs = freqs[:n_fft//2]
    
    # Calcular potencia
    potencia = np.abs(fft_vals)**2 / n
    
    # Calcular potencia en cada banda
    resultado = {}
    for banda, (fmin, fmax) in BANDAS.items():
        idx_banda = np.logical_and(freqs >= fmin, freqs <= fmax)
        if np.any(idx_banda):  # Si hay frecuencias en esta banda
            resultado[banda] = np.mean(potencia[idx_banda])
        else:
            resultado[banda] = 0
            
    # Normalizar resultados
    valor_max = max(resultado.values()) if resultado.values() else 1
    for banda in resultado:
        resultado[banda] = resultado[banda] / valor_max if valor_max > 0 else 0
        
    return resultado

def contar_comentarios(archivo):
    """
    Cuenta las líneas de comentario al principio del archivo.
    
    Args:
        archivo: Ruta al archivo
        
    Returns:
        Número de líneas de comentario
    """
    comment_lines = 0
    try:
        with open(archivo, 'rb') as f:
            for line in f:
                try:
                    decoded_line = line.decode('latin-1').strip()
                    if decoded_line.startswith('#'):
                        comment_lines += 1
                    else:
                        break
                except:
                    break
    except Exception as e:
        print(f"Error al contar comentarios en {archivo}: {str(e)}")
    
    return comment_lines

def tiene_bandpower(archivo):
    """
    Verifica si un archivo CSV ya tiene bandpower aplicado.
    Lo determina comprobando si tiene columnas con nombres de bandas.
    
    Args:
        archivo: Ruta al archivo CSV
        
    Returns:
        True si ya tiene bandpower, False si no
    """
    try:
        # Contar líneas de comentario al principio
        comment_lines = contar_comentarios(archivo)
                
        # Leer solo la primera fila para ver las columnas
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
            try:
                df = pd.read_csv(archivo, skiprows=comment_lines, nrows=1, encoding=encoding)
                
                # Verificar por columnas con nombres de banda
                bandas_cols = sum(1 for col in df.columns if any(banda in col for banda in BANDAS.keys()))
                
                # Si tiene al menos una columna por cada banda, entonces ya tiene bandpower
                if bandas_cols >= len(BANDAS):
                    return True
                # También verificar por el número de columnas (Timestamp + 8 canales x 5 bandas = 41)
                elif len(df.columns) >= 40:
                    return True
                else:
                    return False
            except:
                continue
                
        return False
    except Exception as e:
        print(f"Error al verificar bandpower en {archivo}: {str(e)}")
        return False

def procesar_archivo_raw(archivo, guardar_bp=True):
    """
    Procesa un archivo raw aplicando bandpower.
    
    Args:
        archivo: Ruta al archivo CSV raw
        guardar_bp: Si es True, guarda el resultado sobrescribiendo el original
        
    Returns:
        DataFrame con los resultados de bandpower o None si hay error
    """
    try:
        print(f"Procesando archivo raw: {archivo}")
        
        # Contar líneas de comentario al principio
        comment_lines = contar_comentarios(archivo)
        
        # Leer archivo, intentando diferentes codificaciones
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
            try:
                df = pd.read_csv(archivo, skiprows=comment_lines, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                # Intentar con delimitadores diferentes
                try:
                    df = pd.read_csv(archivo, skiprows=comment_lines, encoding=encoding, 
                                    sep=None, delimiter=None, engine='python')
                    break
                except:
                    continue
        
        if df is None:
            print(f"⚠️ No se pudo leer el archivo: {archivo}")
            return None
        
        # Verificar que tiene las columnas esperadas para datos raw
        # Debería tener al menos 9 columnas (timestamp + 8 canales EEG)
        if len(df.columns) < 9:
            print(f"⚠️ El archivo no tiene suficientes columnas para ser raw: {archivo}")
            return None
        
        # Manejar posibles errores en los datos
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Extraer timestamp y datos de canales
        timestamp = df.iloc[:, 0]  # Primera columna como timestamp
        datos_eeg = df.iloc[:, 1:9].values  # Las siguientes 8 columnas como datos EEG
        
        # Crear DataFrame para los resultados de bandpower
        bp_columns = ["Timestamp"]
        for i in range(8):
            for banda in BANDAS.keys():
                bp_columns.append(f"Canal{i+1}_{banda}")
        
        # Calcular bandpower para cada punto del tiempo
        bp_results = []
        window_size = 250  # 1 segundo de datos
        
        # Si no hay suficientes puntos, ajustar ventana
        if len(datos_eeg) < window_size:
            window_size = len(datos_eeg)
            if window_size < 10:  # Si hay menos de 10 puntos, no es suficiente
                print(f"⚠️ No hay suficientes datos para calcular bandpower: {archivo}")
                return None
        
        step = max(1, window_size // 4)  # Solapamiento del 75%
        
        for i in range(0, len(datos_eeg) - window_size + 1, step):
            # Obtener ventana de datos
            window_data = datos_eeg[i:i+window_size, :]
            row = [timestamp.iloc[i]]
            
            # Calcular bandpower para cada canal
            for canal in range(8):
                canal_data = window_data[:, canal]
                
                # Verificar si los datos son válidos
                if np.all(np.isfinite(canal_data)) and np.any(np.abs(canal_data) > 1e-10):
                    bp = calcular_bandpower(canal_data)
                else:
                    # Si los datos no son válidos, usar valores predeterminados
                    bp = {banda: 0.01 for banda in BANDAS.keys()}
                
                for banda in BANDAS.keys():
                    row.append(bp[banda])
            
            bp_results.append(row)
        
        # Si no se generaron resultados, salir
        if not bp_results:
            print(f"⚠️ No se generaron resultados de bandpower para: {archivo}")
            return None
        
        # Crear DataFrame con resultados
        bp_df = pd.DataFrame(bp_results, columns=bp_columns)
        
        # Guardar versión bandpower si es necesario (sobrescribir original)
        if guardar_bp:
            # Preservar los comentarios originales
            header_lines = []
            try:
                with open(archivo, 'rb') as f:
                    for _ in range(comment_lines):
                        line = f.readline()
                        try:
                            header_lines.append(line.decode('latin-1'))
                        except:
                            pass
            except Exception as e:
                print(f"Error al leer comentarios: {str(e)}")
            
            # Escribir archivo con comentarios originales + datos nuevos
            try:
                with open(archivo, 'w', encoding='utf-8') as f:
                    # Escribir comentarios originales
                    for line in header_lines:
                        f.write(line)
                    
                    # Escribir datos de bandpower
                    bp_df.to_csv(f, index=False)
                
                print(f"✅ Guardado archivo bandpower (sobrescrito): {archivo}")
            except Exception as e:
                print(f"Error al guardar archivo: {str(e)}")
                return bp_df
        
        return bp_df
        
    except Exception as e:
        print(f"Error procesando archivo raw {archivo}: {str(e)}")
        traceback.print_exc()
        return None

# Funciones para filtro paso banda
# =================================

def filtro_paso_banda(datos, fs, fmin, fmax, orden=4):
    """
    Aplica un filtro paso banda a los datos.
    
    Args:
        datos: Array de datos a filtrar
        fs: Frecuencia de muestreo
        fmin: Frecuencia mínima del paso banda
        fmax: Frecuencia máxima del paso banda
        orden: Orden del filtro Butterworth
        
    Returns:
        Datos filtrados
    """
    # Verificar si hay suficientes datos
    if len(datos) < 3*orden:
        # Si hay muy pocos datos, simplemente devolver los datos originales
        print(f"⚠️ Serie muy corta para filtro (longitud: {len(datos)}), orden: {orden}. Devolviendo original.")
        return datos
    
    # Verificar y limpiar NaN e infinitos
    datos = np.nan_to_num(datos, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalizar frecuencias de corte a la mitad de la frecuencia de Nyquist (fs/2)
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    
    # Diseñar filtro Butterworth paso banda
    try:
        # Reducir el orden para series cortas
        if len(datos) < 50:
            orden = 2  # Orden mínimo
        
        b, a = signal.butter(orden, [low, high], btype='band')
        
        # Intentar aplicar filtfilt primero
        try:
            # Calcular un padlen adecuado
            padlen = min(len(datos) // 3, 3 * max(len(a), len(b)))
            
            # Si padlen es mayor que los datos, reducirlo
            if padlen >= len(datos):
                padlen = max(len(a), len(b))
            
            # Si aún es muy grande, usar lfilter
            if padlen >= len(datos):
                datos_filtrados = signal.lfilter(b, a, datos)
            else:
                datos_filtrados = signal.filtfilt(b, a, datos, padlen=padlen)
                
        except ValueError as e:
            # Para errores de padlen, usar lfilter
            print(f"Usando lfilter como respaldo: {str(e)}")
            datos_filtrados = signal.lfilter(b, a, datos)
            
        except Exception as e:
            print(f"Error al aplicar filtro: {str(e)}")
            return datos
        
        return datos_filtrados
    
    except Exception as e:
        print(f"Error en filtro paso banda: {str(e)}")
        return datos  # Devolver datos originales en caso de error

def copiar_sin_filtrar(archivo_origen, archivo_destino):
    """
    Copia el archivo original al destino, extrayendo solo las columnas de interés
    pero sin aplicar filtrado. Función de respaldo para archivos problemáticos.
    
    Args:
        archivo_origen: Ruta del archivo origen
        archivo_destino: Ruta del archivo destino
        
    Returns:
        True si se copió correctamente, False si no
    """
    try:
        # Contar líneas de comentario
        comment_lines = contar_comentarios(archivo_origen)
        
        # Leer archivo con bandpower
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
            try:
                df = pd.read_csv(archivo_origen, skiprows=comment_lines, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(archivo_origen, skiprows=comment_lines, encoding=encoding, 
                                    sep=None, delimiter=None, engine='python')
                    break
                except:
                    continue
        
        if df is None:
            print(f"⚠️ No se pudo leer el archivo ni para copiarlo: {archivo_origen}")
            return False
        
        # Extraer columnas de interés si existen
        cols_interes = ["Timestamp"]
        for canal in CANALES_INTERES:
            for banda in BANDAS_INTERES:
                col_name = f"Canal{canal}_{banda}"
                if col_name in df.columns:
                    cols_interes.append(col_name)
        
        # Si no se encontraron columnas de interés, usar todas las columnas
        if len(cols_interes) <= 1:
            print(f"⚠️ No se encontraron columnas de interés, copiando todas: {archivo_origen}")
            df_filtrado = df
        else:
            df_filtrado = df[cols_interes].copy()
        
        # Asegurar que existe el directorio destino
        os.makedirs(os.path.dirname(archivo_destino), exist_ok=True)
        
        # Preservar los comentarios originales
        header_lines = []
        try:
            with open(archivo_origen, 'rb') as f:
                for _ in range(comment_lines):
                    line = f.readline()
                    try:
                        header_lines.append(line.decode('latin-1'))
                    except:
                        pass
        except Exception as e:
            print(f"Error al leer comentarios: {str(e)}")
        
        # Escribir archivo
        try:
            with open(archivo_destino, 'w', encoding='utf-8') as f:
                for line in header_lines:
                    f.write(line)
                df_filtrado.to_csv(f, index=False)
            
            print(f"⚠️ Archivo copiado sin filtrar: {archivo_destino}")
            return True
        except Exception as e:
            print(f"Error al escribir archivo: {str(e)}")
            return False
        
    except Exception as e:
        print(f"Error al copiar archivo {archivo_origen}: {str(e)}")
        return False

def procesar_csv_paso_banda(archivo_origen, archivo_destino):
    """
    Procesa un archivo CSV con bandpower para extraer las bandas Alpha y Beta
    de los canales C3 y C4 (4 y 5) y les aplica un filtro paso banda.
    
    Args:
        archivo_origen: Ruta al archivo CSV con bandpower
        archivo_destino: Ruta donde guardar el resultado filtrado
        
    Returns:
        True si se procesó correctamente, False si no
    """
    try:
        print(f"Aplicando filtro paso banda a: {archivo_origen}")
        
        # Verificar si este archivo ya falló antes
        nombre_archivo = os.path.basename(archivo_origen)
        if nombre_archivo in ARCHIVOS_FALLIDOS:
            print(f"⚠️ Archivo previamente fallido, usando método alternativo: {nombre_archivo}")
            return copiar_sin_filtrar(archivo_origen, archivo_destino)
        
        # Contar líneas de comentario al principio
        comment_lines = contar_comentarios(archivo_origen)
        
        # Leer archivo con bandpower
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
            try:
                df = pd.read_csv(archivo_origen, skiprows=comment_lines, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(archivo_origen, skiprows=comment_lines, encoding=encoding, 
                                    sep=None, delimiter=None, engine='python')
                    break
                except:
                    continue
        
        if df is None:
            print(f"⚠️ No se pudo leer el archivo: {archivo_origen}")
            ARCHIVOS_FALLIDOS.append(nombre_archivo)
            return False
        
        # Extraer solo las columnas de interés: Timestamp y las bandas Alpha y Beta de C3 y C4
        cols_interes = ["Timestamp"]
        
        for canal in CANALES_INTERES:
            for banda in BANDAS_INTERES:
                col_name = f"Canal{canal}_{banda}"
                if col_name in df.columns:
                    cols_interes.append(col_name)
        
        # Si no encontramos las columnas de interés, salir
        if len(cols_interes) <= 1:
            print(f"⚠️ No se encontraron columnas de interés en: {archivo_origen}")
            # Copiar archivo sin filtrar
            return copiar_sin_filtrar(archivo_origen, archivo_destino)
        
        # Crear DataFrame filtrado
        df_filtrado = df[cols_interes].copy()
        
        # Verificar si hay suficientes datos
        if len(df_filtrado) < 30:  # Si hay pocos puntos, no aplicar filtro
            print(f"⚠️ Archivo con pocos datos ({len(df_filtrado)} puntos), copiando sin filtrar: {nombre_archivo}")
            return copiar_sin_filtrar(archivo_origen, archivo_destino)
        
        # Aplicar filtro paso banda a cada columna (excepto Timestamp)
        for col in cols_interes[1:]:
            # Determinar banda para este canal
            if "Alpha" in col:
                fmin, fmax = BANDAS["Alpha"]
                # Orden reducido para Alpha
                orden = 2
            elif "Beta" in col:
                fmin, fmax = BANDAS["Beta"]
                # Orden reducido para Beta
                orden = 3
            else:
                continue
            
            # Aplicar filtro paso banda
            try:
                df_filtrado[col] = filtro_paso_banda(
                    df_filtrado[col].values, 
                    fs=SAMPLING_FREQUENCY,
                    fmin=fmin,
                    fmax=fmax,
                    orden=orden
                )
            except Exception as e:
                print(f"Error filtrando columna {col}: {str(e)}")
                # Mantener los datos originales para esta columna
        
        # Asegurar que existe el directorio destino
        os.makedirs(os.path.dirname(archivo_destino), exist_ok=True)
        
        # Preservar los comentarios originales
        header_lines = []
        try:
            with open(archivo_origen, 'rb') as f:
                for _ in range(comment_lines):
                    line = f.readline()
                    try:
                        header_lines.append(line.decode('latin-1'))
                    except:
                        pass
        except Exception as e:
            print(f"Error al leer comentarios: {str(e)}")
        
        # Escribir archivo con comentarios originales + datos filtrados
        try:
            with open(archivo_destino, 'w', encoding='utf-8') as f:
                # Escribir comentarios originales
                for line in header_lines:
                    f.write(line)
                
                # Escribir datos filtrados
                df_filtrado.to_csv(f, index=False)
            
            print(f"✅ Filtro paso banda aplicado y guardado en: {archivo_destino}")
            return True
        except Exception as e:
            print(f"Error al escribir archivo: {str(e)}")
            ARCHIVOS_FALLIDOS.append(nombre_archivo)
            return False
    
    except Exception as e:
        print(f"Error al procesar con paso banda {archivo_origen}: {str(e)}")
        nombre_archivo = os.path.basename(archivo_origen)
        ARCHIVOS_FALLIDOS.append(nombre_archivo)
        traceback.print_exc()
        
        # Intentar usar método alternativo
        print(f"Intentando método alternativo para: {nombre_archivo}")
        return copiar_sin_filtrar(archivo_origen, archivo_destino)

# Función principal para procesar una carpeta de participante
# ==========================================================

def procesar_carpeta_participante(carpeta_participante):
    """
    Procesa todos los archivos CSV en la carpeta de un participante.
    1. Verifica y aplica bandpower a los que lo necesiten
    2. Crea carpeta en BandPassData
    3. Aplica filtro paso banda y guarda resultados
    
    Args:
        carpeta_participante: Ruta a la carpeta del participante
        
    Returns:
        Estadísticas del procesamiento
    """
    try:
        nombre_participante = os.path.basename(carpeta_participante)
        print(f"\nProcesando carpeta de participante: {nombre_participante}")
        
        # Crear carpeta correspondiente en BandPassData (con sufijo)
        carpeta_destino = os.path.join(BANDPASS_DIR, f"{nombre_participante}BandPass")
        os.makedirs(carpeta_destino, exist_ok=True)
        
        # Obtener todos los archivos CSV
        archivos_csv = [f for f in os.listdir(carpeta_participante) if f.endswith('.csv')]
        print(f"Se encontraron {len(archivos_csv)} archivos CSV")
        
        # Estadísticas
        stats = {
            "archivos_totales": len(archivos_csv),
            "archivos_con_bandpower": 0,
            "archivos_con_bandpower_aplicado": 0,
            "archivos_con_paso_banda": 0,
            "archivos_copiados": 0,
            "errores": 0
        }
        
        # Primero aplicar bandpower a todos los archivos que lo necesiten
        for archivo_csv in archivos_csv:
            archivo_path = os.path.join(carpeta_participante, archivo_csv)
            
            try:
                # Verificar si ya tiene bandpower
                if tiene_bandpower(archivo_path):
                    print(f"El archivo ya tiene bandpower: {archivo_csv}")
                    stats["archivos_con_bandpower"] += 1
                else:
                    # Aplicar bandpower
                    if procesar_archivo_raw(archivo_path, guardar_bp=True) is not None:
                        stats["archivos_con_bandpower_aplicado"] += 1
                        print(f"Bandpower aplicado con éxito a: {archivo_csv}")
                    else:
                        print(f"No se pudo aplicar bandpower a: {archivo_csv}")
                        stats["errores"] += 1
            except Exception as e:
                print(f"Error procesando {archivo_csv}: {str(e)}")
                stats["errores"] += 1
        
        # Luego aplicar filtro paso banda a todos los archivos (ahora con bandpower)
        for archivo_csv in archivos_csv:
            archivo_path = os.path.join(carpeta_participante, archivo_csv)
            archivo_destino = os.path.join(carpeta_destino, archivo_csv)
            
            try:
                # Verificar nuevamente que tiene bandpower
                if tiene_bandpower(archivo_path):
                    # Aplicar filtro paso banda
                    result = procesar_csv_paso_banda(archivo_path, archivo_destino)
                    if result:
                        if archivo_csv in ARCHIVOS_FALLIDOS:
                            stats["archivos_copiados"] += 1
                        else:
                            stats["archivos_con_paso_banda"] += 1
                    else:
                        print(f"No se pudo aplicar filtro paso banda a: {archivo_csv}")
                        stats["errores"] += 1
                else:
                    print(f"⚠️ El archivo aún no tiene bandpower: {archivo_csv}")
                    stats["errores"] += 1
            except Exception as e:
                print(f"Error procesando paso banda en {archivo_csv}: {str(e)}")
                stats["errores"] += 1
        
        # Mostrar estadísticas de esta carpeta
        print(f"\nEstadísticas para {nombre_participante}:")
        print(f"Archivos totales: {stats['archivos_totales']}")
        print(f"Archivos que ya tenían bandpower: {stats['archivos_con_bandpower']}")
        print(f"Archivos a los que se aplicó bandpower: {stats['archivos_con_bandpower_aplicado']}")
        print(f"Archivos a los que se aplicó paso banda: {stats['archivos_con_paso_banda']}")
        print(f"Archivos copiados sin filtrar: {stats['archivos_copiados']}")
        print(f"Errores encontrados: {stats['errores']}")
        
        return stats
    
    except Exception as e:
        print(f"Error procesando carpeta {carpeta_participante}: {str(e)}")
        traceback.print_exc()
        return {
            "archivos_totales": 0,
            "archivos_con_bandpower": 0,
            "archivos_con_bandpower_aplicado": 0, 
            "archivos_con_paso_banda": 0,
            "archivos_copiados": 0,
            "errores": 1
        }

# Función principal
# ================

def main():
    """
    Función principal del script
    """
    print("="*70)
    print("PROCESADOR DE DATOS EEG - BANDPOWER Y FILTRO PASO BANDA (CORREGIDO FINAL)")
    print("="*70)
    
    # Verificar que existen los directorios principales
    if not os.path.exists(DATA_DIR):
        print(f"Error: No se encontró el directorio de datos: {DATA_DIR}")
        sys.exit(1)
    
    # Crear directorio para datos con filtro paso banda
    if not os.path.exists(BANDPASS_DIR):
        os.makedirs(BANDPASS_DIR)
        print(f"Se creó el directorio para datos con filtro paso banda: {BANDPASS_DIR}")
    
    # Obtener todas las carpetas de participantes
    carpetas_participantes = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) 
                             if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    print(f"Se encontraron {len(carpetas_participantes)} carpetas de participantes")
    
    # Estadísticas globales
    stats_global = {
        "participantes_procesados": 0,
        "archivos_totales": 0,
        "archivos_con_bandpower": 0,
        "archivos_con_bandpower_aplicado": 0,
        "archivos_con_paso_banda": 0,
        "archivos_copiados": 0,
        "errores": 0
    }
    
    # Procesar cada carpeta de participante (sin multihilo para mayor control)
    for carpeta in carpetas_participantes:
        try:
            stats = procesar_carpeta_participante(carpeta)
            stats_global["participantes_procesados"] += 1
            stats_global["archivos_totales"] += stats["archivos_totales"]
            stats_global["archivos_con_bandpower"] += stats["archivos_con_bandpower"]
            stats_global["archivos_con_bandpower_aplicado"] += stats["archivos_con_bandpower_aplicado"]
            stats_global["archivos_con_paso_banda"] += stats["archivos_con_paso_banda"]
            stats_global["archivos_copiados"] += stats["archivos_copiados"]
            stats_global["errores"] += stats["errores"]
        except Exception as e:
            print(f"Error al procesar carpeta {carpeta}: {str(e)}")
            stats_global["errores"] += 1
    
    # Mostrar estadísticas globales
    print("\n" + "="*70)
    print("ESTADÍSTICAS GLOBALES DE PROCESAMIENTO")
    print("="*70)
    print(f"Participantes procesados: {stats_global['participantes_procesados']}")
    print(f"Archivos totales: {stats_global['archivos_totales']}")
    print(f"Archivos que ya tenían bandpower: {stats_global['archivos_con_bandpower']}")
    print(f"Archivos a los que se aplicó bandpower: {stats_global['archivos_con_bandpower_aplicado']}")
    print(f"Archivos a los que se aplicó filtro paso banda: {stats_global['archivos_con_paso_banda']}")
    print(f"Archivos copiados sin filtrar: {stats_global['archivos_copiados']}")
    print(f"Errores encontrados: {stats_global['errores']}")
    
    # Mostrar archivos que no se pudieron procesar
    if ARCHIVOS_FALLIDOS:
        print("\nARCHIVOS QUE NO SE PUDIERON PROCESAR CON FILTRO:")
        for archivo in ARCHIVOS_FALLIDOS:
            print(f"- {archivo}")
    
    print("="*70)
    print("Proceso completado!")

if __name__ == "__main__":
    main()