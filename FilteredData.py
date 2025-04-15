#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2 CORREGIDO FINAL: Aplicación de filtro de Kalman a datos EEG
====================================================================

Este script aplica un filtro de Kalman a los datos EEG ya procesados con filtro paso banda
en la carpeta BandPassData, y guarda los resultados en la carpeta FilteredData.

Estructura esperada:
- experiment_gui/
  ├── BandPassData/      # Carpeta de entrada con datos filtrados por banda
  │   ├── AdanBandPass/
  │   │   ├── archivo1.csv
  │   │   ├── archivo2.csv
  └── FilteredData/      # Carpeta de salida con filtro Kalman aplicado
      ├── AdanFiltered/  # Carpetas con sufijo "Filtered" en lugar de "BandPass"
      │   ├── archivo1.csv
      │   ├── archivo2.csv
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
import traceback
import warnings
from pykalman import KalmanFilter

# Ignorar advertencias específicas
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Definir rutas principales
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDPASS_DIR = os.path.join(EXPERIMENT_DIR, "BandPassData")
FILTERED_DIR = os.path.join(EXPERIMENT_DIR, "FilteredData")

# Lista para almacenar archivos que no se pudieron procesar
ARCHIVOS_FALLIDOS = []

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

def aplicar_kalman(serie, initial_state_mean=0, observation_covariance=1.0, 
                  transition_covariance=0.01, transition_matrices=1.0):
    """
    Aplica un filtro de Kalman a una serie temporal usando pykalman.
    
    Args:
        serie: Serie temporal a filtrar
        initial_state_mean: Media inicial del estado (default 0)
        observation_covariance: Covarianza del ruido de observación
        transition_covariance: Covarianza del ruido de proceso
        transition_matrices: Matrices de transición (default 1.0)
        
    Returns:
        Serie temporal filtrada
    """
    # Manejar valores nulos o infinitos
    serie = np.array(serie, dtype=float)
    serie = np.nan_to_num(serie, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Si la serie está vacía o tiene un solo valor, devolverla como está
    if len(serie) <= 1:
        return serie
    
    try:
        # Configurar el filtro de Kalman según el modelo descrito
        kf = KalmanFilter(
            initial_state_mean=initial_state_mean,  # Estado inicial
            n_dim_obs=1,                           # Dimensión de observación (1D)
            observation_covariance=observation_covariance,  # Var. ruido medición
            transition_covariance=transition_covariance,    # Var. ruido proceso
            transition_matrices=transition_matrices         # Matriz transición
        )
        
        # Aplicar el filtro (serie debe estar en formato columna)
        estado_filtrado, _ = kf.filter(serie.reshape(-1, 1))
        
        # Devolver la serie filtrada (aplanar la matriz)
        return estado_filtrado.flatten()
    
    except Exception as e:
        print(f"Error al aplicar filtro Kalman: {str(e)}")
        traceback.print_exc()
        # Si falla, devolver la serie original
        return serie

def procesar_csv_con_kalman(archivo_origen, archivo_destino):
    """
    Procesa un archivo CSV aplicando filtro de Kalman a todas las columnas de datos.
    
    Args:
        archivo_origen: Ruta al archivo CSV con datos procesados por filtro paso banda
        archivo_destino: Ruta donde guardar el resultado con Kalman aplicado
        
    Returns:
        True si se procesó correctamente, False si no
    """
    try:
        print(f"Aplicando filtro Kalman a: {archivo_origen}")
        
        # Verificar si este archivo ya falló antes
        nombre_archivo = os.path.basename(archivo_origen)
        if nombre_archivo in ARCHIVOS_FALLIDOS:
            print(f"⚠️ Archivo previamente fallido, usando método alternativo: {nombre_archivo}")
            return copiar_sin_filtrar(archivo_origen, archivo_destino)
        
        # Contar líneas de comentario al principio
        comment_lines = contar_comentarios(archivo_origen)
        
        # Leer archivo
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
        
        # Si el archivo está vacío, simplemente copiarlo
        if len(df) <= 1:
            print(f"⚠️ El archivo tiene muy pocos datos ({len(df)} filas), copiando sin filtrar: {nombre_archivo}")
            return copiar_sin_filtrar(archivo_origen, archivo_destino)
        
        # Hacer copia para aplicar Kalman
        df_kalman = df.copy()
        
        # Guardar timestamp original (primera columna)
        timestamp_col = df.iloc[:, 0].copy() if len(df.columns) > 0 else None
        
        # Aplicar filtro de Kalman a cada columna (excepto Timestamp)
        for col in df.columns[1:]:
            # Ajustar parámetros del filtro según el tipo de señal
            if "Alpha" in col:
                # Para Alpha, menor varianza del proceso para suavizado más intenso
                transition_covariance = 0.005
                observation_covariance = 0.1
            elif "Beta" in col:
                # Para Beta, mayor varianza del proceso para seguir cambios más rápidos
                transition_covariance = 0.01
                observation_covariance = 0.15
            else:
                # Valores por defecto para otras columnas
                transition_covariance = 0.01
                observation_covariance = 0.1
            
            # Aplicar Kalman
            try:
                df_kalman[col] = aplicar_kalman(
                    df[col].values, 
                    initial_state_mean=df[col].values[0] if len(df[col].values) > 0 else 0,
                    observation_covariance=observation_covariance,
                    transition_covariance=transition_covariance,
                    transition_matrices=1.0
                )
            except Exception as e:
                print(f"Error al aplicar Kalman a columna {col}: {str(e)}")
                # Mantener valor original si hay error
                df_kalman[col] = df[col].values
        
        # Asegurar que el timestamp se mantiene sin cambios
        if timestamp_col is not None:
            df_kalman.iloc[:, 0] = timestamp_col
        
        # Asegurar que existe el directorio destino
        os.makedirs(os.path.dirname(archivo_destino), exist_ok=True)
        
        # Preservar los comentarios originales
        with open(archivo_origen, 'rb') as f:
            header_lines = []
            for _ in range(comment_lines):
                line = f.readline()
                try:
                    header_lines.append(line.decode('latin-1'))
                except:
                    pass
        
        # Escribir archivo con comentarios originales + datos filtrados
        with open(archivo_destino, 'w', encoding='utf-8') as f:
            # Escribir comentarios originales
            for line in header_lines:
                f.write(line)
            
            # Escribir datos filtrados
            df_kalman.to_csv(f, index=False)
        
        print(f"✅ Filtro Kalman aplicado y guardado en: {archivo_destino}")
        return True
    
    except Exception as e:
        print(f"Error al procesar con Kalman {archivo_origen}: {str(e)}")
        nombre_archivo = os.path.basename(archivo_origen)
        ARCHIVOS_FALLIDOS.append(nombre_archivo)
        traceback.print_exc()
        
        # Intentar método alternativo
        print(f"Intentando método alternativo para: {nombre_archivo}")
        return copiar_sin_filtrar(archivo_origen, archivo_destino)

def copiar_sin_filtrar(archivo_origen, archivo_destino):
    """
    Copia el archivo original al destino sin aplicar filtrado.
    Función de respaldo para archivos problemáticos.
    """
    try:
        # Asegurar que existe el directorio destino
        os.makedirs(os.path.dirname(archivo_destino), exist_ok=True)
        
        # Simplemente copiar el archivo
        shutil.copy2(archivo_origen, archivo_destino)
        print(f"⚠️ Archivo copiado sin filtrar: {archivo_destino}")
        return True
    except Exception as e:
        print(f"Error al copiar archivo {archivo_origen}: {str(e)}")
        return False

def procesar_carpeta_participante(carpeta_participante):
    """
    Procesa todos los archivos CSV en la carpeta de un participante,
    aplicando el filtro de Kalman y guardando los resultados en FilteredData.
    
    Args:
        carpeta_participante: Ruta a la carpeta del participante en BandPassData
        
    Returns:
        Estadísticas del procesamiento
    """
    try:
        # Obtener nombre base del participante (quitar "BandPass")
        nombre_base = os.path.basename(carpeta_participante)
        if nombre_base.endswith("BandPass"):
            nombre_participante = nombre_base[:-8]  # Eliminar "BandPass"
        else:
            nombre_participante = nombre_base
            
        print(f"\nProcesando carpeta de participante: {nombre_base}")
        
        # Crear carpeta correspondiente en FilteredData con sufijo "Filtered"
        carpeta_destino = os.path.join(FILTERED_DIR, f"{nombre_participante}Filtered")
        os.makedirs(carpeta_destino, exist_ok=True)
        
        # Obtener todos los archivos CSV
        archivos_csv = [f for f in os.listdir(carpeta_participante) if f.endswith('.csv')]
        print(f"Se encontraron {len(archivos_csv)} archivos CSV")
        
        # Estadísticas
        stats = {
            "archivos_totales": len(archivos_csv),
            "archivos_procesados": 0,
            "archivos_copiados": 0,
            "errores": 0
        }
        
        # Procesar cada archivo CSV
        for archivo_csv in archivos_csv:
            archivo_path = os.path.join(carpeta_participante, archivo_csv)
            archivo_destino = os.path.join(carpeta_destino, archivo_csv)
            
            try:
                # Aplicar filtro Kalman
                result = procesar_csv_con_kalman(archivo_path, archivo_destino)
                if result:
                    if archivo_csv in ARCHIVOS_FALLIDOS:
                        stats["archivos_copiados"] += 1
                    else:
                        stats["archivos_procesados"] += 1
                else:
                    print(f"⚠️ No se pudo aplicar filtro Kalman ni copiar: {archivo_csv}")
                    stats["errores"] += 1
            
            except Exception as e:
                print(f"Error procesando Kalman en {archivo_csv}: {str(e)}")
                traceback.print_exc()
                stats["errores"] += 1
        
        # Mostrar estadísticas de esta carpeta
        print(f"\nEstadísticas para {nombre_base}:")
        print(f"Archivos totales: {stats['archivos_totales']}")
        print(f"Archivos procesados con Kalman: {stats['archivos_procesados']}")
        print(f"Archivos copiados sin filtrar: {stats['archivos_copiados']}")
        print(f"Errores encontrados: {stats['errores']}")
        
        return stats
    
    except Exception as e:
        print(f"Error procesando carpeta {carpeta_participante}: {str(e)}")
        traceback.print_exc()
        return {
            "archivos_totales": 0,
            "archivos_procesados": 0,
            "archivos_copiados": 0,
            "errores": 1
        }

def main():
    """
    Función principal del script
    """
    print("="*70)
    print("PROCESADOR DE DATOS EEG - FILTRO DE KALMAN (CORREGIDO FINAL)")
    print("="*70)
    
    # Verificar que existen los directorios principales
    if not os.path.exists(BANDPASS_DIR):
        print(f"Error: No se encontró el directorio de datos con filtro paso banda: {BANDPASS_DIR}")
        print("Por favor, ejecute primero el script de preparación de datos y filtro paso banda.")
        sys.exit(1)
    
    # Crear directorio para datos con filtro Kalman
    if not os.path.exists(FILTERED_DIR):
        os.makedirs(FILTERED_DIR)
        print(f"Se creó el directorio para datos con filtro Kalman: {FILTERED_DIR}")
    
    # Obtener todas las carpetas de participantes de BandPassData
    carpetas_participantes = [os.path.join(BANDPASS_DIR, d) for d in os.listdir(BANDPASS_DIR) 
                             if os.path.isdir(os.path.join(BANDPASS_DIR, d))]
    
    print(f"Se encontraron {len(carpetas_participantes)} carpetas de participantes en BandPassData")
    
    # Estadísticas globales
    stats_global = {
        "participantes_procesados": 0,
        "archivos_totales": 0,
        "archivos_procesados": 0,
        "archivos_copiados": 0,
        "errores": 0
    }
    
    # Procesar cada carpeta de participante (sin multihilo para mayor control)
    for carpeta in carpetas_participantes:
        try:
            stats = procesar_carpeta_participante(carpeta)
            stats_global["participantes_procesados"] += 1
            stats_global["archivos_totales"] += stats["archivos_totales"]
            stats_global["archivos_procesados"] += stats["archivos_procesados"]
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
    print(f"Archivos procesados con Kalman: {stats_global['archivos_procesados']}")
    print(f"Archivos copiados sin filtrar: {stats_global['archivos_copiados']}")
    print(f"Errores encontrados: {stats_global['errores']}")
    
    # Mostrar archivos que fallaron
    if ARCHIVOS_FALLIDOS:
        print("\nARCHIVOS QUE NO SE PUDIERON PROCESAR CON KALMAN:")
        for archivo in ARCHIVOS_FALLIDOS:
            print(f"- {archivo}")
    
    print("="*70)
    print("Proceso completado!")

if __name__ == "__main__":
    main()