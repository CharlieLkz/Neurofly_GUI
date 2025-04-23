#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversor de Bandpower CSV a Formato AURA CSV
============================================

Este script convierte archivos CSV con datos de bandpower al formato específico
de CSV que AURA puede leer directamente.

El formato de salida tendrá las siguientes columnas:
- Time and date
- F3, Fz, F4, C3, P3, C4, Pz, P4 (datos EEG)
- AccX, AccY, AccZ (acelerómetro)
- GyroX, GyroY, GyroZ (giroscopio)
- Battery, Event (estado)

Uso:
    python bandpower_to_aura_csv.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import time
from datetime import datetime

# Configuraciones
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "aura_data")  # Carpeta específica para archivos AURA

# Configuración de las bandas y canales
BANDAS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
CANALES = list(range(1, 9))  # Canales 1 al 8

# Mapeo de canales de bandpower a los canales de AURA
CANAL_MAP = {
    1: "F3",
    2: "Fz",
    3: "F4",
    4: "C3",
    5: "P3",
    6: "C4",
    7: "Pz", 
    8: "P4"
}

def list_participant_folders():
    """
    Lista todas las carpetas de participantes en el directorio de datos.
    
    Returns:
        Lista de rutas a carpetas de participantes
    """
    if not os.path.exists(DATA_DIR):
        print(f"Error: No se encontró el directorio de datos: {DATA_DIR}")
        return []
    
    # Buscar subcarpetas en el directorio de datos
    participant_folders = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) 
                         if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if not participant_folders:
        print(f"No se encontraron carpetas de participantes en {DATA_DIR}")
    
    return participant_folders

def find_csv_files(folder):
    """
    Encuentra todos los archivos CSV en una carpeta.
    
    Args:
        folder: Ruta a la carpeta donde buscar
        
    Returns:
        Lista de rutas a archivos CSV
    """
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    csv_files.sort()  # Ordenar archivos por nombre
    
    return csv_files

def count_comment_lines(file_path):
    """
    Cuenta las líneas de comentario al principio del archivo.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Número de líneas de comentario
    """
    try:
        # Probar diferentes codificaciones
        for encoding in ['utf-8', 'latin-1', 'ISO-8859-1', 'windows-1252', 'cp1252']:
            try:
                comment_lines = 0
                with open(file_path, 'r', encoding=encoding) as f:
                    for line in f:
                        if line.startswith('#'):
                            comment_lines += 1
                        else:
                            break
                return comment_lines, encoding
            except UnicodeDecodeError:
                continue
        
        # Si ninguna codificación funciona, usar latin-1 como respaldo
        comment_lines = 0
        with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
            for line in f:
                if line.startswith('#'):
                    comment_lines += 1
                else:
                    break
        return comment_lines, 'latin-1'
    except Exception as e:
        print(f"Error al contar líneas de comentario: {str(e)}")
        return 0, 'latin-1'

def read_bandpower_csv(file_path):
    """
    Lee un archivo CSV de bandpower y extrae los datos.
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos o None si hay error
    """
    try:
        print(f"  - Procesando: {os.path.basename(file_path)}")
        
        # Contar líneas de comentario y detectar codificación
        comment_lines, encoding = count_comment_lines(file_path)
        print(f"    Líneas de comentario: {comment_lines}, Codificación: {encoding}")
        
        # Intentar leer el archivo
        try:
            df = pd.read_csv(file_path, skiprows=comment_lines, encoding=encoding)
        except Exception as e:
            print(f"    Error al leer CSV con pandas: {str(e)}")
            # Intentar leer manualmente
            return read_bandpower_csv_manual(file_path, comment_lines, encoding)
        
        # Verificar que tenga columna Timestamp
        if 'Timestamp' not in df.columns:
            print(f"    No se encontró columna Timestamp")
            return None
        
        # Verificar que sea un archivo de bandpower
        bandpower_columns = []
        for canal in CANALES:
            for banda in BANDAS:
                col_name = f"Canal{canal}_{banda}"
                if col_name in df.columns:
                    bandpower_columns.append(col_name)
        
        if len(bandpower_columns) < 10:  # Al menos 10 columnas de bandpower
            print(f"    No parece ser un archivo de bandpower")
            return None
        
        print(f"    Archivo bandpower confirmado con {len(bandpower_columns)} columnas")
        
        # Mantener solo timestamp y columnas de bandpower
        selected_columns = ['Timestamp'] + bandpower_columns
        df = df[selected_columns]
        
        # Convertir datos a numéricos
        for col in df.columns:
            if col != 'Timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar filas con NaN
        df_cleaned = df.dropna()
        if len(df_cleaned) < len(df):
            print(f"    Se eliminaron {len(df) - len(df_cleaned)} filas con valores no numéricos")
        
        if df_cleaned.empty:
            print(f"    Todos los datos se convirtieron a NaN")
            return None
        
        print(f"    Datos leídos correctamente: {df_cleaned.shape}")
        return df_cleaned
    
    except Exception as e:
        print(f"    Error general al procesar {file_path}: {str(e)}")
        return None

def read_bandpower_csv_manual(file_path, comment_lines, encoding):
    """
    Lee un archivo CSV manualmente, línea por línea.
    
    Args:
        file_path: Ruta al archivo CSV
        comment_lines: Número de líneas de comentario a saltar
        encoding: Codificación del archivo
        
    Returns:
        DataFrame con los datos leídos
    """
    try:
        print(f"    Intentando lectura manual con codificación {encoding}")
        data = []
        headers = None
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            # Saltar líneas de comentario
            for _ in range(comment_lines):
                next(f, None)
            
            # Leer encabezados
            header_line = next(f, None)
            if header_line:
                headers = [h.strip() for h in header_line.split(',')]
            
            # Leer datos
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                values = [v.strip() for v in line.split(',')]
                if len(values) != len(headers):
                    print(f"    Advertencia: Línea con {len(values)} valores, se esperaban {len(headers)}")
                    # Intentar ajustar valores
                    if len(values) < len(headers):
                        values.extend([''] * (len(headers) - len(values)))
                    else:
                        values = values[:len(headers)]
                
                try:
                    # Convertir a números excepto Timestamp
                    processed_values = []
                    for i, v in enumerate(values):
                        if headers[i] == 'Timestamp':
                            processed_values.append(v)
                        else:
                            try:
                                processed_values.append(float(v) if v and v != 'nan' else np.nan)
                            except ValueError:
                                processed_values.append(np.nan)
                    data.append(processed_values)
                except Exception as e:
                    print(f"    Error procesando línea: {str(e)}")
        
        if not data or not headers:
            print(f"    No se encontraron datos o encabezados válidos")
            return None
        
        # Crear DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Verificar que tenga columna Timestamp
        if 'Timestamp' not in df.columns:
            print(f"    No se encontró columna Timestamp")
            return None
        
        # Verificar que sea un archivo de bandpower
        bandpower_columns = []
        for canal in CANALES:
            for banda in BANDAS:
                col_name = f"Canal{canal}_{banda}"
                if col_name in df.columns:
                    bandpower_columns.append(col_name)
        
        if len(bandpower_columns) < 10:  # Al menos 10 columnas de bandpower
            print(f"    No parece ser un archivo de bandpower")
            return None
        
        # Mantener solo timestamp y columnas de bandpower
        selected_columns = ['Timestamp'] + bandpower_columns
        df = df[selected_columns]
        
        # Eliminar filas con NaN
        df_cleaned = df.dropna()
        if len(df_cleaned) < len(df):
            print(f"    Se eliminaron {len(df) - len(df_cleaned)} filas con valores no numéricos")
        
        if df_cleaned.empty:
            print(f"    Todos los datos se convirtieron a NaN")
            return None
        
        print(f"    Datos leídos correctamente (manual): {df_cleaned.shape}")
        return df_cleaned
    
    except Exception as e:
        print(f"    Error en lectura manual: {str(e)}")
        return None

def combine_bandpower_data(csv_files):
    """
    Combina los datos de bandpower de múltiples archivos CSV.
    
    Args:
        csv_files: Lista de rutas a archivos CSV
        
    Returns:
        DataFrame combinado con datos de bandpower o None si no hay datos válidos
    """
    combined_data = None
    
    for file_path in csv_files:
        data = read_bandpower_csv(file_path)
        
        if data is None:
            continue
        
        if combined_data is None:
            combined_data = data
        else:
            # Concatenar datos
            combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    if combined_data is None:
        print("  - No se encontraron datos de bandpower válidos para combinar")
    else:
        print(f"  - Datos combinados: {combined_data.shape[0]} muestras, {combined_data.shape[1]} columnas")
    
    return combined_data

def convert_to_aura_format(df, task_info):
    """
    Convierte datos de bandpower al formato específico de AURA CSV.
    
    Args:
        df: DataFrame con datos de bandpower
        task_info: Información sobre la tarea/experimento
        
    Returns:
        DataFrame en formato AURA
    """
    # Crear DataFrame vacío con las columnas de AURA
    aura_columns = [
        'Time and date', 
        'F3', 'Fz', 'F4', 'C3', 'P3', 'C4', 'Pz', 'P4',  # Canales EEG
        'AccX', 'AccY', 'AccZ',  # Acelerómetro
        'GyroX', 'GyroY', 'GyroZ',  # Giroscopio
        'Battery', 'Event'  # Estado
    ]
    
    aura_data = []
    
    for idx, row in df.iterrows():
        aura_row = [None] * len(aura_columns)
        
        # Formatear timestamp como 'Time and date'
        timestamp = row['Timestamp']
        try:
            # Si es un valor numérico, convertir a datetime
            timestamp_float = float(timestamp)
            dt = datetime.fromtimestamp(timestamp_float)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        except (ValueError, TypeError):
            # Si ya es un string, usar como está
            time_str = str(timestamp)
        
        aura_row[0] = time_str
        
        # Llenar canales EEG usando la banda Alpha para todos
        for canal_idx, canal_name in CANAL_MAP.items():
            col_name = f"Canal{canal_idx}_Alpha"  # Usar Alpha como default
            if col_name in row:
                # Índice en aura_columns (restamos 1 porque Time and date es índice 0)
                aura_idx = list(aura_columns).index(canal_name)
                # Escalar valor para que se parezca a datos crudos
                aura_row[aura_idx] = str(int(row[col_name] * 10000))
        
        # Simular datos de acelerómetro y giroscopio (constantes o pequeñas variaciones)
        acc_idx = list(aura_columns).index('AccX')
        aura_row[acc_idx:acc_idx+3] = ['0', '0', '0']  # AccX, AccY, AccZ
        
        gyro_idx = list(aura_columns).index('GyroX')
        aura_row[gyro_idx:gyro_idx+3] = ['0', '0', '0']  # GyroX, GyroY, GyroZ
        
        # Batería al 100%
        battery_idx = list(aura_columns).index('Battery')
        aura_row[battery_idx] = '100'
        
        # Evento - usar información de la tarea
        event_idx = list(aura_columns).index('Event')
        if idx == 0:
            # Primer registro - iniciar tarea
            aura_row[event_idx] = f"START {task_info}"
        elif idx == len(df) - 1:
            # Último registro - finalizar tarea
            aura_row[event_idx] = f"END {task_info}"
        else:
            # Registros intermedios - vacío
            aura_row[event_idx] = ''
        
        aura_data.append(aura_row)
    
    # Crear DataFrame con formato AURA
    aura_df = pd.DataFrame(aura_data, columns=aura_columns)
    
    return aura_df

def save_aura_csv(df, participant_name, output_path):
    """
    Guarda datos en formato CSV de AURA.
    
    Args:
        df: DataFrame en formato AURA
        participant_name: Nombre del participante
        output_path: Ruta donde guardar el archivo
        
    Returns:
        True si se guardó correctamente, False en caso de error
    """
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar CSV sin índice
        df.to_csv(output_path, index=False)
        
        print(f"  - Archivo AURA CSV creado: {output_path}")
        return True
    except Exception as e:
        print(f"  - Error al guardar archivo AURA CSV: {str(e)}")
        return False

def process_participant_folder(folder):
    """
    Procesa una carpeta de participante.
    
    Args:
        folder: Ruta a la carpeta del participante
        
    Returns:
        True si se procesó correctamente, False en caso de error
    """
    participant_name = os.path.basename(folder)
    print(f"\nProcesando participante: {participant_name}")
    
    # Buscar archivos CSV
    csv_files = find_csv_files(folder)
    
    if not csv_files:
        print(f"  - No se encontraron archivos CSV en {folder}")
        return False
    
    print(f"  - Se encontraron {len(csv_files)} archivos CSV")
    
    # Combinar datos de bandpower
    combined_data = combine_bandpower_data(csv_files)
    
    if combined_data is None or combined_data.empty:
        return False
    
    # Determinar información de la tarea (usaremos el nombre del participante)
    task_info = f"EEG_{participant_name}"
    
    # Convertir a formato AURA
    aura_df = convert_to_aura_format(combined_data, task_info)
    
    # Generar nombre de archivo de salida con formato AURA
    current_time = datetime.now().strftime("%Y%m%d___%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"AURA_RAW___{current_time}.csv")
    
    # Guardar CSV en formato AURA
    result = save_aura_csv(aura_df, participant_name, output_file)
    
    return result

def main():
    """
    Función principal del script.
    """
    print("=="*40)
    print("CONVERSOR DE BANDPOWER CSV A FORMATO AURA CSV")
    print("=="*40)
    
    # Crear directorios si no existen
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Listar carpetas de participantes
    participant_folders = list_participant_folders()
    
    if not participant_folders:
        print(f"Por favor, coloca tus datos en subcarpetas dentro de: {DATA_DIR}")
        print("Ejemplo: data/Participante1/archivo.csv")
        return 1
    
    print(f"Se encontraron {len(participant_folders)} carpetas de participantes")
    
    # Procesar cada carpeta
    successful = 0
    for folder in participant_folders:
        if process_participant_folder(folder):
            successful += 1
    
    # Resumen
    print("\n" + "=="*40)
    print(f"Proceso completado: {successful}/{len(participant_folders)} participantes procesados")
    
    if successful > 0:
        print(f"\nLos archivos CSV en formato AURA están listos.")
        print(f"Puedes encontrarlos en: {OUTPUT_DIR}")
        print("\nFormato del nombre: AURA_RAW___YYYYMMDD___HHMMSS.csv")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())