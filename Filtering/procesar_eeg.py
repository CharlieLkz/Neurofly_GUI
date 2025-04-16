#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Mejorado para Procesar Datos EEG y Generar Gráficas
"""

import os
import pandas as pd
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
import traceback
import shutil

# Definiciones de bandas de frecuencia
BANDAS = {
    'Delta': [0.5, 4.0],
    'Theta': [4.0, 8.0],
    'Alpha': [8.0, 12.0],
    'Beta': [12.0, 30.0],
    'Gamma': [30.0, 100.0]
}

# Parámetros
SAMPLING_FREQUENCY = 250
CANALES_INTERES = [4, 5]  # C3 y C4
BANDAS_INTERES = ['Alpha', 'Beta']

# Colores para gráficas
COLORES = {
    'Delta': 'purple',
    'Theta': 'orange',
    'Alpha': 'blue',
    'Beta': 'red',
    'Gamma': 'green'
}

def calcular_bandpower(datos, fs=SAMPLING_FREQUENCY):
    """
    Calcula el bandpower de una señal usando FFT
    """
    # Aplicar ventana Hamming
    datos = datos * np.hamming(len(datos))
    
    # Calcular FFT
    n = len(datos)
    fft_vals = fft.fft(datos)
    fft_vals = fft_vals[:n//2]  # Solo la mitad positiva del espectro
    
    # Calcular frecuencias
    freqs = fft.fftfreq(n, 1/fs)
    freqs = freqs[:n//2]
    
    # Calcular potencia
    potencia = np.abs(fft_vals)**2 / n
    
    # Calcular potencia en cada banda
    resultado = {}
    for banda, (fmin, fmax) in BANDAS.items():
        idx_banda = np.logical_and(freqs >= fmin, freqs <= fmax)
        if any(idx_banda):  # Si hay frecuencias en esta banda
            resultado[banda] = np.mean(potencia[idx_banda])
        else:
            resultado[banda] = 0
            
    # Normalizar resultados
    valor_max = max(resultado.values()) if resultado.values() else 1
    for banda in resultado:
        resultado[banda] = resultado[banda] / valor_max if valor_max > 0 else 0
        
    return resultado

def detectar_tipo_archivo(archivo):
    """
    Detecta si un archivo es raw o bandpower basado en sus columnas y formato
    """
    try:
        # Primero leemos las primeras líneas para ver si hay comentarios al inicio
        with open(archivo, 'rb') as f:
            # Leer los primeros bytes para determinar la codificación
            raw_data = f.read(4096)
            
        # Intentar diferentes codificaciones
        content = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
            try:
                content = raw_data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"⚠️ No se pudo determinar la codificación para {archivo}")
            return "error"
            
        # Dividir por líneas
        lines = content.splitlines()
        
        # Contar líneas de comentarios (las que empiezan con #)
        comment_lines = 0
        for line in lines:
            if line.strip().startswith('#'):
                comment_lines += 1
            else:
                break
        
        # Leer el archivo con pandas, saltando las líneas de comentarios
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
            try:
                df = pd.read_csv(archivo, skiprows=comment_lines, nrows=1, encoding=encoding)
                
                # Verificar si es un archivo bandpower contando columnas con nombres de bandas
                bandas_cols = sum(1 for col in df.columns if any(banda in col for banda in BANDAS.keys()))
                
                if bandas_cols >= len(BANDAS):  # Al menos debe tener una columna por banda
                    return "bandpower"
                elif "Canal1" in df.columns or len(df.columns) == 9:  # Timestamp + 8 canales
                    return "raw"
                else:
                    # Intentar inferir por el número de columnas
                    # Si tiene 41 columnas (Timestamp + 8 canales * 5 bandas), probablemente es bandpower
                    if len(df.columns) == 41:
                        return "bandpower"
                    # Si tiene 9 columnas (Timestamp + 8 canales), probablemente es raw
                    elif len(df.columns) == 9:
                        return "raw"
                    
                    return "desconocido"
            except Exception as e:
                if encoding == 'ISO-8859-1':
                    pass  # No imprimimos errores en esta etapa
        
        # Si llegamos aquí, intentamos leer las primeras 10 líneas para analizar mejor
        try:
            # Abrir el archivo y leer las primeras líneas después de los comentarios
            with open(archivo, 'rb') as f:
                lines = f.read().decode('latin-1').splitlines()
            
            # Saltar líneas de comentarios
            data_lines = lines[comment_lines:]
            
            # Verificar la primera línea de datos (debe ser la cabecera)
            if len(data_lines) > 0:
                header = data_lines[0].split(',')
                
                # Si la cabecera tiene cerca de 41 columnas, probablemente es bandpower
                if len(header) > 35:
                    return "bandpower"
                # Si la cabecera tiene cerca de 9 columnas, probablemente es raw
                elif len(header) >= 8 and len(header) <= 10:
                    return "raw"
        except Exception:
            pass
            
        return "desconocido"
    except Exception as e:
        print(f"Error al detectar tipo de archivo {archivo}: {str(e)}")
        return "error"

def procesar_archivo_raw(archivo, guardar_bp=False):
    """
    Procesa un archivo raw aplicando bandpower
    """
    try:
        print(f"Procesando archivo raw: {archivo}")
        
        # Primero contamos las líneas de comentario
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
            print(f"Error al contar líneas de comentario: {str(e)}")
        
        # Leer archivo, intentando diferentes codificaciones y saltando líneas de comentario
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
            try:
                df = pd.read_csv(archivo, skiprows=comment_lines, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                # Intentar con delimitadores diferentes si hay error de parseo
                try:
                    df = pd.read_csv(archivo, skiprows=comment_lines, encoding=encoding, 
                                    sep=None, delimiter=None, 
                                    engine='python')
                    break
                except:
                    continue
        
        if df is None:
            print(f"⚠️ No se pudo leer el archivo: {archivo}")
            return None
        
        # Verificar que tiene las columnas esperadas para un archivo raw
        if len(df.columns) < 9:
            print(f"⚠️ El archivo no tiene suficientes columnas: {archivo}, tiene {len(df.columns)}")
            return None
        
        # Manejar posibles errores en los datos
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Verificar si no hay datos suficientes
        if len(df) < 5:
            print(f"⚠️ El archivo tiene muy pocos datos: {archivo}, filas: {len(df)}")
            return None
        
        # Extraer timestamp y datos de canales
        timestamp = df.iloc[:, 0]
        datos_eeg = df.iloc[:, 1:9].values
        
        # Crear DataFrame para los resultados de bandpower
        bp_columns = ["Timestamp"]
        for i in range(8):
            for banda in BANDAS.keys():
                bp_columns.append(f"Canal{i+1}_{banda}")
        
        # Calcular bandpower para cada punto del tiempo
        bp_results = []
        window_size = 250  # 1 segundo de datos
        
        # Si no hay suficientes puntos, usar todos los disponibles
        if len(datos_eeg) < window_size:
            window_size = len(datos_eeg)
            if window_size < 50:  # Si hay menos de 50 puntos, no es suficiente para un análisis útil
                print(f"⚠️ No hay suficientes datos para calcular bandpower: {archivo}, puntos: {len(datos_eeg)}")
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
                    # Si los datos no son válidos, usar valores predeterminados bajos
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
        
        # Guardar versión bandpower si es necesario
        if guardar_bp:
            output_path = archivo.replace(".csv", "_BP.csv")
            bp_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ Guardado archivo bandpower: {output_path}")
        
        return bp_df
        
    except Exception as e:
        print(f"Error procesando archivo raw {archivo}: {str(e)}")
        traceback.print_exc()
        return None

def generar_grafico_general(df, archivo, carpeta_salida):
    """
    Genera un gráfico general de todas las bandas
    """
    try:
        print(f"Generando gráfico general para: {archivo}")
        
        # Crear carpeta de salida si no existe
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)
            
        # Obtener nombre base para el gráfico
        nombre_base = os.path.splitext(os.path.basename(archivo))[0]
        output_path = os.path.join(carpeta_salida, f"{nombre_base}.png")
        
        # Detectar columnas por banda
        promedios_por_banda = {}
        for banda in BANDAS.keys():
            cols = [col for col in df.columns if banda in col]
            if cols:
                promedios_por_banda[banda] = df[cols].mean(axis=1)
        
        # Verificar si hay datos para graficar
        if not promedios_por_banda:
            print(f"⚠️ No se encontraron columnas de bandas en {archivo}")
            return None
        
        # Crear gráfico
        plt.figure(figsize=(12, 6))
        
        for banda, valores in promedios_por_banda.items():
            plt.plot(valores, label=banda, color=COLORES.get(banda, 'black'), linewidth=2)
        
        plt.title(f"{nombre_base} - Bandas de Frecuencia EEG", fontsize=14)
        plt.xlabel("Muestras", fontsize=12)
        plt.ylabel("Potencia Normalizada", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"✅ Gráfico guardado: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generando gráfico para {archivo}: {str(e)}")
        traceback.print_exc()
        return None

def generar_grafico_c3c4(df, archivo, carpeta_salida):
    """
    Genera un gráfico específico para C3 y C4 con Alpha y Beta
    """
    try:
        print(f"Generando gráfico C3/C4 para: {archivo}")
        
        # Crear carpeta de salida si no existe
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)
            
        # Obtener nombre base para el gráfico
        nombre_base = os.path.splitext(os.path.basename(archivo))[0]
        output_path = os.path.join(carpeta_salida, f"{nombre_base}_C3C4_AlphaBeta.png")
        
        # Extraer columnas para C3 y C4 (canales 4 y 5)
        c3c4_data = {}
        for canal in CANALES_INTERES:
            for banda in BANDAS_INTERES:
                col_name = f"Canal{canal}_{banda}"
                if col_name in df.columns:
                    nombre = f"{'C3' if canal == 4 else 'C4'} - {banda}"
                    c3c4_data[nombre] = df[col_name]
        
        # Si no hay datos para estos canales, salir
        if not c3c4_data:
            print(f"⚠️ No se encontraron datos para C3/C4 en {archivo}")
            return None
        
        # Crear gráfico
        plt.figure(figsize=(12, 6))
        
        for nombre, valores in c3c4_data.items():
            canal = 'C3' if 'C3' in nombre else 'C4'
            banda = 'Alpha' if 'Alpha' in nombre else 'Beta'
            color = COLORES[banda]
            estilo = '-' if canal == 'C3' else '--'
            plt.plot(valores, label=nombre, color=color, linestyle=estilo, linewidth=2.5)
        
        plt.title(f"{nombre_base} - Canales C3/C4, Bandas Alpha/Beta", fontsize=14)
        plt.xlabel("Muestras", fontsize=12)
        plt.ylabel("Potencia Normalizada", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"✅ Gráfico C3/C4 guardado: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generando gráfico C3/C4 para {archivo}: {str(e)}")
        traceback.print_exc()
        return None

def procesar_archivo_bandpower(archivo, comment_lines=0):
    """
    Procesa un archivo bandpower directamente
    """
    try:
        print(f"Procesando archivo bandpower: {archivo}")
        
        # Leer archivo, intentando diferentes codificaciones y saltando líneas de comentario
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
            try:
                df = pd.read_csv(archivo, skiprows=comment_lines, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                # Intentar con delimitadores diferentes si hay error de parseo
                try:
                    df = pd.read_csv(archivo, skiprows=comment_lines, encoding=encoding, 
                                    sep=None, delimiter=None, 
                                    engine='python')
                    break
                except:
                    continue
        
        if df is None:
            print(f"⚠️ No se pudo leer el archivo: {archivo}")
            return None
        
        # Verificar si hay suficientes columnas para un archivo bandpower
        if len(df.columns) < 10:  # Al menos Timestamp y algunas bandas
            print(f"⚠️ El archivo no parece ser un archivo bandpower válido: {archivo}, columnas: {len(df.columns)}")
            return None
        
        # Manejar posibles errores en los datos
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Verificar si hay datos suficientes
        if len(df) < 2:
            print(f"⚠️ El archivo bandpower tiene muy pocos datos: {archivo}, filas: {len(df)}")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error procesando archivo bandpower {archivo}: {str(e)}")
        traceback.print_exc()
        return None

def contar_lineas_comentario(archivo):
    """
    Cuenta las líneas de comentario al inicio del archivo
    """
    try:
        comment_lines = 0
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
        return comment_lines
    except Exception as e:
        print(f"Error al contar líneas de comentario: {str(e)}")
        return 0

def main():
    """
    Función principal del script
    """
    print("="*70)
    print("PROCESADOR DE DATOS EEG - PROYECTO NEUROFLY")
    print("="*70)
    
    # Determinar directorio principal
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Directorio de trabajo: {experiment_dir}")
    
    # Definir directorios
    data_dir = os.path.join(experiment_dir, "data")
    graficas_dir = os.path.join(experiment_dir, "graficas")
    
    # Verificar que existe el directorio de datos
    if not os.path.exists(data_dir):
        print(f"⚠️ No se encontró el directorio de datos: {data_dir}")
        return
    
    # Crear directorio de gráficas
    if not os.path.exists(graficas_dir):
        os.makedirs(graficas_dir)
    
    # Estadísticas
    stats = {
        "archivos_raw": 0,
        "archivos_bandpower": 0,
        "graficos_generados": 0,
        "errores": 0,
        "archivos_salt": 0
    }
    
    # Lista para archivos problemáticos
    archivos_problema = []
    
    # Recorrer carpetas de sujetos
    sujetos = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Se encontraron {len(sujetos)} carpetas de sujeto")
    
    for sujeto in sujetos:
        sujeto_dir = os.path.join(data_dir, sujeto)
        print(f"\nProcesando sujeto: {sujeto}")
        
        # Crear carpeta para gráficas del sujeto
        sujeto_graficas_dir = os.path.join(graficas_dir, f"{sujeto}Graficas")
        if not os.path.exists(sujeto_graficas_dir):
            os.makedirs(sujeto_graficas_dir)
        
        # Procesar archivos CSV del sujeto
        archivos_csv = [f for f in os.listdir(sujeto_dir) if f.endswith('.csv')]
        print(f"Se encontraron {len(archivos_csv)} archivos CSV")
        
        for archivo_csv in archivos_csv:
            archivo_path = os.path.join(sujeto_dir, archivo_csv)
            
            try:
                # Contar líneas de comentario
                comment_lines = contar_lineas_comentario(archivo_path)
                if comment_lines > 0:
                    print(f"El archivo {archivo_csv} tiene {comment_lines} líneas de comentario")
                
                # Detectar tipo de archivo
                tipo = detectar_tipo_archivo(archivo_path)
                
                if tipo == "raw":
                    print(f"Archivo raw: {archivo_csv}")
                    stats["archivos_raw"] += 1
                    
                    # Procesar archivo raw
                    df_bp = procesar_archivo_raw(archivo_path, guardar_bp=True)
                    
                    if df_bp is not None:
                        # Generar gráficos
                        gen_ok = generar_grafico_general(df_bp, archivo_path, sujeto_graficas_dir)
                        c3c4_ok = generar_grafico_c3c4(df_bp, archivo_path, sujeto_graficas_dir)
                        
                        if gen_ok and c3c4_ok:
                            stats["graficos_generados"] += 2
                        elif gen_ok or c3c4_ok:
                            stats["graficos_generados"] += 1
                            archivos_problema.append(f"{archivo_csv} (gráficos parciales)")
                        else:
                            archivos_problema.append(f"{archivo_csv} (sin gráficos)")
                    else:
                        archivos_problema.append(f"{archivo_csv} (error en procesamiento)")
                    
                elif tipo == "bandpower":
                    print(f"Archivo bandpower: {archivo_csv}")
                    stats["archivos_bandpower"] += 1
                    
                    # Leer archivo de bandpower directamente con la función específica
                    df = procesar_archivo_bandpower(archivo_path, comment_lines)
                    
                    if df is not None:
                        # Generar gráficos
                        gen_ok = generar_grafico_general(df, archivo_path, sujeto_graficas_dir)
                        c3c4_ok = generar_grafico_c3c4(df, archivo_path, sujeto_graficas_dir)
                        
                        if gen_ok and c3c4_ok:
                            stats["graficos_generados"] += 2
                        elif gen_ok or c3c4_ok:
                            stats["graficos_generados"] += 1
                            archivos_problema.append(f"{archivo_csv} (gráficos parciales)")
                        else:
                            archivos_problema.append(f"{archivo_csv} (sin gráficos)")
                    else:
                        archivos_problema.append(f"{archivo_csv} (no se pudo leer)")
                
                elif tipo == "error":
                    print(f"❌ Error al procesar el archivo: {archivo_csv}")
                    stats["errores"] += 1
            except Exception as e:
                print(f"Error inesperado al procesar el archivo {archivo_csv}: {str(e)}")
                stats["errores"] += 1

if __name__ == "__main__":
    main()
    