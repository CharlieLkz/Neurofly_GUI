#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar gráficos a partir de archivos CSV con datos EEG procesados con bandpower.
Visualiza las bandas Alpha, Beta, Gamma y Delta sumadas o promediadas por banda.

Características:
- Genera un gráfico general de todas las bandas para cada archivo CSV
- Resalta los datos de los electrodos C3 y C4 (canales 4 y 5)
- Crea un gráfico adicional específico para las bandas Alpha y Beta de estos electrodos
- Mantiene la estructura de carpetas con subcarpetas por sujeto

Estructura de carpetas:
- experiment_gui/
  - data/                    # Carpeta con datos organizados por sujeto
    - Adan/                  # Subcarpeta con datos de un sujeto
      - flecha_arriba.csv    # Archivos CSV con datos EEG procesados
    - Jimena/
      - ...
  - graficas/                # Carpeta de salida para los gráficos
    - AdanGraficas/          # Subcarpetas con el nombre del sujeto + "Graficas"
      - flecha_arriba.png    # Gráfico general
      - flecha_arriba_C3C4_AlphaBeta.png  # Gráfico específico para C3/C4 Alpha/Beta
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import glob

# Colores para cada banda de frecuencia
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

def identificar_columnas_por_banda(columnas):
    """
    Identifica las columnas del DataFrame que corresponden a cada banda de frecuencia.
    
    Args:
        columnas (list): Lista de nombres de columnas del DataFrame
        
    Returns:
        dict: Diccionario donde las claves son los nombres de las bandas y los valores son listas de columnas
    """
    patrones = {
        'Alpha': re.compile(r'.*alpha.*', re.IGNORECASE),
        'Beta': re.compile(r'.*beta.*', re.IGNORECASE),
        'Gamma': re.compile(r'.*gamma.*', re.IGNORECASE),
        'Delta': re.compile(r'.*delta.*', re.IGNORECASE),
        'Theta': re.compile(r'.*theta.*', re.IGNORECASE)
    }
    
    columnas_por_banda = {banda: [] for banda in patrones}
    
    for columna in columnas:
        for banda, patron in patrones.items():
            if patron.match(columna):
                columnas_por_banda[banda].append(columna)
                break
    
    # Eliminar bandas sin columnas
    return {k: v for k, v in columnas_por_banda.items() if v}

def identificar_columnas_por_canal(columnas):
    """
    Identifica las columnas que corresponden a cada canal.
    
    Args:
        columnas (list): Lista de nombres de columnas del DataFrame
        
    Returns:
        dict: Diccionario donde las claves son los números de canal y los valores son listas de columnas
    """
    columnas_por_canal = {}
    
    # Buscar números de canal en los nombres de columnas (Canal1, Canal2, etc., o Ch1, Ch2, etc.)
    patron_canal = re.compile(r'.*(?:Canal|Ch|canal|ch)(\d+).*', re.IGNORECASE)
    
    for columna in columnas:
        match = patron_canal.match(columna)
        if match:
            canal = int(match.group(1))
            if canal not in columnas_por_canal:
                columnas_por_canal[canal] = []
            columnas_por_canal[canal].append(columna)
    
    return columnas_por_canal

def identificar_columnas_especificas(columnas, canales_interes, bandas_interes):
    """
    Identifica columnas específicas por canal y banda de interés.
    
    Args:
        columnas (list): Lista de nombres de columnas
        canales_interes (list): Lista de números de canal de interés
        bandas_interes (list): Lista de nombres de bandas de interés
        
    Returns:
        dict: Diccionario con la estructura {canal: {banda: [columnas]}}
    """
    resultado = {canal: {banda: [] for banda in bandas_interes} for canal in canales_interes}
    
    # Patrones para bandas
    patrones_banda = {
        'Alpha': re.compile(r'.*alpha.*', re.IGNORECASE),
        'Beta': re.compile(r'.*beta.*', re.IGNORECASE),
        'Gamma': re.compile(r'.*gamma.*', re.IGNORECASE),
        'Delta': re.compile(r'.*delta.*', re.IGNORECASE),
        'Theta': re.compile(r'.*theta.*', re.IGNORECASE)
    }
    
    # Patrón para canales
    patron_canal = re.compile(r'.*(?:Canal|Ch|canal|ch)(\d+).*', re.IGNORECASE)
    
    for columna in columnas:
        # Verificar si la columna pertenece a un canal de interés
        match_canal = patron_canal.match(columna)
        if match_canal:
            canal = int(match_canal.group(1))
            if canal in canales_interes:
                # Verificar si la columna pertenece a una banda de interés
                for banda in bandas_interes:
                    if patrones_banda[banda].match(columna):
                        resultado[canal][banda].append(columna)
    
    return resultado

def calcular_valores_por_banda(df, columnas_por_banda):
    """
    Calcula la suma o promedio de los valores para cada banda de frecuencia.
    
    Args:
        df (DataFrame): DataFrame con los datos EEG
        columnas_por_banda (dict): Diccionario con las columnas correspondientes a cada banda
        
    Returns:
        dict: Diccionario donde las claves son los nombres de las bandas y los valores son arrays
    """
    valores_por_banda = {}
    
    for banda, columnas in columnas_por_banda.items():
        if len(columnas) > 0:
            # Calcula el promedio de todas las columnas para esta banda
            valores_por_banda[banda] = df[columnas].mean(axis=1).values
    
    return valores_por_banda

def generar_grafico_general(df, csv_path, output_path):
    """
    Genera un gráfico con las curvas de todas las bandas de frecuencia.
    
    Args:
        df (DataFrame): DataFrame con los datos EEG
        csv_path (str): Ruta del archivo CSV
        output_path (str): Ruta donde se guardará el gráfico
    """
    # Identificar columnas por banda
    columnas_por_banda = identificar_columnas_por_banda(df.columns)
    
    # Si no se encontraron columnas para ninguna banda, mostrar error
    if not any(columnas_por_banda.values()):
        print(f"⚠️ No se encontraron columnas de bandas de frecuencia en {csv_path}")
        return
    
    # Calcular valores por banda
    valores_por_banda = calcular_valores_por_banda(df, columnas_por_banda)
    
    # Configurar el gráfico
    plt.figure(figsize=(12, 6))
    
    # Eje X (tiempo o muestras)
    x = range(len(df))
    
    # Graficar cada banda
    for banda, valores in valores_por_banda.items():
        plt.plot(x, valores, label=banda, color=COLOR_MAP.get(banda, 'black'), linewidth=2)
    
    # Configurar título y etiquetas
    filename = os.path.basename(csv_path)
    title = os.path.splitext(filename)[0]
    plt.title(f"Bandas de frecuencia EEG - {title}", fontsize=16)
    plt.xlabel("Muestras", fontsize=12)
    plt.ylabel("Amplitud (normalizada)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Gráfico general generado: {output_path}")

def generar_grafico_especifico_c3c4(df, csv_path, output_path):
    """
    Genera un gráfico específico para las bandas Alpha y Beta de los electrodos C3 y C4.
    
    Args:
        df (DataFrame): DataFrame con los datos EEG
        csv_path (str): Ruta del archivo CSV
        output_path (str): Ruta donde se guardará el gráfico
    """
    # Identificar columnas específicas por canal y banda
    columnas_especificas = identificar_columnas_especificas(df.columns, CANALES_INTERES, BANDAS_INTERES)
    
    # Verificar si se encontraron columnas para los canales y bandas de interés
    if not any(any(columnas_especificas[canal][banda] for banda in BANDAS_INTERES) for canal in CANALES_INTERES):
        print(f"⚠️ No se encontraron columnas específicas para C3/C4 y Alpha/Beta en {csv_path}")
        return
    
    # Configurar el gráfico
    plt.figure(figsize=(12, 6))
    
    # Eje X (tiempo o muestras)
    x = range(len(df))
    
    # Graficar cada combinación de canal y banda
    for canal in CANALES_INTERES:
        for banda in BANDAS_INTERES:
            columnas = columnas_especificas[canal][banda]
            if columnas:
                # Calcular promedio de valores para esta combinación
                valores = df[columnas].mean(axis=1).values
                nombre_canal = NOMBRES_CANALES.get(canal, f"Canal{canal}")
                label = f"{nombre_canal} - {banda}"
                # Usar línea más gruesa para resaltar
                plt.plot(x, valores, label=label, linewidth=2.5)
    
    # Configurar título y etiquetas
    filename = os.path.basename(csv_path)
    title = os.path.splitext(filename)[0]
    plt.title(f"Electrodos C3/C4 - Bandas Alpha/Beta - {title}", fontsize=16)
    plt.xlabel("Muestras", fontsize=12)
    plt.ylabel("Amplitud (normalizada)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Gráfico específico C3/C4 generado: {output_path}")

def procesar_archivos_csv(experiment_dir):
    """
    Busca y procesa todos los archivos CSV en la estructura de carpetas,
    manteniendo la organización por sujeto.
    
    Args:
        experiment_dir (str): Directorio raíz del experimento
    """
    # Definir directorios
    data_dir = os.path.join(experiment_dir, "data")
    graficas_dir = os.path.join(experiment_dir, "graficas")
    
    # Crear directorio de gráficas principal si no existe
    if not os.path.exists(graficas_dir):
        os.makedirs(graficas_dir)
        print(f"📁 Directorio creado: {graficas_dir}")
    
    # Buscar carpetas de sujetos dentro de data_dir
    carpetas_sujeto = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    if not carpetas_sujeto:
        print(f"⚠️ No se encontraron carpetas de sujeto en {data_dir}")
        return
    
    print(f"🔍 Se encontraron {len(carpetas_sujeto)} carpetas de sujeto para procesar")
    
    # Procesar cada carpeta de sujeto
    for sujeto in carpetas_sujeto:
        carpeta_sujeto = os.path.join(data_dir, sujeto)
        
        # Crear carpeta de salida para este sujeto
        carpeta_salida_sujeto = os.path.join(graficas_dir, f"{sujeto}Graficas")
        os.makedirs(carpeta_salida_sujeto, exist_ok=True)
        print(f"📁 Procesando carpeta del sujeto: {sujeto}")
        
        # Buscar todos los archivos CSV en esta carpeta
        archivos_csv = [f for f in os.listdir(carpeta_sujeto) if f.endswith('.csv')]
        
        if not archivos_csv:
            print(f"⚠️ No se encontraron archivos CSV para el sujeto {sujeto}")
            continue
        
        print(f"   └─ Se encontraron {len(archivos_csv)} archivos CSV")
        
        # Generar gráfico resumen de Alpha/Beta para C3/C4 por sujeto
        # (se realizará al final, después de procesar todos los archivos)
        
        # Procesar cada archivo CSV
        for csv_filename in archivos_csv:
            try:
                csv_path = os.path.join(carpeta_sujeto, csv_filename)
                
                # Leer el archivo CSV
                df = pd.read_csv(csv_path)
                
                # Definir nombres de archivo para los gráficos
                base_filename = os.path.splitext(csv_filename)[0]
                output_general = os.path.join(carpeta_salida_sujeto, f"{base_filename}.png")
                output_especifico = os.path.join(carpeta_salida_sujeto, f"{base_filename}_C3C4_AlphaBeta.png")
                
                # Generar gráficos
                generar_grafico_general(df, csv_path, output_general)
                generar_grafico_especifico_c3c4(df, csv_path, output_especifico)
                
            except Exception as e:
                print(f"❌ Error al procesar {csv_filename}: {str(e)}")
        
        print(f"✅ Finalizado el procesamiento para el sujeto {sujeto}")

def generar_grafico_resumen_por_sujeto(experiment_dir):
    """
    Genera un gráfico resumen de las bandas Alpha y Beta para los electrodos C3 y C4
    para cada sujeto, combinando datos de todos sus archivos CSV.
    Esta función puede implementarse como una extensión futura.
    """
    # Este es un placeholder para una posible extensión futura
    pass

def main():
    """
    Función principal del script.
    """
    # Obtener directorio actual o especificar manualmente
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.dirname(script_dir) if script_dir.endswith("scripts") else script_dir
    
    print("=" * 60)
    print("GENERADOR DE GRÁFICOS PARA DATOS EEG")
    print("=" * 60)
    print(f"📂 Directorio del experimento: {experiment_dir}")
    print(f"📊 Resaltando electrodos: C3 (canal 4) y C4 (canal 5)")
    print(f"📊 Bandas de interés: Alpha y Beta")
    print("=" * 60)
    
    # Procesar los archivos CSV
    procesar_archivos_csv(experiment_dir)
    
    print("=" * 60)
    print("¡Proceso finalizado! Se han generado:")
    print("1. Gráficos generales de todas las bandas")
    print("2. Gráficos específicos para C3/C4 en Alpha/Beta")
    print("=" * 60)

if __name__ == "__main__":
    main()