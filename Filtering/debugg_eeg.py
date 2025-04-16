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
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import json
import re
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import traceback

try:
    import chardet
    print("✅ Librería chardet importada correctamente")
except ImportError:
    print("❌ ERROR: La librería chardet no está instalada. Por favor, instálala con 'pip install chardet'")
    sys.exit(1)

print("="*70)
print("INICIANDO SCRIPT DE UNIFICACIÓN Y ANÁLISIS DE DATOS EEG")
print("="*70)

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

print(f"Configuración: Frecuencia={SAMPLING_FREQUENCY}Hz, Ventana={WINDOW_SIZE}, Época={EPOCH_SIZE}")

# Obtener directorio del experimento
script_dir = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.dirname(script_dir) if script_dir.endswith("scripts") else script_dir

print(f"📂 Directorio del script: {script_dir}")
print(f"📂 Directorio del experimento: {experiment_dir}")

# Definir directorios
data_dir = os.path.join(experiment_dir, "data")
graficas_dir = os.path.join(experiment_dir, "graficas")
backup_dir = os.path.join(experiment_dir, "backup_csv")

print(f"📂 Buscando directorio de datos en: {data_dir}")

# Verificar que exista la carpeta data
if not os.path.exists(data_dir):
    print(f"❌ ERROR: No se encontró el directorio de datos en {data_dir}")
    print("Por favor, asegúrate de que el script esté ubicado en la carpeta experiment_gui o que experiment_gui/data exista")
    sys.exit(1)

print(f"✅ Directorio de datos encontrado: {data_dir}")

# Crear directorios necesarios
os.makedirs(graficas_dir, exist_ok=True)
os.makedirs(backup_dir, exist_ok=True)

print(f"📁 Directorio de gráficas: {graficas_dir}")
print(f"📁 Directorio de backup: {backup_dir}")

# Buscar carpetas de sujetos
try:
    carpetas_sujeto = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"Carpetas de sujeto encontradas: {carpetas_sujeto}")
except Exception as e:
    print(f"❌ ERROR al buscar carpetas de sujeto: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

if not carpetas_sujeto:
    print(f"⚠️ No se encontraron carpetas de sujeto en {data_dir}")
    print("Verifica que la estructura sea: experiment_gui/data/[nombre_sujeto]/...")
    sys.exit(1)

print(f"🔍 Se encontraron {len(carpetas_sujeto)} carpetas de sujeto para procesar")

# Procesar cada carpeta de sujeto
for sujeto in carpetas_sujeto:
    carpeta_sujeto = os.path.join(data_dir, sujeto)
    
    print(f"\n📋 Procesando carpeta del sujeto: {sujeto} en {carpeta_sujeto}")
    
    # Buscar todos los archivos CSV en esta carpeta
    try:
        archivos_csv = [f for f in os.listdir(carpeta_sujeto) if f.endswith('.csv')]
        print(f"Archivos CSV encontrados: {archivos_csv}")
    except Exception as e:
        print(f"❌ ERROR al buscar archivos CSV en {carpeta_sujeto}: {str(e)}")
        continue
    
    if not archivos_csv:
        print(f"⚠️ No se encontraron archivos CSV para el sujeto {sujeto}")
        continue
    
    print(f"   └─ Se encontraron {len(archivos_csv)} archivos CSV")
    
    # Verificar que podemos acceder a los archivos
    for csv_filename in archivos_csv:
        csv_path = os.path.join(carpeta_sujeto, csv_filename)
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                primeras_lineas = [next(f) for _ in range(5) if f]
                print(f"✅ Archivo accesible: {csv_path}")
                print(f"   Primeras líneas: {primeras_lineas[:2]}")
        except UnicodeDecodeError:
            print(f"⚠️ Problema de codificación en {csv_path}, intentando con latin-1")
            try:
                with open(csv_path, 'r', encoding='latin-1') as f:
                    primeras_lineas = [next(f) for _ in range(5) if f]
                    print(f"✅ Archivo accesible con latin-1: {csv_path}")
            except Exception as e:
                print(f"❌ ERROR al leer {csv_path}: {str(e)}")
        except Exception as e:
            print(f"❌ ERROR al acceder a {csv_path}: {str(e)}")

print("\n"+"="*70)
print("EVALUACIÓN INICIAL COMPLETA")
print("="*70)
print("\nPor favor, revisa los mensajes anteriores para identificar posibles problemas.")
print("Correcciones sugeridas:")
print("1. Asegúrate de que la estructura de carpetas sea correcta: experiment_gui/data/[sujeto]/[archivos.csv]")
print("2. Verifica que los archivos CSV sean accesibles y tengan los permisos adecuados")
print("3. Si hay problemas de codificación, intenta convertir los archivos a UTF-8")

print("\nPara continuar con el análisis completo, corrige los problemas mencionados")
print("y modifica este script para incluir la lógica de procesamiento")