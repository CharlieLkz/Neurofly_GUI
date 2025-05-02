import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from io import StringIO
import time
import torch.nn.functional as F

# Configuración de directorios
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / ".." / "OrganizedData"
GRAPHS_DIR = BASE_DIR / "graficas"
MODELS_DIR = BASE_DIR / "models"

GRAPHS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Semilla para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuración
CONFIG = {
    'batch_size': 5,  # Batch más pequeño para más clases
    'learning_rate': 0.0002,  # Learning rate aún más bajo para fine-tuning
    'weight_decay': 0.0001,  # Regularización L2
    'num_epochs': 350,  # Más épocas para más clases
    'early_stopping_patience': 70,  # Mayor paciencia
    'lr_scheduler_patience': 15,  # Paciencia del scheduler
    'target_samples_per_class': 80,  # Más muestras por clase
    'validation_size': 0.15,  # Porcentaje de datos para validación
    'test_size': 0.15,  # Porcentaje de datos para prueba
    'use_extra_features': True,  # Usar características derivadas
    'use_cross_validation': True,  # Usar validación cruzada
    'n_folds': 5,  # Número de folds para validación cruzada
    'ensemble_models': 3,  # Número de modelos para ensemble
    'use_focal_loss': True,  # Usar Focal Loss para clases desbalanceadas
    'use_class_weights': True,  # Usar ponderación por clase
    'mixup_alpha': 0.2,  # Parámetro para Mixup data augmentation
    'hidden_dim': 384,  # Dimensión de capas ocultas más grande
    'channels': [64, 128, 192],  # Más canales
    'dropout_rate': 0.25,  # Dropout ajustado
    'attention_mechanism': True,  # Usar mecanismo de atención
}

# Focal Loss para clases desbalanceadas
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        CE_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
            
# Clase para el modelo CNN mejorado
class EEGCNNMulticlass(nn.Module):
    def __init__(self, in_channels=20, n_classes=4, config=None):
        super(EEGCNNMulticlass, self).__init__()
        
        # Configuración por defecto si no se proporciona
        if config is None:
            config = {
                'hidden_dim': 384,
                'channels': [64, 128, 192],
                'dropout_rate': 0.25,
                'attention_mechanism': True
            }
        
        # Parámetros de la arquitectura
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.hidden_dim = config.get('hidden_dim', 384)
        self.channels = config.get('channels', [64, 128, 192])
        self.dropout_rate = config.get('dropout_rate', 0.25)
        self.use_attention = config.get('attention_mechanism', True)
        
        # Capas convolucionales con BatchNorm
        self.conv1 = nn.Conv1d(self.in_channels, self.channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.channels[0])
        self.conv2 = nn.Conv1d(self.channels[0], self.channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.channels[1])
        self.conv3 = nn.Conv1d(self.channels[1], self.channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(self.channels[2])
        
        # Mecanismo de atención completo (como en el checkpoint)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv1d(self.channels[2], self.channels[2] // 4, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(self.channels[2] // 4, self.channels[2], kernel_size=1),
                nn.Sigmoid()
            )
        
        # Capas fully connected con la arquitectura original
        # Calculamos fc1_features basándonos en la forma de entrada esperada
        self.fc1_features = self.channels[2] * 2  # Después de los max_pool
        self.fc1 = nn.Linear(self.fc1_features, self.hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.n_classes)
        
        # Dropout con tasas diferentes
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate + 0.05)  # Más dropout en capas fully connected
    
    def forward(self, x):
        # Primera capa convolucional
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.dropout1(x)
        
        # Segunda capa convolucional
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.dropout1(x)
        
        # Tercera capa convolucional
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.dropout1(x)
        
        # Mecanismo de atención
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Aplanar para capas fully connected
        x = x.view(x.size(0), -1)
        
        # Capas fully connected con BatchNorm y Dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def set_inference_mode(self):
        """Configura el modelo específicamente para inferencia"""
        self.eval()  # Poner en modo evaluación
        
        # Configurar todas las capas para inferencia
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
                module.track_running_stats = False
            elif isinstance(module, nn.Dropout):
                module.eval()
    
    def get_attention_weights(self, x):
        """Obtiene los pesos de atención para visualización"""
        if not self.use_attention:
            return None
        
        # Procesar hasta obtener los pesos de atención
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        
        attention_weights = self.attention(x)
        return attention_weights

# Dataset personalizado para datos EEG
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def compute_derived_features(df):
    """
    Calcula características derivadas a partir de las bandas de frecuencia originales
    """
    # Las 4 características originales
    original_features = df.copy()
    
    # Relación Alpha/Beta para cada canal
    df['Canal4_Alpha_Beta_Ratio'] = df['Canal4_Alpha'] / (df['Canal4_Beta'] + 1e-10)
    df['Canal5_Alpha_Beta_Ratio'] = df['Canal5_Alpha'] / (df['Canal5_Beta'] + 1e-10)
    
    # Diferencia entre canales para cada banda
    df['Alpha_Channel_Diff'] = df['Canal4_Alpha'] - df['Canal5_Alpha']
    df['Beta_Channel_Diff'] = df['Canal4_Beta'] - df['Canal5_Beta']
    
    # Suma de poder por banda entre canales
    df['Alpha_Power_Total'] = df['Canal4_Alpha'] + df['Canal5_Alpha']
    df['Beta_Power_Total'] = df['Canal4_Beta'] + df['Canal5_Beta']
    
    # Dominancia hemisférica (proporción entre canales)
    df['Alpha_Hemisphere_Ratio'] = df['Canal4_Alpha'] / (df['Canal5_Alpha'] + 1e-10)
    df['Beta_Hemisphere_Ratio'] = df['Canal4_Beta'] / (df['Canal5_Beta'] + 1e-10)
    
    # Variación temporal (aproximación de la derivada)
    for col in original_features.columns:
        df[f'{col}_Diff'] = df[col].diff().fillna(0)
    
    # Medidas de variabilidad
    for col in original_features.columns:
        # Ventana deslizante para calcular la varianza local (3 muestras)
        rolling = df[col].rolling(window=3, min_periods=1)
        df[f'{col}_RollingVar'] = rolling.var().fillna(0)
    
    return df

def load_and_preprocess_data(use_extra_features=True):
    """
    Carga y preprocesa los datos de EEG desde las carpetas especificadas.
    Eliminando la columna Timestamp y procesa solo los valores de BandPower.
    """
    # Definir las clases thinking para clasificar - CUATRO CLASES
    thinking_classes = [
        "LeftArmThinking",  # Mantener las dos clases originales
        "RightArmThinking", 
        "LeftFistThinking",  # Añadir estas dos clases nuevas
        "RightFistThinking"
    ]
    # Incluir clases moving como datos de referencia
    moving_classes = [
        "LeftArmMoving",
        "RightArmMoving",
        "LeftFistMoving",
        "RightFistMoving"
    ]
    
    all_data = []
    all_labels = []
    label_map = {class_name: idx for idx, class_name in enumerate(thinking_classes)}
    
    # Variable para almacenar la ruta de datos final
    data_directory = DATA_DIR
    
    # Verificar que existe el directorio de datos
    if not os.path.exists(data_directory):
        print(f"ERROR: No se encuentra el directorio de datos: {data_directory}")
        print(f"Directorio actual: {os.getcwd()}")
        print(f"Contenido del directorio actual: {os.listdir('.')}")
        print(f"Contenido del directorio padre: {os.listdir('..')}")
        
        # Intentar encontrar la carpeta OrganizedData en algún lugar cercano
        possible_locations = [
            Path("../OrganizedData"),
            Path("../../OrganizedData"),
            Path("../../../OrganizedData"),
            Path("./OrganizedData"),
            Path("../../experiment_gui/OrganizedData"),
        ]
        
        for loc in possible_locations:
            if os.path.exists(loc):
                print(f"¡ENCONTRADA! La carpeta OrganizedData está en: {os.path.abspath(loc)}")
                data_directory = loc
                break
        else:
            print("No se pudo encontrar la carpeta OrganizedData en ninguna ubicación cercana.")
            sys.exit(1)
    
    # Imprimir estructura de carpetas para debug
    print(f"Estructura de carpetas de datos:")
    try:
        for root, dirs, files in os.walk(data_directory):
            level = root.replace(str(data_directory), '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files[:5]:  # Mostrar primeros 5 archivos de cada carpeta
                print(f"{sub_indent}{f}")
            if len(files) > 5:
                print(f"{sub_indent}... y {len(files)-5} archivos más")
    except Exception as e:
        print(f"Error al mostrar estructura de carpetas: {str(e)}")
    
    # Contador de archivos procesados
    files_processed = 0
    
    # NUEVO: Priorizar archivos BP (BandPower)
    # Cargar datos de las clases thinking (objetivo)
    print(f"\nCargando datos de clases thinking...")
    for class_name in thinking_classes:
        class_dir = data_directory / class_name
        if not os.path.exists(class_dir):
            print(f"ADVERTENCIA: No existe la carpeta {class_dir}")
            continue
            
        # Primero buscar archivos con _BP (que sabemos que funcionan)
        csv_files = sorted(glob.glob(os.path.join(class_dir, "*_BP.csv")))
        
        if not csv_files:  # Si no hay archivos BP, usar los normales
            csv_files = glob.glob(os.path.join(class_dir, "*.csv"))
            
        print(f"  - {class_name}: {len(csv_files)} archivos encontrados")
        
        for csv_file in csv_files:
            try:
                # Intentar diferentes separadores y métodos de parseo
                try:
                    # Primero intentar método estándar
                    df = pd.read_csv(csv_file, on_bad_lines='skip')
                except:
                    # Si falla, intentar con delimitador de tab
                    try:
                        df = pd.read_csv(csv_file, sep='\t', on_bad_lines='skip')
                    except:
                        # Si también falla, intentar leer manualmente
                        with open(csv_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Filtrar líneas válidas (que tengan 4 o 5 valores numéricos)
                        data_lines = []
                        for line in lines[1:]:  # Saltamos la cabecera
                            values = line.strip().split(',')
                            if len(values) >= 4:  # Al menos 4 valores (podría tener Timestamp)
                                try:
                                    # Ver si podemos convertir a float los valores
                                    if len(values) == 5:  # Con Timestamp
                                        [float(v) for v in values[1:]]  # Ignorar Timestamp
                                        data_lines.append(','.join(values[1:]))  # Guardar sin Timestamp
                                    else:  # Sin Timestamp
                                        [float(v) for v in values]
                                        data_lines.append(','.join(values))
                                except:
                                    pass  # Si no se pueden convertir, ignorar la línea
                        
                        # Reconstruir CSV con las líneas válidas
                        header = "Canal4_Alpha,Canal4_Beta,Canal5_Alpha,Canal5_Beta"
                        csv_content = header + '\n' + '\n'.join(data_lines)
                        
                        # Crear DataFrame
                        df = pd.read_csv(StringIO(csv_content))
                
                # Verificar que tenga datos
                if df.empty:
                    print(f"    ADVERTENCIA: El archivo {csv_file} está vacío")
                    continue
                
                # Verificar las columnas esperadas
                expected_columns = ['Canal4_Alpha', 'Canal4_Beta', 'Canal5_Alpha', 'Canal5_Beta']
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"    ADVERTENCIA: El archivo {csv_file} le faltan columnas: {missing_columns}")
                    continue
                
                # Eliminar columna Timestamp si existe
                if 'Timestamp' in df.columns:
                    df = df.drop('Timestamp', axis=1)
                
                # Eliminar filas con NaN
                df = df.dropna()
                
                # Eliminar outliers (valores extremos)
                for col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                # Verificar que aún queden datos después de limpiar
                if df.empty or len(df) < 10:  # Mínimo 10 filas para que sea útil
                    print(f"    ADVERTENCIA: El archivo {csv_file} quedó con muy pocos datos después de limpiar")
                    continue
                
                # Verificar las columnas para depuración
                if files_processed == 0:
                    print(f"    Columnas en el primer CSV: {df.columns.tolist()}")
                    print(f"    Primeras 2 filas: \n{df.head(2)}")
                
                # Crear características derivadas si se solicita
                if use_extra_features:
                    df = compute_derived_features(df)
                
                # Normalizar los datos dentro de cada archivo
                scaler = StandardScaler()
                df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
                
                # Añadir datos a la lista
                all_data.append(df_normalized.values)
                all_labels.append(label_map[class_name])
                files_processed += 1
                
            except Exception as e:
                print(f"    ERROR al procesar {csv_file}: {str(e)}")
    
    # Cargar datos de las clases moving (referencia)
    print(f"\nCargando datos de clases moving...")
    for class_name in moving_classes:
        class_dir = data_directory / class_name
        if not os.path.exists(class_dir):
            print(f"ADVERTENCIA: No existe la carpeta {class_dir}")
            continue
        
        # Primero buscar archivos con _BP (que sabemos que funcionan)
        csv_files = sorted(glob.glob(os.path.join(class_dir, "*_BP.csv")))
        
        if not csv_files:  # Si no hay archivos BP, usar los normales
            csv_files = glob.glob(os.path.join(class_dir, "*.csv"))
            
        print(f"  - {class_name}: {len(csv_files)} archivos encontrados")
        
        for csv_file in csv_files:
            try:
                # Intentar diferentes separadores y métodos de parseo
                try:
                    # Primero intentar método estándar
                    df = pd.read_csv(csv_file, on_bad_lines='skip')
                except:
                    # Si falla, intentar con delimitador de tab
                    try:
                        df = pd.read_csv(csv_file, sep='\t', on_bad_lines='skip')
                    except:
                        # Si también falla, intentar leer manualmente
                        with open(csv_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Filtrar líneas válidas (que tengan 4 o 5 valores numéricos)
                        data_lines = []
                        for line in lines[1:]:  # Saltamos la cabecera
                            values = line.strip().split(',')
                            if len(values) >= 4:  # Al menos 4 valores (podría tener Timestamp)
                                try:
                                    # Ver si podemos convertir a float los valores
                                    if len(values) == 5:  # Con Timestamp
                                        [float(v) for v in values[1:]]  # Ignorar Timestamp
                                        data_lines.append(','.join(values[1:]))  # Guardar sin Timestamp
                                    else:  # Sin Timestamp
                                        [float(v) for v in values]
                                        data_lines.append(','.join(values))
                                except:
                                    pass  # Si no se pueden convertir, ignorar la línea
                        
                        # Reconstruir CSV con las líneas válidas
                        header = "Canal4_Alpha,Canal4_Beta,Canal5_Alpha,Canal5_Beta"
                        csv_content = header + '\n' + '\n'.join(data_lines)
                        
                        # Crear DataFrame
                        df = pd.read_csv(StringIO(csv_content))
                
                # Verificar que tenga datos
                if df.empty:
                    print(f"    ADVERTENCIA: El archivo {csv_file} está vacío")
                    continue
                
                # Verificar las columnas esperadas
                expected_columns = ['Canal4_Alpha', 'Canal4_Beta', 'Canal5_Alpha', 'Canal5_Beta']
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"    ADVERTENCIA: El archivo {csv_file} le faltan columnas: {missing_columns}")
                    continue
                
                # Eliminar columna Timestamp si existe
                if 'Timestamp' in df.columns:
                    df = df.drop('Timestamp', axis=1)
                
                # Eliminar filas con NaN
                df = df.dropna()
                
                # Eliminar outliers (valores extremos)
                for col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                # Verificar que aún queden datos después de limpiar
                if df.empty or len(df) < 10:  # Mínimo 10 filas para que sea útil
                    print(f"    ADVERTENCIA: El archivo {csv_file} quedó con muy pocos datos después de limpiar")
                    continue
                
                # Crear características derivadas si se solicita
                if use_extra_features:
                    df = compute_derived_features(df)
                
                # Normalizar los datos dentro de cada archivo
                scaler = StandardScaler()
                df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
                
                # Para los datos moving, buscamos su equivalente thinking
                thinking_equivalent = class_name.replace("Moving", "Thinking")
                if thinking_equivalent in thinking_classes:
                    # Añadir datos a la lista con la etiqueta del thinking equivalente
                    all_data.append(df_normalized.values)
                    all_labels.append(label_map[thinking_equivalent])
                    files_processed += 1
                    
            except Exception as e:
                print(f"    ERROR al procesar {csv_file}: {str(e)}")
    
    print(f"\nTotal de archivos procesados correctamente: {files_processed}")
    
    # Verificar que se han cargado datos
    if len(all_data) == 0:
        print("ERROR: No se han podido cargar datos. Verifica la estructura de carpetas y archivos.")
        sys.exit(1)
    
    # Convertir listas a arrays numpy
    print(f"Forma de los datos antes de procesamiento:")
    for i in range(min(5, len(all_data))):
        print(f"  - Ejemplo {i}: {all_data[i].shape}")
    
    # Encontrar la forma común para todos los ejemplos
    min_rows = min(arr.shape[0] for arr in all_data)
    n_features = all_data[0].shape[1]
    print(f"Estandarizando datos: {min_rows} filas por {n_features} columnas")
    
    # Crear un array 3D con dimensiones consistentes
    X = np.zeros((len(all_data), min_rows, n_features))
    for i, arr in enumerate(all_data):
        X[i, :, :] = arr[:min_rows, :]
    
    y = np.array(all_labels)
    
    print(f"Forma de X después de estandarizar: {X.shape}")
    print(f"Forma de y: {y.shape}")
    print(f"Distribución de clases: {np.bincount(y)}")
    
    # Reorganizar para entrada de CNN (batch, channels, seq_len)
    X = X.transpose(0, 2, 1)
    
    print(f"Forma final de X (para CNN): {X.shape}")
    
    return X, y, thinking_classes
    
def mixup_data(x, y, alpha=0.2):
    """
    Mixup data augmentation.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Criterion for mixup.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def advanced_data_augmentation(X, y, target_samples_per_class):
    """
    Técnicas avanzadas de data augmentation para datos EEG
    """
    X_augmented = []
    y_augmented = []
    
    # Contador de clases
    class_counts = np.bincount(y)
    print(f"Distribución original de clases: {class_counts}")
    
    # Augmentar datos para cada clase
    for class_idx in np.unique(y):
        # Índices de las muestras de esta clase
        indices = np.where(y == class_idx)[0]
        
        # Añadir las muestras originales
        for idx in indices:
            X_augmented.append(X[idx])
            y_augmented.append(y[idx])
        
        # Si hay pocas muestras, aumentarlas
        if len(indices) < target_samples_per_class:
            # Calcular cuántas muestras adicionales necesitamos
            samples_to_generate = target_samples_per_class - len(indices)
            print(f"Generando {samples_to_generate} muestras adicionales para clase {class_idx}")
            
            # Generar muestras adicionales con múltiples técnicas
            for _ in range(samples_to_generate):
                # Mezcla de técnicas para mayor variedad
                n_techniques = np.random.randint(1, 4)  # Aplicar 1-3 técnicas a la vez
                
                # Elegir aleatoriamente una muestra de esta clase como base
                sample_idx = np.random.choice(indices)
                augmented_sample = X[sample_idx].copy()
                
                # Aplicar técnicas seleccionadas
                for _ in range(n_techniques):
                    technique = np.random.choice(['noise', 'shift', 'mix', 'scale', 'flip', 'filter', 'jitter'])
                    
                    if technique == 'noise':
                        # Ruido gaussiano selectivo (solo a algunos canales)
                        noise_level = np.random.uniform(0.01, 0.05)
                        channels_to_noise = np.random.choice(augmented_sample.shape[0], 
                                                            size=np.random.randint(1, augmented_sample.shape[0]+1),
                                                            replace=False)
                        
                        noise = np.zeros_like(augmented_sample)
                        for ch in channels_to_noise:
                            noise[ch] = np.random.normal(0, noise_level * np.std(augmented_sample[ch]), augmented_sample[ch].shape)
                        
                        augmented_sample = augmented_sample + noise
                    
                    elif technique == 'shift':
                        # Desplazamiento temporal aleatorio variado por canal
                        shifted_sample = np.zeros_like(augmented_sample)
                        for ch in range(augmented_sample.shape[0]):
                            shift_amount = np.random.randint(-3, 4)  # Desplazamiento entre -3 y 3
                            if shift_amount > 0:
                                shifted_sample[ch, shift_amount:] = augmented_sample[ch, :-shift_amount]
                            elif shift_amount < 0:
                                shifted_sample[ch, :shift_amount] = augmented_sample[ch, -shift_amount:]
                            else:  # No shift
                                shifted_sample[ch] = augmented_sample[ch]
                        
                        augmented_sample = shifted_sample
                    
                    elif technique == 'mix':
                        # Mezclar dos muestras de la misma clase
                        if len(indices) >= 2:
                            other_idx = np.random.choice([i for i in indices if i != sample_idx])
                            other_sample = X[other_idx].copy()
                            
                            # Mezclar con proporción aleatoria
                            mix_ratio = np.random.uniform(0.2, 0.8)
                            augmented_sample = augmented_sample * mix_ratio + other_sample * (1 - mix_ratio)
                    
                    elif technique == 'scale':
                        # Escalar por canal con factores diferentes
                        scaled_sample = np.zeros_like(augmented_sample)
                        for ch in range(augmented_sample.shape[0]):
                            scale_factor = np.random.uniform(0.8, 1.2)  # Factor de escala entre 0.8 y 1.2
                            scaled_sample[ch] = augmented_sample[ch] * scale_factor
                        
                        augmented_sample = scaled_sample
                    
                    elif technique == 'flip':
                        # Voltear canales seleccionados
                        flipped_sample = augmented_sample.copy()
                        channels_to_flip = np.random.choice([True, False], size=augmented_sample.shape[0])
                        
                        for ch in range(augmented_sample.shape[0]):
                            if channels_to_flip[ch]:
                                flipped_sample[ch] = -augmented_sample[ch]  # Invertir señal
                        
                        augmented_sample = flipped_sample
                    
                    elif technique == 'filter':
                        # Filtrado simple (suavizado por media móvil)
                        filtered_sample = augmented_sample.copy()
                        window_size = np.random.randint(2, 5)
                        
                        for ch in range(augmented_sample.shape[0]):
                            kernel = np.ones(window_size) / window_size
                            # Convolución 1D para suavizado
                            smoothed = np.convolve(augmented_sample[ch], kernel, mode='same')
                            filtered_sample[ch] = smoothed
                        
                        augmented_sample = filtered_sample
                    
                    elif technique == 'jitter':
                        # Añadir jitter (pequeñas variaciones aleatorias en tiempo)
                        jittered_sample = augmented_sample.copy()
                        
                        for ch in range(augmented_sample.shape[0]):
                            # Pequeño desplazamiento aleatorio para cada punto
                            for i in range(1, augmented_sample.shape[1]-1):
                                if np.random.random() > 0.7:  # 30% de probabilidad
                                    direction = np.random.choice([-1, 1])
                                    jittered_sample[ch, i] = augmented_sample[ch, i + direction]
                        
                        augmented_sample = jittered_sample
                
                X_augmented.append(augmented_sample)
                y_augmented.append(class_idx)
    
    # Convertir a arrays numpy
    X_result = np.array(X_augmented)
    y_result = np.array(y_augmented)
    
    print(f"Distribución después de augmentation: {np.bincount(y_result)}")
    print(f"Forma de X después de augmentation: {X_result.shape}")
    
    return X_result, y_result

def train_and_evaluate_model(X, y, config, class_names):
    """
    Entrena y evalúa el modelo utilizando validación cruzada si está habilitada
    """
    if config['use_cross_validation']:
        return cross_validation_training(X, y, config, class_names)
    else:
        return single_model_training(X, y, config, class_names)

def single_model_training(X, y, config, class_names):
    """
    Entrenamiento de un único modelo con división simple de datos
    """
    # Dividir datos en conjuntos de entrenamiento, validación y prueba
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=config['validation_size'], 
        random_state=SEED, stratify=y_train_val
    )
    
    print(f"Forma de los datos de entrenamiento: {X_train.shape}")
    print(f"Forma de los datos de validación: {X_val.shape}")
    print(f"Forma de los datos de prueba: {X_test.shape}")
    
    # Crear datasets y dataloaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Determinar dispositivo (GPU si está disponible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    # Crear modelo
    n_features, seq_len = X_train.shape[1], X_train.shape[2]
    num_classes = len(class_names)
    
    model = EEGCNNMulticlass(n_features, num_classes, config).to(device)
    print("Arquitectura del modelo:")
    print(model)
    
    print("\nIniciando entrenamiento...")
    train_losses, val_losses, val_accuracies, best_model_state = train_model(
        model, train_loader, val_loader, device, config
    )
    
    # Restaurar el mejor modelo
    model.load_state_dict(best_model_state)
    
    print("\nEvaluando modelo en conjunto de prueba...")
    test_accuracy, confusion_mat = evaluate_model(model, test_loader, device, class_names)
    
    # Evaluar la precisión por clase para verificar si las clases originales mantienen alta precisión
    class_accuracies = evaluate_class_performance(model, test_loader, device, class_names)
    
    # Guardar el modelo entrenado
    if test_accuracy >= 0.80:
        model_save_path = MODELS_DIR / f"eeg_model_multiclass_{test_accuracy:.4f}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'accuracy': test_accuracy,
            'class_names': class_names,
            'n_features': n_features,
            'seq_len': seq_len,
            'class_accuracies': class_accuracies
        }, model_save_path)
        print(f"\nModelo guardado en: {model_save_path}")
    
    # Visualizar curvas de pérdida y precisión
    plot_training_metrics(train_losses, val_losses, val_accuracies)
    
    return model, test_accuracy, confusion_mat, class_accuracies

def cross_validation_training(X, y, config, class_names):
    """
    Entrena y evalúa utilizando validación cruzada para mayor robustez
    """
    print(f"\nIniciando entrenamiento con {config['n_folds']}-fold validación cruzada...")
    
    # Configurar K-Fold con stratify para mantener la distribución de clases
    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=SEED)
    fold_accuracies = []
    fold_class_accuracies = []
    fold_models = []
    
    # Determinar dispositivo (GPU si está disponible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    # Para cada fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{config['n_folds']} ---")
        
        # Dividir los datos para este fold
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Separar validación (manteniendo stratify)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold, test_size=config['validation_size'], 
            random_state=SEED, stratify=y_train_fold
        )
        
        # Crear datasets y dataloaders
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test_fold, y_test_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        
        # Crear modelo para este fold
        n_features, seq_len = X_train.shape[1], X_train.shape[2]
        num_classes = len(class_names)
        
        fold_model = EEGCNNMulticlass(n_features, num_classes, config).to(device)
        
        # Entrenar modelo
        _, _, _, best_state = train_model(
            fold_model, train_loader, val_loader, device, config
        )
        
        # Cargar el mejor estado del modelo
        fold_model.load_state_dict(best_state)
        
        # Evaluar en el conjunto de prueba
        fold_acc, _ = evaluate_model(
            fold_model, test_loader, device, class_names
        )
        
        # Evaluar la precisión por clase
        class_accs = evaluate_class_performance(fold_model, test_loader, device, class_names)
        fold_class_accuracies.append(class_accs)
        
        print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")
        print(f"Accuracy por clase: {class_accs}")
        fold_accuracies.append(fold_acc)
        fold_models.append((fold_model, fold_acc))
    
    # Métricas globales
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nValidación cruzada completa:")
    print(f"Accuracy promedio: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Calcular promedio de precisión por clase
    avg_class_accuracies = {}
    for class_idx, class_name in enumerate(class_names):
        class_accs = [fold_acc[class_name] for fold_acc in fold_class_accuracies]
        avg_class_accuracies[class_name] = np.mean(class_accs)
    
    print(f"Accuracy promedio por clase:")
    for class_name, acc in avg_class_accuracies.items():
        print(f"  - {class_name}: {acc:.4f}")
    
    # Seleccionar el mejor modelo
    best_model_idx = np.argmax(fold_accuracies)
    best_model, best_acc = fold_models[best_model_idx]
    best_class_accuracies = fold_class_accuracies[best_model_idx]
    
    # Guardar el mejor modelo
    if best_acc >= 0.80:
        model_save_path = MODELS_DIR / f"eeg_model_cv_best_{best_acc:.4f}.pt"
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'config': config,
            'accuracy': best_acc,
            'class_accuracies': best_class_accuracies,
            'class_names': class_names,
            'cross_validation_mean': mean_acc,
            'cross_validation_std': std_acc,
            'n_features': n_features,
            'seq_len': seq_len
        }, model_save_path)
        print(f"\nMejor modelo guardado en: {model_save_path}")
    
    # Evaluar todos los modelos en todo el conjunto
    if config['ensemble_models'] > 1 and len(fold_models) >= config['ensemble_models']:
        # Ordenar modelos por precisión y tomar los mejores n
        top_models = sorted(fold_models, key=lambda x: x[1], reverse=True)[:config['ensemble_models']]
        ensemble_model_list = [model for model, _ in top_models]
        
        # Crear dataset de todo el conjunto
        full_dataset = EEGDataset(X, y)
        full_loader = DataLoader(full_dataset, batch_size=config['batch_size'])
        
        # Evaluar ensemble
        ensemble_acc, ensemble_class_accs = evaluate_ensemble(ensemble_model_list, full_loader, device, class_names)
        print(f"\nEnsemble de {config['ensemble_models']} modelos:")
        print(f"Accuracy global: {ensemble_acc:.4f}")
        print(f"Accuracy por clase: {ensemble_class_accs}")
        
        # Guardar modelos ensemble
        ensemble_dir = MODELS_DIR / f"ensemble_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(ensemble_dir, exist_ok=True)
        
        for i, (model, acc) in enumerate(top_models[:config['ensemble_models']]):
            model_save_path = ensemble_dir / f"model_{i+1}_{acc:.4f}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'accuracy': acc,
                'class_names': class_names,
                'n_features': n_features,
                'seq_len': seq_len
            }, model_save_path)
        
        # Guardar metadata del ensemble
        ensemble_meta_path = ensemble_dir / "ensemble_info.txt"
        with open(ensemble_meta_path, 'w') as f:
            f.write(f"Ensemble creado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Accuracy global: {ensemble_acc:.4f}\n")
            f.write(f"Accuracy por clase:\n")
            for class_name, acc in ensemble_class_accs.items():
                f.write(f"  - {class_name}: {acc:.4f}\n")
            f.write(f"\nModelos incluidos:\n")
            for i, (_, acc) in enumerate(top_models[:config['ensemble_models']]):
                f.write(f"  - Modelo {i+1}: {acc:.4f}\n")
        
        print(f"Ensemble guardado en: {ensemble_dir}")
        
        return ensemble_model_list, ensemble_acc, None, ensemble_class_accs
    
    return best_model, best_acc, None, best_class_accuracies

def train_model(model, train_loader, val_loader, device, config):
    """
    Entrena el modelo CNN con los datos proporcionados.
    """
    # Definir función de pérdida
    if config['use_focal_loss']:
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Definir optimizador
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=config['lr_scheduler_patience'], 
        factor=0.7, 
        verbose=True
    )
    
    # Listas para almacenar métricas durante el entrenamiento
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Para early stopping
    best_val_accuracy = 0
    counter = 0
    best_model_state = None
    
    # Parámetro para mixup
    mixup_alpha = config['mixup_alpha']
    
    # Entrenamiento del modelo
    for epoch in range(config['num_epochs']):
        # Modo entrenamiento
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Aplicar mixup si está habilitado
            if mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
                
                # Forward pass con mixup
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                # Forward pass normal
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calcular pérdida promedio en entrenamiento
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Modo evaluación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calcular pérdida (siempre sin mixup en validación)
                if isinstance(criterion, FocalLoss):
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Calcular precisión
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calcular pérdida y precisión promedio en validación
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        # Ajustar learning rate
        scheduler.step(val_loss)
        
        # Imprimir progreso cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1}/{config['num_epochs']}, "
                  f"Pérdida entrenamiento: {train_loss:.4f}, "
                  f"Pérdida validación: {val_loss:.4f}, "
                  f"Precisión validación: {val_accuracy:.2f}%")
        
        # Guardar el mejor modelo
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            counter = 0
            print(f"Nueva mejor precisión: {val_accuracy:.2f}% (Época {epoch+1})")
        else:
            counter += 1
            if counter >= config['early_stopping_patience']:
                print(f"Early stopping en época {epoch+1}")
                break
    
    # Restaurar el mejor modelo
    if best_model_state is not None:
        print(f"Restaurado el mejor modelo con precisión de validación: {best_val_accuracy:.2f}%")
    
    return train_losses, val_losses, val_accuracies, best_model_state

def evaluate_model(model, test_loader, device, class_names):
    """
    Evalúa el modelo entrenado y genera una matriz de confusión.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Guardar predicciones y etiquetas verdaderas
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular matriz de confusión
    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Crear gráfica de la matriz de confusión con mejores etiquetas
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title(f'Matriz de Confusión - Accuracy: {accuracy:.4f}')
    
    # Guardar la gráfica
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "confusion_matrix_multiclass.png")
    plt.close()
    
    # Mostrar reporte de clasificación detallado
    print("\nReporte de clasificación:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    return accuracy, cm

def evaluate_class_performance(model, data_loader, device, class_names):
    """
    Evalúa el rendimiento del modelo para cada clase
    """
    model.eval()
    
    # Diccionario para almacenar predicciones y etiquetas por clase
    class_predictions = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Actualizar contadores por clase
            for i in range(len(labels)):
                true_class = class_names[labels[i].item()]
                is_correct = (predicted[i] == labels[i]).item()
                
                class_predictions[true_class]['total'] += 1
                if is_correct:
                    class_predictions[true_class]['correct'] += 1
    
    # Calcular accuracy por clase
    class_accuracies = {}
    for class_name, counts in class_predictions.items():
        if counts['total'] > 0:
            class_accuracies[class_name] = counts['correct'] / counts['total']
        else:
            class_accuracies[class_name] = 0.0
    
    return class_accuracies

def evaluate_ensemble(models, data_loader, device, class_names):
    """
    Evalúa un conjunto de modelos mediante votación
    """
    all_predictions = []
    all_labels = []
    
    # Diccionario para almacenar predicciones y etiquetas por clase
    class_predictions = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Predicciones de cada modelo
            model_predictions = []
            for model in models:
                model.eval()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                model_predictions.append(predicted.cpu().numpy())
            
            # Votación por mayoría
            ensemble_predictions = np.stack(model_predictions)
            final_predictions = []
            
            for i in range(len(labels)):
                # Contar predicciones para cada muestra
                counts = np.bincount(ensemble_predictions[:, i], minlength=len(class_names))
                # La predicción final es la más votada
                final_predictions.append(np.argmax(counts))
                
                # Actualizar contadores por clase
                true_class = class_names[labels[i].item()]
                is_correct = (final_predictions[-1] == labels[i].item())
                
                class_predictions[true_class]['total'] += 1
                if is_correct:
                    class_predictions[true_class]['correct'] += 1
            
            all_predictions.extend(final_predictions)
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular precisión global
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Calcular accuracy por clase
    class_accuracies = {}
    for class_name, counts in class_predictions.items():
        if counts['total'] > 0:
            class_accuracies[class_name] = counts['correct'] / counts['total']
        else:
            class_accuracies[class_name] = 0.0
    
    # Crear matriz de confusión
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Crear gráfica de la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title(f'Matriz de Confusión (Ensemble) - Accuracy: {accuracy:.4f}')
    
    # Guardar la gráfica
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "confusion_matrix_ensemble_multiclass.png")
    plt.close()
    
    return accuracy, class_accuracies

def plot_training_metrics(train_losses, val_losses, val_accuracies):
    """
    Visualiza las métricas de entrenamiento
    """
    # Visualizar curvas de pérdida
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Pérdida de entrenamiento')
    plt.plot(val_losses, label='Pérdida de validación')
    plt.title('Curvas de Pérdida durante Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Visualizar curva de precisión
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Precisión de validación')
    plt.title('Precisión durante Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.ylim([0, 105])
    plt.legend()
    
    # Guardar la gráfica
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "training_metrics_multiclass.png")
    plt.close()

def main():
    """
    Función principal que ejecuta todo el proceso.
    """
    print("Iniciando clasificación avanzada de EEG multiclase...")
    print(f"Configuración: {CONFIG}")
    
    try:
        # Cargar y preprocesar datos
        print("Cargando y preprocesando datos...")
        X, y, class_names = load_and_preprocess_data(use_extra_features=CONFIG['use_extra_features'])
        
        # Verificar que tenemos suficientes datos
        if len(X) < 10:
            print("ERROR: No hay suficientes datos para entrenar el modelo.")
            sys.exit(1)
        
        # Data augmentation
        print("Aplicando técnicas avanzadas de augmentación de datos...")
        X, y = advanced_data_augmentation(X, y, CONFIG['target_samples_per_class'])
        
        # Aplicar Feature Scaling a nivel global
        print("Aplicando normalización avanzada...")
        n_samples, n_features, seq_len = X.shape
        X_reshaped = X.reshape(n_samples, n_features * seq_len)
        
        # StandardScaler para normalizar
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_reshaped)
        
        # Volver a la forma original
        X = X_normalized.reshape(n_samples, n_features, seq_len)
        
        # Entrenar y evaluar el modelo
        model, accuracy, _, class_accuracies = train_and_evaluate_model(X, y, CONFIG, class_names)
        
        # Análisis final
        print("\n=== ANÁLISIS DE RENDIMIENTO FINAL ===")
        print(f"Accuracy global: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Accuracy por clase:")
        for class_name, acc in class_accuracies.items():
            print(f"  - {class_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Verificar si las clases originales mantienen buena precisión
        original_classes = ["LeftArmThinking", "RightArmThinking"]
        original_acc = np.mean([class_accuracies.get(cls, 0) for cls in original_classes])
        print(f"\nAccuracy promedio en las clases originales (Arm): {original_acc:.4f} ({original_acc*100:.2f}%)")
        
        if accuracy >= 0.95:
            print("\n¡Excelente! El modelo ha alcanzado una precisión muy alta (≥95%) en clasificación multiclase.")
            print("Recomendaciones para el siguiente paso:")
            print("1. Probar con todas las clases (incluir Foot)")
            print("2. Evaluar el modelo con datos completamente nuevos")
            print("3. Implementar un sistema en tiempo real utilizando este modelo")
            
            if original_acc >= 0.95:
                print("\nLas clases originales (Arm) mantienen excelente precisión. El modelo es robusto.")
            else:
                print("\n⚠️ Advertencia: Las clases originales (Arm) han perdido precisión. Considera entrenar un modelo separado para cada par de clases.")
                
        elif accuracy >= 0.85:
            print("\nMuy buena precisión (≥85%) en clasificación multiclase. Posibles mejoras adicionales:")
            print("1. Fine-tuning de hiperparámetros (probar con menor learning rate o aumentar dropout)")
            print("2. Aumentar aún más el conjunto de datos (especialmente para las clases con menor precisión)")
            print("3. Probar con técnicas de ensemble más sofisticadas")
            
            if original_acc >= 0.95:
                print("\nLas clases originales (Arm) mantienen excelente precisión. El enfoque está funcionando bien.")
            else:
                print("\n⚠️ Advertencia: Las clases originales (Arm) han perdido algo de precisión. Considera ajustar el modelo.")
                
        else:
            print("\nBuena precisión pero aún se puede mejorar:")
            print("1. Recolectar más datos de ejemplo, especialmente para las clases con peor rendimiento")
            print("2. Probar arquitecturas alternativas (LSTM, Transformer)")
            print("3. Considerar entrenar modelos binarios separados para cada par de clases")
            
            if original_acc >= 0.90:
                print("\nLas clases originales (Arm) mantienen buena precisión. El enfoque es prometedor.")
            else:
                print("\n⚠️ Advertencia: Las clases originales (Arm) han perdido precisión significativa. Reconsiderar enfoque.")
        
        print("\nProceso completo. Las gráficas y modelos han sido guardados.")
        
    except Exception as e:
        print(f"ERROR GENERAL: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTiempo total de ejecución: {(time.time() - start_time)/60:.2f} minutos")