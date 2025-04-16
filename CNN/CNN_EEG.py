import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from io import StringIO

# Configuración de directorios
BASE_DIR = Path(".")
DATA_DIR = Path("../OrganizedData")
print(f"Directorio base (script): {os.path.abspath(BASE_DIR)}")
print(f"Directorio de datos: {os.path.abspath(DATA_DIR)}")
GRAPHS_DIR = BASE_DIR / "graficas"

# Crear directorio para gráficas si no existe
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Clase para el modelo CNN
class EEGCNN(nn.Module):
    def __init__(self, input_features, num_classes):
        super(EEGCNN, self).__init__()
        
        # Arquitectura más compleja
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Capas de activación y pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Dropout para regularización (reducido)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Capas fully connected - se ajustarán dinámicamente
        self.fc1 = nn.Linear(128, 256)  # Dimensión inicial, se ajustará dinámicamente
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Aplicar primera capa convolucional con batch normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Aplicar segunda capa convolucional con batch normalization
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Aplicar tercera capa convolucional con batch normalization
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Aplanar el tensor para las capas fully connected
        x = x.view(x.size(0), -1)
        
        # Ajustar dinámicamente la capa fully connected según dimensión real
        if hasattr(self, 'fc1') and self.fc1.in_features != x.size(1):
            old_size = self.fc1.in_features
            new_size = x.size(1)
            print(f"Ajustando capa fc1: {old_size} -> {new_size}")
            # Crear una nueva capa con las dimensiones correctas
            self.fc1 = nn.Linear(new_size, 256).to(x.device)
            self.bn_fc1 = nn.BatchNorm1d(256).to(x.device)
        
        # Aplicar capas fully connected con batch normalization
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# Dataset personalizado para datos EEG
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_and_preprocess_data():
    """
    Carga y preprocesa los datos de EEG desde las carpetas especificadas.
    Elimina la columna Timestamp y procesa solo los valores de BandPower.
    """
    # Definir las clases thinking para clasificar - Reducimos a solo 2 clases
    thinking_classes = [
        "LeftArmThinking", 
        "RightArmThinking"
    ]
    # Incluir clases moving como datos de referencia
    moving_classes = [
        "LeftArmMoving", 
        "RightArmMoving"
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

def train_model(model, train_loader, val_loader, device, num_epochs=300, learning_rate=0.0005):
    """
    Entrena el modelo CNN con los datos proporcionados.
    """
    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)  # Reducir L2 regularización
    
    # Learning rate scheduler - más suave
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.7, verbose=True)
    
    # Listas para almacenar métricas durante el entrenamiento
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Para early stopping
    best_val_accuracy = 0
    patience = 50  # Aumentar paciencia significativamente
    counter = 0
    best_model_state = None
    
    # Entrenamiento del modelo
    for epoch in range(num_epochs):
        # Modo entrenamiento
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
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
            print(f"Época {epoch+1}/{num_epochs}, "
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
            if counter >= patience:
                print(f"Early stopping en época {epoch+1}")
                break
    
    # Restaurar el mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restaurado el mejor modelo con precisión de validación: {best_val_accuracy:.2f}%")
    
    return train_losses, val_losses, val_accuracies

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
    
    # Crear gráfica de la matriz de confusión
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title(f'Matriz de Confusión - Accuracy: {accuracy:.4f}')
    
    # Guardar la gráfica
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "confusion_matrix.png")
    plt.close()
    
    return accuracy, cm

def analyze_performance_and_suggest_improvements(accuracy, train_losses, val_losses):
    """
    Analiza el rendimiento del modelo y sugiere mejoras si la precisión es baja.
    """
    print("\n=== ANÁLISIS DE RENDIMIENTO ===")
    print(f"Accuracy global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Determinar si la accuracy es baja
    if accuracy < 0.8:  # Aumentamos el umbral a 80%
        print("\nLa accuracy es BAJA. Sugerencias para mejorar el modelo:")
        
        # Analizar posible sobreajuste o subajuste
        is_overfit = (train_losses[-1] < val_losses[-1] * 0.7)
        is_underfit = (train_losses[-1] > val_losses[-1] * 0.9)
        
        if is_overfit:
            print("- El modelo muestra signos de sobreajuste (overfitting):")
            print("  * Aumentar regularización (incrementar dropout a 0.4-0.5)")
            print("  * Reducir complejidad del modelo (menos capas o filtros)")
            print("  * Aplicar más técnicas de data augmentation")
        elif is_underfit:
            print("- El modelo muestra signos de subajuste (underfitting):")
            print("  * Aumentar complejidad del modelo (más capas o filtros)")
            print("  * Disminuir regularización (reducir dropout a 0.1)")
            print("  * Entrenar por más épocas o con learning rate más alto inicialmente")
        
        # Sugerencias generales sobre hiperparámetros
        print("\nAjustes recomendados en hiperparámetros:")
        if accuracy < 0.6:
            print("- Learning rate: Probar con un rango de valores (0.001 - 0.0001)")
            print("- Arquitectura: Considerar modelos más simples o más complejos")
            print("- Preprocesamiento: Usar técnicas adicionales como filtrado de frecuencias")
        else:
            print("- Fine-tuning: Pequeños ajustes en la arquitectura actual")
            print("- Regularización: Ajustar batch normalization y weight decay")
            print("- Augmentación: Técnicas más sofisticadas de data augmentation")
    else:
        print("\nLa accuracy es BUENA, se puede considerar:")
        print("- Guardar el modelo para uso en producción")
        print("- Probar en datos completamente nuevos para verificar generalización")
        print("- Implementar técnicas de interpretabilidad para entender qué patrones ha aprendido")
    
    # Visualizar curvas de pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Pérdida de entrenamiento')
    plt.plot(val_losses, label='Pérdida de validación')
    plt.title('Curvas de Pérdida durante Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig(GRAPHS_DIR / "loss_curves.png")
    plt.close()
    
    print(f"\nLas gráficas se han guardado en la carpeta: {GRAPHS_DIR}")

def main():
    """
    Función principal que ejecuta todo el proceso.
    """
    print("Cargando y preprocesando datos...")
    try:
        X, y, class_names = load_and_preprocess_data()
        
        # Verificamos que tenemos suficientes datos
        if len(X) < 10:
            print("ERROR: No hay suficientes datos para entrenar el modelo.")
            sys.exit(1)
            
        # Data augmentation - generar más muestras para balancear el conjunto
        X_augmented = []
        y_augmented = []
        
        # Contador de clases
        class_counts = np.bincount(y)
        print(f"Distribución original de clases: {class_counts}")
        
        # Determinar la clase con más ejemplos
        max_samples = max(class_counts)
        target_samples = max(max_samples, 50)  # Aumentar a 50 muestras mínimo por clase
        
        # Augmentar datos para cada clase
        for class_idx in range(len(class_names)):
            # Índices de las muestras de esta clase
            indices = np.where(y == class_idx)[0]
            
            # Añadir las muestras originales
            for idx in indices:
                X_augmented.append(X[idx])
                y_augmented.append(y[idx])
            
            # Si hay pocas muestras, aumentarlas
            if len(indices) < target_samples:
                # Calcular cuántas muestras adicionales necesitamos
                samples_to_generate = target_samples - len(indices)
                print(f"Generando {samples_to_generate} muestras adicionales para clase {class_names[class_idx]}")
                
                # Generar muestras adicionales (con técnicas avanzadas)
                for _ in range(samples_to_generate):
                    # Elegir aleatoriamente una muestra de esta clase
                    sample_idx = np.random.choice(indices)
                    sample = X[sample_idx].copy()
                    
                    # Técnica de augmentación: mezcla de ruido y desplazamiento temporal
                    # Añadir ruido gaussiano
                    noise_level = np.random.uniform(0.02, 0.06)  # Reducir nivel de ruido
                    noise = np.random.normal(0, noise_level * np.std(sample), sample.shape)
                    
                    # Desplazamiento temporal aleatorio (shift)
                    shift_amount = np.random.randint(-2, 3)  # Desplazar -2 a 2 posiciones
                    if shift_amount != 0:
                        shifted_sample = np.zeros_like(sample)
                        if shift_amount > 0:
                            shifted_sample[:, shift_amount:] = sample[:, :-shift_amount]
                        else:
                            shifted_sample[:, :shift_amount] = sample[:, -shift_amount:]
                        sample = shifted_sample
                    
                    # Aplicar ruido a la muestra desplazada
                    augmented_sample = sample + noise
                    
                    X_augmented.append(augmented_sample)
                    y_augmented.append(class_idx)
        
        # Convertir a arrays numpy
        X = np.array(X_augmented)
        y = np.array(y_augmented)
        
        print(f"Distribución después de augmentation: {np.bincount(y)}")
        print(f"Forma de X después de augmentation: {X.shape}")
        
        # Aplicar Feature Scaling a nivel global - más efectivo
        print("Aplicando normalización avanzada...")
        # Reshape para normalizar a nivel de características entre todas las muestras
        n_samples, n_features, seq_len = X.shape
        X_reshaped = X.reshape(n_samples, n_features * seq_len)
        
        # StandardScaler para normalizar
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_reshaped)
        
        # Volver a la forma original
        X = X_normalized.reshape(n_samples, n_features, seq_len)
        
        # Dividir datos en conjuntos de entrenamiento, validación y prueba - más datos para entrenamiento
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val)
        
        print(f"Forma de los datos de entrenamiento: {X_train.shape}")
        print(f"Forma de los datos de validación: {X_val.shape}")
        print(f"Forma de los datos de prueba: {X_test.shape}")
        
        # Crear datasets y dataloaders
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)
        
        batch_size = 8  # Batch más pequeño para mejor aprendizaje
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Determinar dispositivo (GPU si está disponible)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Utilizando dispositivo: {device}")
        
        # Crear modelo
        n_features, seq_len = X_train.shape[1], X_train.shape[2]
        num_classes = len(class_names)
        
        model = EEGCNN(n_features, num_classes).to(device)
        print("Arquitectura del modelo:")
        print(model)
        
        print("\nIniciando entrenamiento...")
        num_epochs = 300
        learning_rate = 0.0005
        train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, device, 
            num_epochs=num_epochs, learning_rate=learning_rate
        )
        
        print("\nEvaluando modelo en conjunto de prueba...")
        accuracy, _ = evaluate_model(model, test_loader, device, class_names)
        
        analyze_performance_and_suggest_improvements(accuracy, train_losses, val_losses)
        
        # Guardar el modelo si la precisión es buena
        if accuracy >= 0.8:
            model_save_path = GRAPHS_DIR / "eeg_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'class_names': class_names,
                'n_features': n_features,
                'seq_len': seq_len
            }, model_save_path)
            print(f"\nModelo guardado en: {model_save_path}")
        
        print("\nProceso completo. Las gráficas han sido guardadas en la carpeta:")
        print(f"{GRAPHS_DIR}")
        
    except Exception as e:
        print(f"ERROR GENERAL: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()