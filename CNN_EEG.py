import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import joblib

# Configuración
DATA_DIR = 'OrganizedData'
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
USE_WINDOW = True
WINDOW_SIZE = 10  # Cantidad de muestras consecutivas a considerar como una ventana

def load_data(data_dir):
    """
    Carga los datos de EEG de todas las subcarpetas en data_dir.
    
    Args:
        data_dir: Directorio que contiene las subcarpetas con archivos CSV
        
    Returns:
        X_data: Array con los datos de EEG
        y_labels: Array con las etiquetas (nombres de carpetas)
        class_names: Lista de nombres de clases únicas
    """
    print(f"Cargando datos desde: {data_dir}")
    
    all_data = []
    all_labels = []
    
    # Obtener todas las carpetas (clases)
    class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Clases encontradas: {class_folders}")
    
    total_files = 0
    
    # Para cada carpeta (clase)
    for class_folder in class_folders:
        class_path = os.path.join(data_dir, class_folder)
        print(f"Procesando clase: {class_folder}")
        
        # Para cada archivo CSV en la carpeta
        csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
        
        # Contador de archivos para esta clase
        files_processed = 0
        
        for csv_file in csv_files:
            file_path = os.path.join(class_path, csv_file)
            
            try:
                # Leer el CSV, saltando líneas de comentario que empiezan con #
                df = pd.read_csv(file_path, comment='#')
                
                # Extraer columnas de interés
                feature_cols = ['Canal4_Alpha', 'Canal4_Beta', 'Canal5_Alpha', 'Canal5_Beta']
                
                # Verificar que todas las columnas existan
                missing_cols = [col for col in feature_cols if col not in df.columns]
                if missing_cols:
                    print(f"  Advertencia: Columnas faltantes en {csv_file}: {missing_cols}")
                    continue
                
                # Extraer solo las columnas de interés
                features = df[feature_cols].values
                
                if USE_WINDOW:
                    # Crear ventanas deslizantes
                    for i in range(0, len(features) - WINDOW_SIZE + 1, WINDOW_SIZE // 2):  # 50% de solapamiento
                        window = features[i:i+WINDOW_SIZE]
                        all_data.append(window)
                        all_labels.append(class_folder)
                else:
                    # Usar cada fila como una muestra individual
                    for row in features:
                        all_data.append(row)
                        all_labels.append(class_folder)
                
                files_processed += 1
                
            except Exception as e:
                print(f"  Error al procesar {file_path}: {str(e)}")
        
        total_files += files_processed
        print(f"  Procesados {files_processed} archivos para la clase {class_folder}")
    
    print(f"Total de archivos procesados: {total_files}")
    
    if not all_data:
        raise ValueError("No se pudieron cargar datos. Verifica la estructura de carpetas y archivos.")
    
    return np.array(all_data), np.array(all_labels), sorted(list(set(all_labels)))

def preprocess_data(X, y):
    """
    Preprocesa los datos: codifica etiquetas y normaliza valores.
    
    Args:
        X: Datos de entrada
        y: Etiquetas
        
    Returns:
        X_normalized: Datos normalizados
        y_encoded: Etiquetas codificadas en one-hot
        label_encoder: Codificador de etiquetas
        scaler: Normalizador
    """
    print("Preprocesando datos...")
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_numeric = label_encoder.fit_transform(y)
    
    # Normalizar datos
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    
    scaler = StandardScaler()
    X_normalized_flat = scaler.fit_transform(X_reshaped)
    
    # Restaurar forma original
    X_normalized = X_normalized_flat.reshape(original_shape)
    
    print(f"Datos preprocesados - Forma: {X_normalized.shape}, Clases: {len(label_encoder.classes_)}")
    
    return X_normalized, y_numeric, label_encoder, scaler

def build_model(input_shape, num_classes):
    """
    Construye el modelo CNN para clasificación de EEG.
    
    Args:
        input_shape: Forma de los datos de entrada
        num_classes: Número de clases a predecir
        
    Returns:
        model: Modelo compilado
    """
    print(f"Construyendo modelo CNN - Input shape: {input_shape}, Clases: {num_classes}")
    
    model = Sequential()
    
    # Primera capa convolucional
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', 
                     input_shape=input_shape, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Segunda capa convolucional
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Aplanar para capas densas
    model.add(Flatten())
    
    # Capa densa
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Mostrar resumen
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Entrena el modelo con los datos proporcionados.
    
    Args:
        model: Modelo a entrenar
        X_train, y_train: Datos y etiquetas de entrenamiento
        X_val, y_val: Datos y etiquetas de validación
        
    Returns:
        history: Historial de entrenamiento
    """
    print("Entrenando modelo...")
    
    # Configurar callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_eeg_model.h5', monitor='val_accuracy', 
                        save_best_only=True, verbose=1)
    ]
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evalúa el modelo y muestra métricas de rendimiento.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba (one-hot)
        class_names: Nombres de las clases
    """
    print("Evaluando modelo...")
    
    # Evaluar con el conjunto de prueba
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Precisión en conjunto de prueba: {accuracy*100:.2f}%')
    
    # Obtener predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Reporte de clasificación
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print("\nReporte de Clasificación:")
    print(report)
    
    # Guardar reporte en archivo
    with open('classification_report.txt', 'w') as f:
        f.write(report)

def plot_training_history(history):
    """
    Visualiza el historial de entrenamiento.
    
    Args:
        history: Historial del entrenamiento
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfica de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    """Función principal que ejecuta todo el proceso"""
    
    # Establecer semillas para reproducibilidad
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    try:
        # 1. Cargar datos
        X, y, class_names = load_data(DATA_DIR)
        
        # 2. Preprocesar datos
        X_processed, y_numeric, label_encoder, scaler = preprocess_data(X, y)
        
        # 3. Dividir datos
        # Primero separar conjunto de prueba
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_processed, y_numeric, 
            test_size=TEST_SPLIT, 
            random_state=RANDOM_SEED,
            stratify=y_numeric
        )
        
        # Luego separar conjunto de validación
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_SEED,
            stratify=y_temp
        )
        
        # Convertir etiquetas a one-hot
        num_classes = len(class_names)
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        print(f"Conjuntos de datos - Entrenamiento: {X_train.shape}, Validación: {X_val.shape}, Prueba: {X_test.shape}")
        
        # 4. Construir modelo
        if USE_WINDOW:
            # Si usamos ventanas, la forma es [muestras, ventana, características]
            input_shape = (X_train.shape[1], X_train.shape[2])
        else:
            # Si no usamos ventanas, la forma es [muestras, características, 1]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            input_shape = (X_train.shape[1], 1)
        
        model = build_model(input_shape, num_classes)
        
        # 5. Entrenar modelo
        history = train_model(model, X_train, y_train_cat, X_val, y_val_cat)
        
        # 6. Evaluar modelo
        evaluate_model(model, X_test, y_test_cat, class_names)
        
        # 7. Visualizar resultados
        plot_training_history(history)
        
        # 8. Guardar modelo y preprocesadores
        model.save('eeg_cnn_model.h5')
        joblib.dump(scaler, 'eeg_scaler.pkl')
        joblib.dump(label_encoder, 'eeg_label_encoder.pkl')
        
        print("\nProceso completado con éxito.")
        print("Archivos guardados:")
        print(" - eeg_cnn_model.h5: Modelo entrenado")
        print(" - eeg_scaler.pkl: Normalizador")
        print(" - eeg_label_encoder.pkl: Codificador de etiquetas")
        print(" - confusion_matrix.png: Matriz de confusión")
        print(" - training_history.png: Gráficas de entrenamiento")
        print(" - classification_report.txt: Reporte detallado")
        
    except Exception as e:
        print(f"Error durante el proceso: {str(e)}")

if __name__ == "__main__":
    main()