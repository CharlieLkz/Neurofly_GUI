import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Definición del modelo CNN
class EEG_CNN(nn.Module):
    def __init__(self, input_channels, num_classes, window_size=10):
        super(EEG_CNN, self).__init__()
        
        # Primera capa convolucional
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Segunda capa convolucional
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calcular el tamaño de salida después de las capas convolucionales
        # Después de dos MaxPool con kernel_size=2, la dimensión se reduce a 1/4
        conv_output_size = (window_size // 4) * 128
        if conv_output_size == 0:  # En caso de que window_size//4 sea 0
            conv_output_size = 128
        
        # Capas densas
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

def load_model(model_path, model_info_path, num_classes):
    """
    Carga un modelo guardado desde un archivo .pth
    
    Args:
        model_path: Ruta al archivo del modelo (.pth)
        model_info_path: Ruta al archivo de información del modelo (.json)
        num_classes: Número de clases
        
    Returns:
        model: Modelo cargado
    """
    # Cargar información del modelo
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    input_channels = model_info.get('input_channels', 4)
    window_size = model_info.get('window_size', 10)
    
    # Crear modelo con la misma arquitectura
    model = EEG_CNN(input_channels, num_classes, window_size)
    
    # Cargar pesos del modelo
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Modo evaluación
    model.eval()
    
    return model, model_info

def predict_from_csv(model_path, model_info_path, scaler_path, encoder_path, csv_path):
    """
    Realiza predicciones sobre un archivo CSV de señales EEG.
    
    Args:
        model_path: Ruta al modelo guardado (.pth)
        model_info_path: Ruta al archivo de información del modelo (.json)
        scaler_path: Ruta al escalador guardado (.pkl)
        encoder_path: Ruta al codificador de etiquetas (.pkl)
        csv_path: Ruta al archivo CSV a predecir
        
    Returns:
        predicted_label: Etiqueta más probable para el archivo
        probabilities: Probabilidades para todas las clases
    """
    print(f"Realizando predicciones para: {csv_path}")
    
    # Cargar preprocesadores
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    num_classes = len(label_encoder.classes_)
    
    # Cargar modelo
    model, model_info = load_model(model_path, model_info_path, num_classes)
    
    # Extraer configuración
    use_window = model_info.get('use_window', True)
    window_size = model_info.get('window_size', 10)
    
    try:
        # Leer archivo CSV
        df = pd.read_csv(csv_path, comment='#')
        
        # Extraer características
        feature_cols = ['Canal4_Alpha', 'Canal4_Beta', 'Canal5_Alpha', 'Canal5_Beta']
        
        # Verificar que las columnas existan
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en el CSV: {missing_cols}")
        
        # Extraer los datos
        features = df[feature_cols].values
        
        if use_window:
            # Crear ventanas
            windows = []
            stride = window_size // 2  # 50% de solapamiento
            
            for i in range(0, len(features) - window_size + 1, stride):
                windows.append(features[i:i+window_size])
            
            if not windows:
                raise ValueError(f"No hay suficientes datos para crear ventanas (se necesitan al menos {window_size} muestras)")
            
            # Convertir a array numpy
            X = np.array(windows)
            
            # Normalizar datos
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_normalized_flat = scaler.transform(X_reshaped)
            X_normalized = X_normalized_flat.reshape(X.shape)
            
            # Transponer para formato PyTorch: [batch, channels, sequence]
            X_normalized = np.transpose(X_normalized, (0, 2, 1))
            
        else:
            # Usar cada fila como una muestra individual
            X = features
            
            # Normalizar
            X_normalized = scaler.transform(X)
            
            # Reshape para CNN (agregar dimensión de canal)
            X_normalized = X_normalized.reshape(X_normalized.shape[0], 1, -1)
        
        # Convertir a tensor de PyTorch
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        
        # Realizar predicciones
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Promediar predicciones de todas las ventanas/muestras
        avg_probs = torch.mean(probs, dim=0).numpy()
        
        # Obtener la clase más probable
        predicted_class = np.argmax(avg_probs)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        # Obtener todas las clases y probabilidades
        all_classes = label_encoder.classes_
        class_probabilities = {cls: prob for cls, prob in zip(all_classes, avg_probs)}
        
        # Ordenar por probabilidad
        sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        print("\nResultados de la predicción:")
        print(f"Etiqueta más probable: {predicted_label} con {avg_probs[predicted_class]*100:.2f}% de confianza")
        print("\nProbabilidades para todas las clases:")
        for cls, prob in sorted_probs:
            print(f"  {cls}: {prob*100:.2f}%")
        
        return predicted_label, class_probabilities
        
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Predecir clase de EEG desde archivo CSV')
    parser.add_argument('--model', default='eeg_cnn_model.pth', help='Ruta al modelo guardado')
    parser.add_argument('--model-info', default='model_info.json', help='Ruta al archivo de información del modelo')
    parser.add_argument('--scaler', default='eeg_scaler.pkl', help='Ruta al escalador guardado')
    parser.add_argument('--encoder', default='eeg_label_encoder.pkl', help='Ruta al codificador de etiquetas')
    parser.add_argument('--csv', required=True, help='Ruta al archivo CSV a predecir')
    
    args = parser.parse_args()
    
    predict_from_csv(
        args.model,
        args.model_info,
        args.scaler, 
        args.encoder, 
        args.csv
    )

if __name__ == "__main__":
    main()