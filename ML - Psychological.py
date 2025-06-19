import torch
import torch.nn.functional as F
import numpy as np
import pickle  # Para cargar el LabelEncoder guardado

# === 1. Definir la arquitectura del modelo (MLP) ===
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Capa oculta con ReLU
        x = self.fc2(x)  # Capa de salida
        return x

# === 2. Cargar el modelo guardado ===
checkpoint = torch.load('ML-Psychological/model_psychological.pth')

# Crear una nueva instancia del modelo MLP
model = MLP(in_channels=6, hidden_channels=16, out_channels=3)  # 6 entradas, 3 salidas (facultades)

# Cargar el estado del modelo
model.load_state_dict(checkpoint['model_state_dict'])

# Colocar el modelo en modo de evaluación (predicción)
model.eval()

# === 3. Cargar el LabelEncoder guardado ===
with open('ML-Psychological/label_encoder_psychological.pkl', 'rb') as f:
    le = pickle.load(f)

# === 4. Realizar predicciones para un nuevo estudiante ===
print("\nIntroduce las características del nuevo estudiante (Realista, Investigador, Artistico, Social, Emprendedor, Convencional) separados por coma:")
new_student_input = input()  # Entrada del nuevo estudiante

# Convertir la entrada en un array de valores numéricos
new_student_data = np.array([float(i) for i in new_student_input.split(',')]).reshape(1, -1)

# Preparar los datos del nuevo estudiante para PyTorch
x_new = torch.tensor(new_student_data, dtype=torch.float)

# Realizar la predicción para el nuevo estudiante
out_new = model(x_new)  # Pasamos los datos de entrada por la red
pred_new = out_new.argmax(dim=1)  # Predicción: la facultad con la mayor probabilidad

# Mostrar la facultad recomendada
facultades = le.classes_  # Facultades disponibles (debe ser el mismo `LabelEncoder` que usaste durante el entrenamiento)
facultad_recomendada = facultades[pred_new.item()]
print(f"La facultad recomendada para el nuevo estudiante es: {facultad_recomendada}")
