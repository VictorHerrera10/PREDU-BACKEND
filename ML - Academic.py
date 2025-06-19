import torch
import numpy as np
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# === 1. Cargar el modelo y el LabelEncoder ===
# Cargar el modelo GraphSAGE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.nn.Module()  # Placeholder for the model

# Definir el modelo GraphSAGE igual que se hizo antes
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=8):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))  # Agregar activación
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Cargar el modelo guardado
model_state = torch.load('ML-Academic/model_academic.pth', map_location=device)
model = GraphSAGE(in_channels=11, out_channels=3)  # Asumimos que el modelo tiene 11 entradas y 3 salidas
model.load_state_dict(model_state['model_state_dict'])
model.to(device)
model.eval()

# Cargar el LabelEncoder (career_map)
with open('ML-Academic/label_encoder_academic.pkl', 'rb') as f:
    career_map = pickle.load(f)

career_names = {v: k for k, v in career_map.items()}

# === 2. Función para predecir la carrera con el modelo GraphSAGE ===
def predecir_carrera_graphsage(model, notas_nuevo_estudiante):
    calificacion_map = {'AD': 4, 'A': 3, 'B': 2, 'C': 1}
    notas_nuevo_estudiante = [calificacion_map.get(nota, 0) for nota in notas_nuevo_estudiante]  # Mapear notas
    notas_nuevo_estudiante = np.array(notas_nuevo_estudiante).reshape(1, -1)  # Convertir a array de una fila

    notas_tensor = torch.tensor(notas_nuevo_estudiante, dtype=torch.float).to(device)

    # Preparar un "edge_index" vacío porque no lo necesitamos para predicciones individuales
    edge_index_empty = torch.tensor([[], []], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        out = model(notas_tensor, edge_index_empty)
        pred = out.argmax(dim=1)

    carrera_predicha = career_names[pred.item()]
    return carrera_predicha

# === 3. Ingresar las calificaciones del estudiante y predecir la carrera ===
print("\nIntroduce las calificaciones del nuevo estudiante para los cursos (ARTE Y CULTURA, CASTELLANO COMO SEGUNDA LENGUA, CIENCIA Y TECNOLOGIA, CIENCIAS SOCIALES, COMUNICACION, DESARROLLO PERSONAL, CIUDADANIA Y CIVICA, EDUCACION FISICA, EDUCACION PARA EL TRABAJO, EDUCACION RELIGIOSA, INGLES, MATEMATICA) separados por coma (ejemplo 'AD', 'A', 'B', 'C'):")

# Esperar la entrada del usuario
new_student_input = input()  # Entrada del nuevo estudiante (como 'A', 'B', 'AD', 'C', etc.)

# Convertir la entrada en un array de valores numéricos
new_student_data = [i for i in new_student_input.split(',')]  # Ahora se ingresan como 'AD', 'A', 'B', 'C'

# Predicción de la carrera
carrera_predicha_graphsage = predecir_carrera_graphsage(model, new_student_data)
print(f"La carrera predicha para el nuevo estudiante con GraphSAGE es: {carrera_predicha_graphsage}")
