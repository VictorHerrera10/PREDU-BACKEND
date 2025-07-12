from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import pickle
import torch.nn.functional as F
from starlette.middleware.cors import CORSMiddleware
from torch_geometric.nn import SAGEConv
from fastapi.middleware.cors import CORSMiddleware


# === 1. Definir los modelos ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelo GraphSAGE para el análisis académico
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=8):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))  # Agregar activación
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Cargar el modelo GraphSAGE
model_graphsage = GraphSAGE(in_channels=11, out_channels=3)  # Asumimos que el modelo tiene 11 entradas y 3 salidas
model_state = torch.load('ML-Academic/model_academic.pth', map_location=device)
model_graphsage.load_state_dict(model_state['model_state_dict'])
model_graphsage.to(device)
model_graphsage.eval()

# Cargar el LabelEncoder (career_map) para el modelo académico
with open('ML-Academic/label_encoder_academic.pkl', 'rb') as f:
    career_map = pickle.load(f)

career_names = {v: k for k, v in career_map.items()}

# Modelo MLP para el análisis psicológico
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Capa oculta con ReLU
        x = self.fc2(x)  # Capa de salida
        return x

# Cargar el modelo MLP psicológico
model_psychological = MLP(in_channels=6, hidden_channels=16, out_channels=3)
checkpoint = torch.load('ML-Psychological/model_psychological.pth')
model_psychological.load_state_dict(checkpoint['model_state_dict'])
model_psychological.eval()

# Cargar el LabelEncoder para el modelo psicológico
with open('ML-Psychological/label_encoder_psychological.pkl', 'rb') as f:
    le_psychological = pickle.load(f)

# === 2. Crear la aplicación FastAPI ===
app = FastAPI()


origins = [
    "*",  # Permitir todas las orígenes
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Pydantic Models for Input Data
class EstudianteInput(BaseModel):
    arte_y_cultura: str
    castellano_como_segunda_lengua: str
    ciencia_y_tecnologia: str
    ciencias_sociales: str
    comunicacion: str
    desarrollo_personal: str
    educacion_fisica: str
    educacion_para_el_trabajo: str
    educacion_religiosa: str
    ingles: str
    matematica: str

class PsicoInput(BaseModel):
    realista: float
    investigador: float
    artistico: float
    social: float
    emprendedor: float
    convencional: float

# === 3. Funciones de Predicción ===

# Predicción del modelo GraphSAGE (Académico)
def predecir_carrera_graphsage(model, notas_nuevo_estudiante):
    calificacion_map = {'AD': 4, 'A': 3, 'B': 2, 'C': 1}
    notas_nuevo_estudiante = [calificacion_map.get(nota, 0) for nota in notas_nuevo_estudiante]
    notas_nuevo_estudiante = np.array(notas_nuevo_estudiante).reshape(1, -1)
    notas_tensor = torch.tensor(notas_nuevo_estudiante, dtype=torch.float).to(device)
    edge_index_empty = torch.tensor([[], []], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        out = model(notas_tensor, edge_index_empty)
        pred = out.argmax(dim=1)

    carrera_predicha = career_names[pred.item()]
    return carrera_predicha

# Predicción del modelo MLP (Psicológico)
def predecir_facultad_psychological(model, datos_psicologicos):
    x_new = torch.tensor(datos_psicologicos, dtype=torch.float)
    out_new = model(x_new)

    # Verificar la forma de la salida
    print(f"Salida del modelo psicológico: {out_new.shape}")

    # Asegurarse de que la salida sea del tipo correcto
    if len(out_new.shape) == 2 and out_new.shape[1] > 1:
        pred_new = out_new.argmax(dim=1)  # Predicción: la facultad con la mayor probabilidad
    else:
        pred_new = out_new.argmax(dim=0)  # Si la salida es unidimensional

    facultades = le_psychological.classes_
    facultad_recomendada = facultades[pred_new.item()]
    return facultad_recomendada


# === 4. Endpoints ===

@app.post("/prediccion/academico/")
async def prediccion_academica(data: EstudianteInput):
    new_student_data = [
        data.arte_y_cultura, data.castellano_como_segunda_lengua, data.ciencia_y_tecnologia,
        data.ciencias_sociales, data.comunicacion, data.desarrollo_personal, data.educacion_fisica,
        data.educacion_para_el_trabajo, data.educacion_religiosa, data.ingles, data.matematica
    ]
    carrera_predicha = predecir_carrera_graphsage(model_graphsage, new_student_data)
    return {"carrera_predicha": carrera_predicha}

@app.post("/prediccion/psicologica/")
async def prediccion_psicologica(data: PsicoInput):
    new_student_psychological = [
        data.realista, data.investigador, data.artistico, data.social, data.emprendedor, data.convencional
    ]
    facultad_predicha = predecir_facultad_psychological(model_psychological, new_student_psychological)
    return {"facultad_predicha": facultad_predicha}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)