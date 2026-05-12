
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Título de la aplicación
st.set_page_config(page_title="Predicción de Deserción Estudiantil")
st.title('📊 Predicción de Deserción Estudiantil')

st.markdown("Bienvenido a la aplicación de predicción de deserción. Ingresa los datos del estudiante para obtener una predicción.")

# Cargar el modelo entrenado
model_filename = 'modelo_desercion.pkl'
try:
    model = joblib.load(model_filename)
    st.success(f"Modelo '{model_filename}' cargado exitosamente.")
except FileNotFoundError:
    st.error(f"Error: El archivo del modelo '{model_filename}' no se encontró. Asegúrate de que el modelo esté guardado en el directorio correcto.")
    st.stop() # Detener la ejecución si el modelo no está disponible

# --- Widgets de entrada para las características ---
st.header('Datos del Estudiante')

# Edad (age): Normal distribution, clipped to a realistic range for university students
edad = st.slider('Edad', min_value=18, max_value=35, value=20, step=1)

# Promedio (GPA/average): Normal distribution, 0-5 scale
promedio = st.slider('Promedio (GPA)', min_value=1.0, max_value=5.0, value=3.2, step=0.1)

# Asistencia (attendance): Percentage, 0-100
asistencia = st.slider('Asistencia (%)', min_value=50, max_value=100, value=85, step=1)

# Horas de estudio (study hours per week)
horas_estudio = st.slider('Horas de Estudio por Semana', min_value=0, max_value=40, value=15, step=1)

# Uso de plataforma (platform usage hours per week)
uso_plataforma = st.slider('Horas de Uso de Plataforma por Semana', min_value=0, max_value=20, value=8, step=1)

# Materias perdidas (failed courses): Poisson distribution, clipped
materias_perdidas = st.slider('Materias Perdidas', min_value=0, max_value=5, value=0, step=1)

# Nivel socioeconomico (socioeconomic level): 1-5 scale (1=low, 5=high)
nivel_socioeconomico = st.selectbox('Nivel Socioeconómico (1=Bajo, 5=Alto)', options=[1, 2, 3, 4, 5], index=2)

# Trabaja (works): Binary (0=no, 1=yes)
trabaja_map = {'No': 0, 'Sí': 1}
trabaja_selection = st.selectbox('¿El estudiante trabaja?', options=['No', 'Sí'], index=0)
trabaja = trabaja_map[trabaja_selection]

# Acceso a internet (internet access): Binary (0=no, 1=yes)
acceso_internet_map = {'No': 0, 'Sí': 1}
acceso_internet_selection = st.selectbox('¿Tiene acceso a internet?', options=['No', 'Sí'], index=1)
acceso_internet = acceso_internet_map[acceso_internet_selection]

# Crear un DataFrame con las entradas del usuario
input_data = pd.DataFrame([[edad, promedio, asistencia, horas_estudio, uso_plataforma,
                            materias_perdidas, nivel_socioeconomico, trabaja, acceso_internet]],
                          columns=['edad', 'promedio', 'asistencia', 'horas_estudio', 'uso_plataforma',
                                   'materias_perdidas', 'nivel_socioeconomico', 'trabaja', 'acceso_internet'])

# Botón para realizar la predicción
if st.button('Predecir Deserción'):
    if model is not None:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1] # Probabilidad de deserción

        st.subheader('Resultado de la Predicción')
        if prediction[0] == 1:
            st.error(f"¡ALERTA! El estudiante tiene ALTA probabilidad de desertar. (Probabilidad: {prediction_proba[0]*100:.2f}%) 🔴")
            st.markdown("Se recomienda una intervención temprana para apoyar al estudiante.")
        else:
            st.success(f"El estudiante tiene BAJA probabilidad de desertar. (Probabilidad: {prediction_proba[0]*100:.2f}%) 🟢")
            st.markdown("El estudiante parece estar en buen camino.")
    else:
        st.warning("El modelo no está disponible para hacer predicciones. Por favor, asegúrate de que se haya cargado correctamente.")

st.markdown("""
---
*Nota: Esta aplicación es un prototipo y la predicción se basa en un modelo entrenado con datos sintéticos. Siempre considera el contexto real y otros factores."
""")
