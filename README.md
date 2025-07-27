# API de Predicción de Defunción por Dengue

Este proyecto utiliza un modelo de Machine Learning (Random Forest) para predecir si un paciente con diagnóstico de dengue tiene riesgo de fallecer, basado en variables clínicas y demográficas.

## Estructura

- `app.py`: Backend con Flask que expone una API `/predict`
- `modelo/modelo_rf.pkl`: Modelo entrenado
- `uploads/`: Carpeta temporal para archivos subidos
- `requirements.txt`: Librerías necesarias

## Instrucciones

1. Instala dependencias:
pip install -r requirements.txt

2. Ejecuta la app:
python app.py

3. Envia un archivo `.csv` a `http://localhost:5000/predict` con el campo `file`.

## Autores

- Marco Rocha & equipo
