from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar modelo
modelo = joblib.load('models/modelo_rf.pkl')

@app.route('/')
def home():
    return "API de predicción de defunción por dengue - Activa"

@app.route('/predict', methods=['POST'])
def predict():
    print("== REQUEST DEBUG ==")
    print("request.method:", request.method)
    print("request.content_type:", request.content_type)
    print("request.files:", request.files)
    print("request.form:", request.form)

    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo CSV'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Leer CSV
    df = pd.read_csv(filepath)

    # Preparar datos: seleccionar columnas y transformar categóricas
    features = ['EDAD_ANOS', 'SEXO', 'TIPO_PACIENTE', 'DICTAMEN',
                'DIABETES', 'HIPERTENSION', 'EMBARAZO', 'INMUNOSUPR']
    df = df[features]
    df = pd.get_dummies(df, columns=['SEXO', 'TIPO_PACIENTE', 'DICTAMEN'], drop_first=True)

    # Predecir
    predicciones = modelo.predict(df)

    return jsonify({'predicciones': predicciones.tolist()})

@app.route('/regresion-anual', methods=['GET'])
def regresion_anual():
    # Ruta del archivo CSV base (puedes ajustar)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dengue_combinado_filtrado_ordenado.csv')

    if not os.path.exists(filepath):
        return jsonify({'error': 'Archivo CSV no encontrado en /uploads'}), 404

    try:
        columnas = ['FECHA_SIGN_SINTOMAS', 'DEFUNCION']
        df = pd.read_csv(filepath, usecols=columnas)
        df['AÑO'] = pd.to_datetime(df['FECHA_SIGN_SINTOMAS'], errors='coerce').dt.year
        df = df.dropna(subset=['AÑO'])

        df_def = df[df['DEFUNCION'] == 1]
        muertes_por_año = df_def.groupby('AÑO').size()

        X = muertes_por_año.index.values.reshape(-1, 1)
        y = muertes_por_año.values

        modelo_rl = LinearRegression()
        modelo_rl.fit(X, y)
        pred = modelo_rl.predict(X)

        response = {
            'años': X.flatten().tolist(),
            'muertes': y.tolist(),
            'tendencia': np.round(pred).astype(int).tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
