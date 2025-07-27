from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

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

if __name__ == '__main__':
    app.run(debug=True)
