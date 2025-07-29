from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import joblib
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from unidecode import unidecode

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CSV_PATH = os.path.join(UPLOAD_FOLDER, 'dengue_combinado_filtrado_ordenado.csv')
CATALOGO_PATH = os.path.join(UPLOAD_FOLDER, 'Catálogos_Dengue.xlsx')

# Cargar modelo de clasificación
modelo = joblib.load('models/modelo_rf.pkl')

# -------------------- FUNCIONES AUXILIARES --------------------

def cargar_datos_unidos():
    if not os.path.exists(CSV_PATH) or not os.path.exists(CATALOGO_PATH):
        return None, "Archivo CSV o catálogo no encontrado.", []

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.upper()

    required_cols = ['MUNICIPIO_RES', 'ENTIDAD_RES', 'DEFUNCION', 'DIABETES', 'HIPERTENSION', 'EMBARAZO', 'INMUNOSUPR', 'FECHA_SIGN_SINTOMAS']
    for col in required_cols:
        if col not in df.columns:
            return None, f"Falta la columna '{col}'", []

    df['MUNICIPIO_RES'] = df['MUNICIPIO_RES'].astype(str).str.zfill(3)
    df['ENTIDAD_RES'] = df['ENTIDAD_RES'].astype(str).str.zfill(2)
    df['DEFUNCION'] = df['DEFUNCION'].astype(int)

    cat_mun = pd.read_excel(CATALOGO_PATH, sheet_name='CATÁLOGO MUNICIPIO')
    cat_mun.columns = cat_mun.columns.str.upper()
    cat_mun['CLAVE_MUNICIPIO'] = cat_mun['CLAVE_MUNICIPIO'].astype(str).str.zfill(3)
    cat_mun['CLAVE_ENTIDAD'] = cat_mun['CLAVE_ENTIDAD'].astype(str).str.zfill(2)

    cat_ent = pd.read_excel(CATALOGO_PATH, sheet_name='CATÁLOGO ENTIDAD')
    cat_ent.columns = cat_ent.columns.str.upper()
    cat_ent['CLAVE_ENTIDAD'] = cat_ent['CLAVE_ENTIDAD'].astype(str).str.zfill(2)
    cat_ent.rename(columns={'ENTIDAD_FEDERATIVA': 'NOMBRE_ENTIDAD'}, inplace=True)

    df = df.merge(
        cat_mun[['CLAVE_ENTIDAD', 'CLAVE_MUNICIPIO', 'MUNICIPIO']],
        left_on=['ENTIDAD_RES', 'MUNICIPIO_RES'],
        right_on=['CLAVE_ENTIDAD', 'CLAVE_MUNICIPIO'],
        how='left'
    )

    df = df.merge(
        cat_ent[['CLAVE_ENTIDAD', 'NOMBRE_ENTIDAD']],
        left_on='ENTIDAD_RES',
        right_on='CLAVE_ENTIDAD',
        how='left'
    )

    errores = df[df['MUNICIPIO'].isnull()][['ENTIDAD_RES', 'MUNICIPIO_RES']].drop_duplicates().to_dict(orient='records')
    df = df[df['MUNICIPIO'].notnull() & df['NOMBRE_ENTIDAD'].notnull()]

    return df, None, errores

def generar_resumen_municipios(filtro_entidad=None):
    df, error, errores = cargar_datos_unidos()
    if error:
        return {"error": error}, 500

    if filtro_entidad:
        df = df[unidecode(df['NOMBRE_ENTIDAD'].str.lower()) == unidecode(filtro_entidad.strip().lower())]
        if df.empty:
            return {"error": f"No se encontraron datos para la entidad '{filtro_entidad}'"}, 404

    resumen = (
        df.groupby(['NOMBRE_ENTIDAD', 'MUNICIPIO'])
        .agg(
            total_casos=('DEFUNCION', 'count'),
            muertes=('DEFUNCION', lambda x: (x == 1).sum()),
            diabetes=('DIABETES', 'sum'),
            hipertension=('HIPERTENSION', 'sum'),
            embarazo=('EMBARAZO', 'sum'),
            inmunosupresion=('INMUNOSUPR', 'sum')
        )
        .reset_index()
        .sort_values(by='muertes', ascending=False)
    )

    resultado = [
        {
            "entidad": row['NOMBRE_ENTIDAD'],
            "municipio": row['MUNICIPIO'],
            "total_casos": int(row['total_casos']),
            "muertes": int(row['muertes']),
            "diabetes": int(row['diabetes']),
            "hipertension": int(row['hipertension']),
            "embarazo": int(row['embarazo']),
            "inmunosupresion": int(row['inmunosupresion']),
        }
        for _, row in resumen.iterrows()
    ]

    return {"municipios": resultado, "registros_no_encontrados": errores}, 200

# -------------------- ENDPOINTS --------------------

@app.route('/')
def home():
    return "API de predicción de defunción por dengue - Activa"

<<<<<<< HEAD
@app.route('/api/estadisticas-generales', methods=['GET'])
def estadisticas_generales():
    df, error, _ = cargar_datos_unidos()
    if error:
        return jsonify({"error": error}), 500

    try:
        total_casos = len(df)
        defunciones = df['DEFUNCION'].sum()
        edad_promedio = int(round(df['EDAD_ANOS'].mean(), 0))
        entidad_mas_casos = (
            df.groupby('NOMBRE_ENTIDAD').size().sort_values(ascending=False).idxmax()
        )

        return jsonify({
            "casos_totales": total_casos,
            "defunciones": int(defunciones),
            "edad_promedio": edad_promedio,
            "entidad_mas_casos": entidad_mas_casos
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict_desde_uploads():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dengue_combinado_filtrado_ordenado.csv')

    if not os.path.exists(filepath):
        return jsonify({'error': 'Archivo CSV no encontrado en uploads'}), 404

    try:
        df = pd.read_csv(filepath)

        features = ['EDAD_ANOS', 'SEXO', 'TIPO_PACIENTE', 'DICTAMEN',
                    'DIABETES', 'HIPERTENSION', 'EMBARAZO', 'INMUNOSUPR']

=======
@app.route('/predict', methods=['GET'])
def predict_desde_uploads():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dengue_combinado_filtrado_ordenado.csv')

    if not os.path.exists(filepath):
        return jsonify({'error': 'Archivo CSV no encontrado en uploads'}), 404

    try:
        df = pd.read_csv(filepath)

        features = ['EDAD_ANOS', 'SEXO', 'TIPO_PACIENTE', 'DICTAMEN',
                    'DIABETES', 'HIPERTENSION', 'EMBARAZO', 'INMUNOSUPR']

>>>>>>> 675b1be3c9b7dad1936ae35ea2c4e3c826bbb2c6
        df = df[features]
        df = pd.get_dummies(df, columns=['SEXO', 'TIPO_PACIENTE', 'DICTAMEN'], drop_first=True)

        predicciones = modelo.predict(df)
        return jsonify({'predicciones': predicciones.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/regresion-anual', methods=['GET'])
def regresion_anual():
    if not os.path.exists(CSV_PATH):
        return jsonify({'error': 'Archivo CSV no encontrado en /uploads'}), 404

    try:
        df = pd.read_csv(CSV_PATH, usecols=['FECHA_SIGN_SINTOMAS', 'DEFUNCION'])
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

@app.route('/api/resumen', methods=['GET'])
def resumen_api():
    entidad_filtro = request.args.get('entidad')
    resultado, status = generar_resumen_municipios(entidad_filtro)
    return jsonify(resultado), status

@app.route('/api/riesgo', methods=['GET'])
def riesgo_municipios():
    df, error, _ = cargar_datos_unidos()
    if error:
        return jsonify({"error": error}), 500

    df['FECHA_SIGN_SINTOMAS'] = pd.to_datetime(df['FECHA_SIGN_SINTOMAS'], errors='coerce')
    df = df[df['FECHA_SIGN_SINTOMAS'].notna()]
    df['AÑO'] = df['FECHA_SIGN_SINTOMAS'].dt.year.astype(str)

    agrupado = df.groupby(['ENTIDAD_RES', 'MUNICIPIO_RES', 'AÑO']).agg(
        casos_totales=('DEFUNCION', 'count'),
        muertes=('DEFUNCION', lambda x: (x == 1).sum())
    ).reset_index()

    X_train = agrupado[['ENTIDAD_RES', 'MUNICIPIO_RES', 'AÑO']]
    y_muertes = agrupado['muertes']
    y_casos = agrupado['casos_totales']

    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['ENTIDAD_RES', 'MUNICIPIO_RES', 'AÑO'])
    ])

    modelo_muertes = Pipeline([('prep', pre), ('rf', RandomForestRegressor(n_estimators=100, random_state=42))])
    modelo_casos = Pipeline([('prep', pre), ('rf', RandomForestRegressor(n_estimators=100, random_state=42))])

    modelo_muertes.fit(X_train, y_muertes)
    modelo_casos.fit(X_train, y_casos)

    municipios_2026 = df[['ENTIDAD_RES', 'MUNICIPIO_RES', 'MUNICIPIO', 'NOMBRE_ENTIDAD']].drop_duplicates()
    municipios_2026['AÑO'] = '2026'

    X_2026 = municipios_2026[['ENTIDAD_RES', 'MUNICIPIO_RES', 'AÑO']]
    municipios_2026['muertes_estimadas'] = modelo_muertes.predict(X_2026).round().astype(int)
    municipios_2026['casos_estimados'] = modelo_casos.predict(X_2026).round().astype(int)

    top = municipios_2026.sort_values(by='muertes_estimadas', ascending=False).head(10)

    resultado = [
        {
            "año": row['AÑO'],
            "entidad": row['NOMBRE_ENTIDAD'],
            "municipio": row['MUNICIPIO'],
            "muertes_estimadas": int(row['muertes_estimadas']),
            "casos_estimados": int(row['casos_estimados'])
        }
        for _, row in top.iterrows()
    ]

    return jsonify({"riesgo_2026": resultado}), 200

# -------------------- MAIN --------------------

if __name__ == '__main__':
    app.run(debug=True)
