import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# === Cargar datos ===
df = pd.read_csv("uploads/dengue_combinado_filtrado_ordenado.csv")
df.columns = df.columns.str.upper()

# === Filtrar columnas necesarias ===
df = df[[ 
    'EDAD_ANOS', 'SEXO', 'TIPO_PACIENTE', 'DICTAMEN',
    'DIABETES', 'HIPERTENSION', 'EMBARAZO', 'INMUNOSUPR', 'DEFUNCION'
]].dropna()

# === Mapear valores como en el frontend ===
df['SEXO'] = df['SEXO'].map({1: 'Hombre', 2: 'Mujer'})
df['TIPO_PACIENTE'] = df['TIPO_PACIENTE'].map({1.0: 'Ambulatorio', 2.0: 'Hospitalizado'})
df['DICTAMEN'] = df['DICTAMEN'].map({
    1.0: 'Sospechoso', 2.0: 'Confirmado', 3.0: 'Otro',
    4.0: 'Otro', 5.0: 'Otro'
})
df['DIABETES'] = df['DIABETES'].map({1: 1, 2: 0})
df['HIPERTENSION'] = df['HIPERTENSION'].map({1: 1, 2: 0})
df['EMBARAZO'] = df['EMBARAZO'].map({1: 1, 2: 0})
df['INMUNOSUPR'] = df['INMUNOSUPR'].map({1: 1, 2: 0})
df = df.dropna()

# === Balancear dataset (undersampling) ===
df_muertes = df[df['DEFUNCION'] == 1]
df_vivos = df[df['DEFUNCION'] == 0].sample(n=len(df_muertes), random_state=42)
df_balanceado = pd.concat([df_muertes, df_vivos]).sample(frac=1, random_state=42)

X = df_balanceado.drop(columns='DEFUNCION')
y = df_balanceado['DEFUNCION']

# === Separar en categóricas y numéricas ===
cat_features = ['SEXO', 'TIPO_PACIENTE', 'DICTAMEN']
num_features = ['EDAD_ANOS', 'DIABETES', 'HIPERTENSION', 'EMBARAZO', 'INMUNOSUPR']

# === Preprocesador ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

# === Pipeline ===
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# === Entrenar y evaluar ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# === Guardar modelo ===
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/modelo_rf.pkl")
print("✅ Modelo balanceado guardado correctamente en 'models/modelo_rf.pkl'")
