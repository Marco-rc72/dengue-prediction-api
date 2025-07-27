# 🦟 API de Predicción de Defunción por Dengue

Este proyecto implementa una API REST utilizando **Flask** y un modelo de **Machine Learning (Random Forest)** para predecir si un paciente con diagnóstico de dengue tiene riesgo de fallecer, basado en variables clínicas y demográficas.

---

## 📁 Estructura del Proyecto

```
dengue_api/
├── app.py                  # Archivo principal con la API Flask
├── modelo/
│   └── modelo_rf.pkl       # Modelo entrenado (NO INCLUIDO en GitHub)
├── uploads/                # Carpeta temporal para archivos CSV subidos
├── requirements.txt        # Lista de dependencias
└── README.md               # Documentación del proyecto
```

---

## ⚙️ Instrucciones de Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

### 2. Crear entorno virtual (opcional)

```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate   # En Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Colocar el modelo entrenado

El archivo `modelo_rf.pkl` **no se incluye en este repositorio** por buenas prácticas.

Debes colocarlo manualmente en una carpeta llamada `modelo/` así:

```
dengue_api/
└── modelo/
    └── modelo_rf.pkl
```

Puedes obtener este archivo desde un enlace compartido por el autor (Drive, WeTransfer, etc).

---

### 5. Ejecutar la API

```bash
python app.py
```

La API estará disponible en:  
📍 `http://localhost:5000`

---

## 📤 Uso del endpoint `/predict`

- Método: `POST`
- URL: `http://localhost:5000/predict`
- Tipo: `multipart/form-data`
- Campo: `file` (archivo CSV)

### Formato del CSV esperado:

Debe incluir estas columnas:

```
EDAD_ANOS, SEXO, TIPO_PACIENTE, DICTAMEN, DIABETES, HIPERTENSION, EMBARAZO, INMUNOSUPR
```

### Ejemplo de respuesta:

```json
{
  "predicciones": [0, 1, 0, 0, 1]
}
```

---

## 👥 Autores

- Marco Rocha  
- Alex  
- Yahir  
- Boris  
- Geovani  
- Soto  
- Tuz

---

## 🛡️ Nota de Seguridad

Por seguridad y limpieza del repositorio, los archivos `.pkl` y `.csv` fueron excluidos con `.gitignore`.

Para usar el modelo, solicita el archivo `modelo_rf.pkl` directamente al autor.
