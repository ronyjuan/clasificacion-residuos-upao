# ¿A qué tacho va? — Clasificación inteligente de residuos (UPAO)

Sistema de clasificación de residuos mediante Deep Learning (transfer learning con **MobileNetV2**
y **EfficientNetB3**) para apoyar la segregación en los puntos ecológicos de la Universidad Privada
Antenor Orrego, Trujillo – Perú, 2026.

La aplicación web recibe una foto (archivo o cámara), la clasifica en 8 categorías y dice a qué
tacho va según la **NTP 900.058:2019** (verde, negro, marrón y rojo), con la confianza y la
distribución de probabilidades.

## Problema y objetivo

Los errores de segregación en el campus (plástico transparente frente a vidrio, papel frente a
cartón) causan contaminación cruzada y costos de separación. El objetivo es un clasificador que
identifique 8 categorías con accuracy y F1 ≥ 0.90 y funcione en tiempo real desde el navegador de
un celular.

## Dataset

**Garbage Classification V2**: 19 762 imágenes en 10 categorías; se descartaron `clothes` y
`shoes`, quedando **8 categorías y 12 458 imágenes**: battery (944), biological (997),
cardboard (1 825), glass (3 061), metal (1 020), paper (1 680), plastic (1 984), trash (947).

Preprocesamiento: 224 × 224 RGB, aumento de datos (rotación, zoom, desplazamientos, espejo,
brillo, contraste), pesos de clase *balanced* y partición estratificada **70/15/15** con semilla 42,
más validación cruzada estratificada de 5 pliegues.

## Resultados

| Modelo | Conjunto evaluado | Accuracy | Macro F1 |
|---|---|---|---|
| HOG + LinearSVC (línea base) | Prueba 20 % (2 492) | 0.4270 | 0.39 |
| EfficientNetB3 | Validación 20 % (2 489), recargado | 0.9032 | 0.90 |
| MobileNetV2 (ensayo original) | Validación 20 % (2 489) | 0.9168 | 0.92 |
| **MobileNetV2 mejorado (sistema)** | **Prueba independiente 15 % (1 869)** | **0.9294** (0.9299 con TTA) | **0.9282** |

Validación cruzada (5 pliegues): 0.9255 ± 0.0048. F1 mínimo por clase: 0.885 (trash).
Latencia: 79 ms/imagen en GPU de Colab; 2 422 984 parámetros.

### Modelo incluido en este repositorio (`models/mobilenetv2_mejorado.keras`)

Entrenado con `src/entrenar_local_cpu.py` en CPU: MobileNetV2 con el backbone congelado
(sin fine-tuning) y la misma cabeza, limpieza y partición 70/15/15. Se usó la **versión actual**
del dataset en Kaggle (carpeta `original/`), que para las 8 clases tiene **8 918 imágenes**
(battery 756, biological 699, cardboard 1 411, glass 1 736, metal 930, paper 1 336,
plastic 1 597, trash 453), por lo que sus cifras no son directamente comparables con las del informe.

| Métrica (prueba, 1 338 imágenes) | Valor |
|---|---|
| Accuracy | 0.9230 (0.9238 con TTA) |
| Macro F1 | 0.9265 |
| F1 mínimo por clase | 0.889 (metal) |
| Validación cruzada (5 pliegues) | 0.9088 de accuracy media |
| Latencia en CPU | ≈ 90 ms por imagen |

Detalle completo en `models/metricas_test.json`.

## Estructura

```
app.py                    Aplicación web (Streamlit)
clases.json               Etiquetas de las 8 clases
requirements.txt          Dependencias
.streamlit/config.toml    Tema de la app
models/                   Modelos entrenados (ver models/README.md)
src/entrenar_mejorado.py  Entrenamiento eficiente de MobileNetV2 (Colab con GPU)
src/evaluar_modelo.py     Métricas de un modelo sobre una carpeta de prueba
src/baseline_hog_svm.py   Modelo base HOG + SVM
notebooks/                Notebooks de Colab
docs/                     Informe del proyecto
tests_imgs/               Imágenes y capturas del plan de pruebas
```

## Ejecución local

Requiere Python 3.10 a 3.13 (TensorFlow aún no publica paquetes para 3.14).

```bash
git clone https://github.com/<usuario>/clasificacion-residuos-upao.git
cd clasificacion-residuos-upao
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
# copia mobilenetv2_mejorado.keras en models/
streamlit run app.py            # http://localhost:8501
```

Para evaluar un modelo sobre una carpeta `tests_imgs/prueba/<clase>/*.jpg`:

```bash
python src/evaluar_modelo.py --modelo models/mobilenetv2_mejorado.keras --datos tests_imgs/prueba
```

(necesita además `pip install scikit-learn`).

## Despliegue en Streamlit Community Cloud

1. Sube a GitHub el código, `requirements.txt`, `clases.json` y el modelo en `models/`.
2. Entra a <https://share.streamlit.io> con tu cuenta de GitHub → **Create app**.
3. Elige el repositorio, la rama `main` y `app.py`. En *Advanced settings* selecciona Python 3.11 o 3.12.
4. **Deploy**: se instalan las dependencias y se obtiene una URL pública con HTTPS
   (necesaria para que el celular permita usar la cámara).
5. Ejecuta el plan de pruebas (PR-01 a PR-07) y guarda las capturas en `tests_imgs/`.

La app muestra la latencia de cada predicción (PR-06) y avisa cuando la confianza es menor a 60 % (PR-05).

## Limitaciones

- El dataset es externo: los residuos y fondos del campus pueden diferir (cambio de dominio).
- Las 8 clases del dataset no coinciden 1 a 1 con la clasificación de la UPAO; se asocian a los 4 tachos de la NTP.
- Es una herramienta de apoyo: no guarda imágenes ni pide datos personales.

## Integrantes

Olazabal Avila, Fernando David · Lujan Holguin, Rony · Camilo Poma, Harrison Claudio · Cueva Mendoza, Jerry

Curso: Inteligencia Artificial, Principios y Técnicas (2026-20) · Docente: Teobaldo Hernan Sagastegui Chigne
