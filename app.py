"""
Clasificador de residuos - UPAO (Inteligencia Artificial: Principios y Técnicas, 2026-20)

Aplicación web (Streamlit). Carga el modelo del sistema (MobileNetV2 mejorado) y,
opcionalmente, los de comparación (MobileNetV2 original y EfficientNetB3), y clasifica
una imagen de un residuo en 8 categorías. Indica además el tacho que corresponde según
la NTP 900.058:2019 (Inacal).

Ejecutar:  streamlit run app.py
"""
import html
import json
import os
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)
CLASSES_PATH = "clases.json"
UMBRAL_CONFIANZA = 0.60

# nombre visible -> (archivo del modelo, preprocesamiento usado al entrenar)
#   "interno"      : el modelo trae su propio preprocesamiento; recibe píxeles 0-255
#   "mobilenet"    : MobileNetV2 del ensayo original, entrenado con rescale = 1/255
#   "efficientnet" : EfficientNetB3 normaliza por dentro; recibe píxeles 0-255
MODELOS = {
    "MobileNetV2 mejorado (sistema)": ("models/mobilenetv2_mejorado.keras", "interno"),
    "MobileNetV2 (ensayo original)": ("models/mobilenetv2_residuos.h5", "mobilenet"),
    "EfficientNetB3 (comparación)": ("models/efficientnetb3_residuos.h5", "efficientnet"),
}

# NTP 900.058:2019 (Inacal): cuatro colores de tacho
TACHOS = {
    "verde": ("Verde", "#1f7a4d", "Aprovechables: plástico, vidrio, metal, papel y cartón"),
    "negro": ("Negro", "#1b1b19", "No aprovechables: papel encerado, sanitarios, colillas"),
    "marron": ("Marrón", "#7a4a2a", "Orgánicos: restos de comida y de poda"),
    "rojo": ("Rojo", "#c8372d", "Peligrosos: pilas, lámparas, medicinas vencidas"),
}

# clase -> (etiqueta legible, tacho, consejo)
INFO = {
    "battery": ("Pilas / baterías", "rojo", "Residuo peligroso: llévalo a un punto de acopio."),
    "biological": ("Orgánico", "marron", "Restos de comida o de poda."),
    "cardboard": ("Cartón", "verde", "Mantenlo seco y limpio; aplánalo si puedes."),
    "glass": ("Vidrio", "verde", "Sin tapa ni etiquetas sueltas."),
    "metal": ("Metal", "verde", "Enjuaga las latas antes de botarlas."),
    "paper": ("Papel", "verde", "Mantenlo seco y limpio."),
    "plastic": ("Plástico", "verde", "Enjuaga el envase y quítale los restos."),
    "trash": ("No aprovechable", "negro", "Residuo que no se recicla."),
}


@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo(ruta):
    import tensorflow as tf  # import diferido: la interfaz aparece antes

    return tf.keras.models.load_model(ruta, compile=False)


@st.cache_data
def cargar_clases():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["class_names"]


def preparar(img: Image.Image, tipo: str) -> np.ndarray:
    """Replica el preprocesamiento usado al entrenar cada modelo (sección 3.2)."""
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype="float32")
    if tipo == "mobilenet":
        x = x / 255.0
    return np.expand_dims(x, axis=0)


def predecir(modelo, x: np.ndarray, tta: bool):
    """Devuelve (probabilidades, milisegundos). Con TTA promedia la imagen y su espejo."""
    lote = np.concatenate([x, x[:, :, ::-1, :]]) if tta else x
    t0 = time.perf_counter()
    p = modelo.predict(lote, verbose=0)
    ms = (time.perf_counter() - t0) * 1000
    return np.asarray(p.mean(axis=0) if tta else p[0], dtype="float64"), ms


ESTILO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,800&family=Instrument+Sans:wght@400;500;600&display=swap');
html,body,.stApp,[class*="css"]{font-family:'Instrument Sans',sans-serif}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1100px;padding-top:2.2rem}
@keyframes rise{from{transform:translateY(105%)}to{transform:none}}
@keyframes fade{from{opacity:0;transform:translateY(10px)}}
@keyframes subir{from{transform:translateY(40px);opacity:0}to{transform:none;opacity:1}}
@keyframes golpe{40%{transform:translateY(7px)}70%{transform:translateY(-3px)}}
@keyframes tapa{from{transform:rotate(0)}to{transform:rotate(-24deg)}}
@keyframes cinta{to{transform:translateX(-50%)}}
.cab{display:flex;justify-content:space-between;border-bottom:1px solid #17140f;padding-bottom:12px;margin-bottom:2rem;font-size:.85rem;color:#6d665a}
.cab b{font-family:'Fraunces',serif;font-weight:800;font-size:1.2rem;color:#17140f;letter-spacing:-.02em}
h1.titulo{font-family:'Fraunces',serif;font-weight:800;font-size:clamp(3.2rem,9vw,7rem);line-height:.9;letter-spacing:-.045em;margin:0 0 1rem;overflow:hidden;padding-bottom:.15em;color:#17140f}
h1.titulo span{display:block;animation:rise 1s cubic-bezier(.2,.8,.2,1) both}
h1.titulo span+span{animation-delay:.12s}h1.titulo em{font-weight:400}
p.sub{color:#3b362d;font-size:1.1rem;max-width:540px;margin-bottom:1.6rem;animation:fade 1s .4s both}
.etq{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#6d665a;margin-bottom:.6rem}
.cinta{background:#f2d764;border-block:1.5px solid #17140f;transform:rotate(-1deg);margin:1.6rem -2rem 2rem;padding:10px 0;overflow:hidden;white-space:nowrap}
.cinta div{display:inline-flex;gap:26px;animation:cinta 38s linear infinite}
.cinta span{font-family:'Fraunces',serif;font-style:italic;font-size:1.5rem;color:#17140f}
.cinta u{text-decoration:none;font-style:normal;font-weight:800;margin-left:26px}
.tachos{padding:1rem 1rem 0;border-radius:4px;transition:background .7s;background:var(--tinte,transparent);display:flex;gap:2%;justify-content:space-between;margin:1rem 0 .4rem;border-bottom:1px solid #17140f}
.tacho{width:23%;text-align:center;opacity:.3;animation:subir .9s cubic-bezier(.2,.8,.2,1) both}
.tacho.hit{opacity:1}.tacho.hit svg{animation:golpe .5s .4s}
.tacho svg{width:100%;height:auto;overflow:visible;display:block}
.tacho .tapa{transform-box:fill-box;transform-origin:0% 100%}
.tacho.hit .tapa{animation:tapa .45s .15s cubic-bezier(.3,1.6,.5,1) both}
.tacho b{font-family:'Fraunces',serif;font-size:1.1rem;display:block;margin-top:.3rem;color:#17140f}
.tacho small{display:block;color:#6d665a;font-size:.72rem;line-height:1.3;padding-bottom:.8rem}
.veredicto{font-family:'Fraunces',serif;font-size:clamp(1.8rem,3.4vw,2.8rem);letter-spacing:-.03em;line-height:1.05;margin:.8rem 0 .3rem;animation:fade .7s both;color:#17140f}
.veredicto small{font-family:'Instrument Sans',sans-serif;font-size:.9rem;letter-spacing:0;color:#6d665a;margin-left:.4rem}
.pct{font-family:'Fraunces',serif;font-weight:800;font-size:4.2rem;letter-spacing:-.05em;line-height:.9;color:#17140f}
.pct small{font-family:'Instrument Sans',sans-serif;font-weight:400;font-size:.85rem;letter-spacing:0;color:#6d665a;margin-left:.6rem}
.consejo{color:#3b362d;margin:.4rem 0 1rem}
.fila{display:grid;grid-template-columns:120px 1fr 60px;gap:12px;align-items:center;padding:9px 0;border-top:1px solid #17140f26;font-size:.92rem;color:#17140f;
 animation:fade .6s both;animation-delay:calc(var(--i)*.07s + .3s)}
.fila .eje{position:relative;height:1px;background:#17140f}
.fila .eje i{position:absolute;top:50%;width:13px;height:13px;border-radius:50%;background:#17140f;transform:translate(-50%,-50%)}
.fila.top{font-weight:600}.fila.top .eje i{background:#1f7a4d;width:17px;height:17px}
.fila span:last-child{text-align:right;font-family:ui-monospace,monospace;font-size:.85rem}
.stButton>button{border-radius:3px;background:#17140f;color:#efe9dc;transition:transform .2s,background .3s}
.stButton>button:hover{background:#1f7a4d;transform:translateY(-2px);color:#fff}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed #17140f66;border-radius:3px;background:transparent}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#1f7a4d;background:#1f7a4d0d}
[data-testid="stImage"] img{border-radius:3px}
</style>
"""

BIN_SVG = (
    '<svg viewBox="0 0 200 250" aria-hidden="true"><g class="tapa">'
    '<rect x="14" y="10" width="172" height="26" rx="5" fill="{c}"/>'
    '<rect x="78" y="0" width="44" height="12" rx="4" fill="{c}"/></g>'
    '<path d="M26 40h148l-12 190a10 10 0 0 1-10 9H48a10 10 0 0 1-10-9z" fill="{c}"/>'
    '<path d="M60 70v140M100 70v140M140 70v140" stroke="#fff" stroke-opacity=".14" '
    'stroke-width="6" stroke-linecap="round"/>'
    '<circle cx="52" cy="244" r="6" fill="#17140f"/><circle cx="148" cy="244" r="6" fill="#17140f"/></svg>'
)
TINTES = {"verde": "#cfe3d3", "negro": "#d8d4cb", "marron": "#e8d3bf", "rojo": "#f2cfc7"}
COLS = ["#1f7a4d", "#17140f", "#7a4a2a", "#c8372d"]
NOMBRES = ["Plástico", "Vidrio", "Metal", "Papel", "Cartón", "Orgánico", "Pilas", "No aprovechable"]


def cinta_html():
    una = "".join(
        f'<span>{n}<u style="color:{COLS[i % 4]}">✱</u></span>' for i, n in enumerate(NOMBRES)
    )
    return f'<div class="cinta"><div>{una}{una}</div></div>'


def tachos_html(activo=None):
    out = []
    for k, (n, c, t) in TACHOS.items():
        out.append(
            f'<div class="tacho{" hit" if k == activo else ""}" style="animation-delay:{.15 * len(out):.2f}s">'
            f"{BIN_SVG.format(c=c)}<b>{n}</b><small>{t}</small></div>"
        )
    tinte = f' style="--tinte:{TINTES[activo]}"' if activo else ""
    return f'<div class="tachos"{tinte}>' + "".join(out) + "</div>"


def barras(clases, probs, n=8):
    orden = np.argsort(probs)[::-1][:n]
    filas = []
    for k, i in enumerate(orden):
        nombre = INFO.get(clases[i], (clases[i],))[0]
        filas.append(
            f'<div class="fila{" top" if k == 0 else ""}" style="--i:{k}"><span>{html.escape(nombre)}</span>'
            f'<div class="eje"><i style="left:{probs[i] * 100:.1f}%"></i></div>'
            f"<span>{probs[i]:.1%}</span></div>"
        )
    return "".join(filas)


# ----------------------------------------------------------------------------- interfaz
st.set_page_config(page_title="¿A qué tacho va? · UPAO", page_icon="♻️", layout="wide")
st.markdown(ESTILO, unsafe_allow_html=True)
st.markdown(
    '<div class="cab"><b>Tacho</b><span>Proyecto de IA · UPAO 2026-20</span></div>'
    '<h1 class="titulo"><span>¿A qué</span><span><em>tacho</em> va?</span></h1>'
    '<p class="sub">Sube o toma la foto de un residuo. Una red neuronal (MobileNetV2) dice qué es '
    "y en cuál de los cuatro tachos de la norma peruana (NTP 900.058:2019) debe ir.</p>"
    + cinta_html(),
    unsafe_allow_html=True,
)

disponibles = {k: v for k, v in MODELOS.items() if os.path.exists(v[0])}
if not disponibles:
    st.error(
        "No se encontró ningún modelo en la carpeta `models/`. Copia "
        "`mobilenetv2_mejorado.keras` (desde Drive: `modelos_upa/entrenamiento_mejorado/`) "
        "y, si quieres comparar, `mobilenetv2_residuos.h5` y `efficientnetb3_residuos.h5`. Ver README."
    )
    st.markdown(tachos_html(), unsafe_allow_html=True)
    st.stop()

with st.sidebar:
    st.markdown("### Configuración")
    nombre = st.selectbox("Modelo", list(disponibles))
    tta = st.toggle(
        "TTA (imagen + espejo)", value=False, help="Promedia la predicción de la imagen y de su reflejo."
    )
    st.caption(
        "Modelo incluido, evaluado en el conjunto de prueba (1 338 imágenes): "
        "accuracy 0.9238 (con TTA) · macro F1 0.9265."
    )

ruta, tipo = disponibles[nombre]
modelo = cargar_modelo(ruta)
clases = cargar_clases()

if "clave" not in st.session_state:
    st.session_state.clave = 0

izq, der = st.columns([4, 6], gap="large")
imagen = None
with izq:
    st.markdown('<div class="etq">1 · La imagen</div>', unsafe_allow_html=True)
    origen = st.radio(
        "Origen", ["Subir archivo", "Usar cámara"], horizontal=True, label_visibility="collapsed"
    )
    clave = f"{origen}-{st.session_state.clave}"
    if origen == "Subir archivo":
        archivo = st.file_uploader(
            "Imagen del residuo", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed", key=clave,
        )
    else:
        archivo = st.camera_input("Foto del residuo", label_visibility="collapsed", key=clave)
    if archivo is not None:
        try:
            imagen = Image.open(archivo)
            imagen.load()
        except Exception:
            st.error("No se pudo leer la imagen. Prueba con un JPG o PNG.")
            imagen = None
    if imagen is not None:
        st.image(ImageOps.exif_transpose(imagen), width="stretch")
        if st.button("Nueva clasificación", width="stretch"):
            st.session_state.clave += 1
            st.rerun()

with der:
    st.markdown('<div class="etq">2 · El tacho</div>', unsafe_allow_html=True)
    if imagen is None:
        st.markdown(tachos_html(), unsafe_allow_html=True)
        st.write("Sube o captura una imagen y el tacho correcto se abre.")
    else:
        probs, ms = predecir(modelo, preparar(imagen, tipo), tta)
        top = int(np.argmax(probs))
        etiqueta, tacho, consejo = INFO.get(clases[top], (clases[top], "negro", ""))
        st.markdown(tachos_html(tacho), unsafe_allow_html=True)
        st.markdown(
            f'<div class="veredicto">{etiqueta} → tacho {TACHOS[tacho][0].lower()}'
            f"<small>{clases[top]}</small></div>"
            f'<div class="pct">{probs[top]:.0%}<small>confianza · {ms:.0f} ms</small></div>'
            f'<div class="consejo">{consejo}</div>',
            unsafe_allow_html=True,
        )
        if probs[top] < UMBRAL_CONFIANZA:
            st.warning("Confianza baja: acerca el objeto, mejora la luz o prueba otra foto.")
        st.markdown(barras(clases, probs), unsafe_allow_html=True)

st.divider()
st.caption(
    f"Modelo activo: {nombre}. Herramienta de apoyo a la decisión; puede equivocarse con residuos "
    "distintos a los del dataset de entrenamiento. Las imágenes no se guardan. "
    "Código de colores: NTP 900.058:2019 (Inacal)."
)
