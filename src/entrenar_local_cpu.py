"""
Entrenamiento de MobileNetV2 en CPU (sin GPU), versión "extractor congelado".

Misma limpieza, partición estratificada 70/15/15 (semilla 42) y cabeza de clasificación que
src/entrenar_mejorado.py, pero sin fine-tuning: las características del backbone se extraen
una sola vez (imagen y su espejo para entrenamiento) y se entrena solo la cabeza. Así el
entrenamiento completo cabe en una laptop. Al final se arma el modelo completo
(entrada 0-255 -> Rescaling -> MobileNetV2 -> cabeza) y se guarda como .keras.

Uso:  python src/entrenar_local_cpu.py <carpeta_dataset> [carpeta_salida]
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import callbacks, layers, losses, models, optimizers

DATA_DIR = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "models"
os.makedirs(OUT, exist_ok=True)
CLASES = ["battery", "biological", "cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG, SEED, LOTE = 224, 42, 64
tf.keras.utils.set_random_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

# ------------------------------------------------------------------ limpieza
EXT = (".jpg", ".jpeg", ".png", ".bmp")
rutas, y, vistos = [], [], set()
ilegibles = duplicados = 0
for i, c in enumerate(CLASES):
    for f in sorted(glob.glob(os.path.join(DATA_DIR, c, "*"))):
        if not f.lower().endswith(EXT):
            ilegibles += 1
            continue
        try:
            with Image.open(f) as im:
                im.verify()
            with open(f, "rb") as fh:
                h = hashlib.md5(fh.read()).hexdigest()
        except Exception:
            ilegibles += 1
            continue
        if h in vistos:
            duplicados += 1
            continue
        vistos.add(h)
        rutas.append(f)
        y.append(i)
rutas, y = np.array(rutas), np.array(y)
print(f"Válidas: {len(rutas)} | ilegibles: {ilegibles} | duplicados: {duplicados}", flush=True)
print({c: int((y == i).sum()) for i, c in enumerate(CLASES)}, flush=True)

r_tr, r_tmp, y_tr, y_tmp = train_test_split(rutas, y, test_size=0.30, stratify=y, random_state=SEED)
r_va, r_te, y_va, y_te = train_test_split(r_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print("train/val/test:", len(y_tr), len(y_va), len(y_te), flush=True)


# ------------------------------------------------------------------ extracción de características
def cargar(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, (IMG, IMG))  # float32 0-255


base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
base.trainable = False
inp = layers.Input((IMG, IMG, 3))
extractor = models.Model(inp, layers.GlobalAveragePooling2D()(base(layers.Rescaling(1 / 127.5, offset=-1)(inp),
                                                                     training=False)))


def caracteristicas(paths, espejo=False):
    ds = tf.data.Dataset.from_tensor_slices(paths).map(cargar, num_parallel_calls=AUTOTUNE)
    if espejo:
        ds = ds.map(tf.image.flip_left_right, num_parallel_calls=AUTOTUNE)
    t = time.time()
    X = extractor.predict(ds.batch(LOTE).prefetch(AUTOTUNE), verbose=0)
    print(f"  {len(paths)} imágenes{' (espejo)' if espejo else ''} en {time.time() - t:.0f} s", flush=True)
    return X


t0 = time.time()
print("Extrayendo características...", flush=True)
X_tr = caracteristicas(r_tr)
X_tr_f = caracteristicas(r_tr, espejo=True)
X_va = caracteristicas(r_va)
X_te = caracteristicas(r_te)
X_te_f = caracteristicas(r_te, espejo=True)


# ------------------------------------------------------------------ cabeza
def cabeza():
    return models.Sequential([
        layers.Input((X_tr.shape[1],)), layers.Dropout(0.3), layers.Dense(128, activation="relu"),
        layers.Dropout(0.2), layers.Dense(len(CLASES), activation="softmax"),
    ], name="cabeza")


def onehot(v):
    return tf.keras.utils.to_categorical(v, len(CLASES))


# validación cruzada (5 pliegues) sobre entrenamiento + validación
X_all, y_all = np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va])
accs, f1s = [], []
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_all, y_all), 1):
    m = cabeza()
    m.compile(optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    pw = dict(enumerate(class_weight.compute_class_weight("balanced", classes=np.arange(8), y=y_all[a])))
    m.fit(X_all[a], y_all[a], epochs=25, batch_size=128, verbose=0, class_weight=pw)
    p = m.predict(X_all[b], verbose=0).argmax(1)
    accs.append(float((p == y_all[b]).mean()))
    f1s.append(float(f1_score(y_all[b], p, average="macro")))
    print(f"Pliegue {k}: accuracy {accs[-1]:.4f} | macro-F1 {f1s[-1]:.4f}", flush=True)
cv = {"accuracy_media": float(np.mean(accs)), "accuracy_desv": float(np.std(accs)),
      "macroF1_media": float(np.mean(f1s)), "macroF1_desv": float(np.std(f1s)), "pliegues": 5}

# entrenamiento final de la cabeza (imagen + espejo), validación para detener a tiempo
Xf = np.concatenate([X_tr, X_tr_f])
yf = np.concatenate([y_tr, y_tr])
pesos = dict(enumerate(class_weight.compute_class_weight("balanced", classes=np.arange(8), y=y_tr)))
cab = cabeza()
cab.compile(optimizers.Adam(1e-3), losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
h = cab.fit(Xf, onehot(yf), validation_data=(X_va, onehot(y_va)), epochs=60, batch_size=128, verbose=0,
            class_weight=pesos,
            callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
                       callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)])
val_acc = max(h.history["val_accuracy"])
print(f"Épocas: {len(h.history['loss'])} | mejor val_accuracy {val_acc:.4f}", flush=True)

# ------------------------------------------------------------------ prueba (una sola vez)
p, pf = cab.predict(X_te, verbose=0), cab.predict(X_te_f, verbose=0)
y_pred, y_tta = p.argmax(1), ((p + pf) / 2).argmax(1)
acc, acc_tta = float((y_pred == y_te).mean()), float((y_tta == y_te).mean())
rep = classification_report(y_te, y_tta, target_names=CLASES, digits=3, output_dict=True)
print(f"PRUEBA accuracy {acc:.4f} | con TTA {acc_tta:.4f}", flush=True)
print(classification_report(y_te, y_tta, target_names=CLASES, digits=3), flush=True)
print(confusion_matrix(y_te, y_tta), flush=True)

# ------------------------------------------------------------------ modelo completo para la app
inp = layers.Input((IMG, IMG, 3), name="imagen_0_255")
x = layers.Rescaling(1 / 127.5, offset=-1)(inp)
x = layers.GlobalAveragePooling2D()(base(x, training=False))
out = cab(x)
completo = models.Model(inp, out, name="mobilenetv2_residuos")
ruta = os.path.join(OUT, "mobilenetv2_mejorado.keras")
completo.save(ruta, include_optimizer=False)

# comprobación: el modelo guardado da lo mismo que la cabeza sobre las características
chk = tf.keras.models.load_model(ruta, compile=False)
muestra = np.stack([cargar(r).numpy() for r in r_te[:16]])
assert np.allclose(chk.predict(muestra, verbose=0), p[:16], atol=1e-3), "el modelo guardado no coincide"

uno = np.zeros((1, IMG, IMG, 3), "float32")
for _ in range(3):
    chk.predict(uno, verbose=0)
t = time.time()
for _ in range(20):
    chk.predict(uno, verbose=0)
lat = (time.time() - t) / 20 * 1000

resumen = {
    "backbone": "mobilenetv2", "entrenamiento": "CPU, backbone congelado (sin fine-tuning)",
    "particion": {"validas": int(len(rutas)), "ilegibles": ilegibles, "duplicados": duplicados,
                  "train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te))},
    "val_accuracy": float(val_acc), "test_accuracy": acc, "test_accuracy_tta": acc_tta,
    "test_macroF1_tta": float(f1_score(y_te, y_tta, average="macro")),
    "test_F1_clase_mas_debil": float(min(rep[c]["f1-score"] for c in CLASES)),
    "validacion_cruzada": cv, "latencia_ms_cpu": lat, "minutos": (time.time() - t0) / 60,
    "parametros": int(completo.count_params()), "reporte": rep,
}
json.dump(resumen, open(os.path.join(OUT, "metricas_test.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in resumen.items() if k != "reporte"}, indent=2, ensure_ascii=False), flush=True)
print("Guardado:", ruta, f"({os.path.getsize(ruta) / 1e6:.1f} MB)", flush=True)
