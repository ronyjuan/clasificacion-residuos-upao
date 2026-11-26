"""
Evalúa un modelo sobre una carpeta de prueba organizada por clase:

    carpeta/
      battery/  *.jpg
      biological/ ...
      ...

Uso:
    python src/evaluar_modelo.py --modelo models/mobilenetv2_mejorado.keras --datos tests_imgs/prueba
    python src/evaluar_modelo.py --modelo models/mobilenetv2_residuos.h5 --tipo mobilenet --datos ...

--tipo: "interno" (entrada 0-255; modelo mejorado y EfficientNetB3) o "mobilenet" (rescale 1/255).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def preparar(ruta, tipo):
    img = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB").resize((224, 224))
    x = np.asarray(img, dtype="float32")
    return x / 255.0 if tipo == "mobilenet" else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--datos", required=True)
    ap.add_argument("--tipo", default="interno", choices=["interno", "mobilenet", "efficientnet"])
    ap.add_argument("--clases", default="clases.json")
    ap.add_argument("--lote", type=int, default=32)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    clases = json.load(open(args.clases, encoding="utf-8"))["class_names"]
    rutas, y = [], []
    for i, c in enumerate(clases):
        for f in sorted(glob.glob(os.path.join(args.datos, c, "*"))):
            if f.lower().endswith(EXT):
                rutas.append(f)
                y.append(i)
    if not rutas:
        sys.exit(f"No hay imágenes en {args.datos}/<clase>/")
    print(f"{len(rutas)} imágenes en {len(set(y))} clases")

    modelo = tf.keras.models.load_model(args.modelo, compile=False)
    preds = []
    t0 = time.perf_counter()
    for k in range(0, len(rutas), args.lote):
        lote = np.stack([preparar(r, args.tipo) for r in rutas[k:k + args.lote]])
        preds.append(modelo.predict(lote, verbose=0))
    ms = (time.perf_counter() - t0) / len(rutas) * 1000
    p = np.concatenate(preds).argmax(1)
    y = np.array(y)

    print(f"Accuracy: {(p == y).mean():.4f}  |  {ms:.0f} ms por imagen (incluye lectura)")
    etiquetas = list(range(len(clases)))
    print(classification_report(y, p, labels=etiquetas, target_names=clases, digits=3, zero_division=0))
    print("Matriz de confusión (filas: real, columnas: predicha)")
    print(confusion_matrix(y, p, labels=etiquetas))


if __name__ == "__main__":
    main()
