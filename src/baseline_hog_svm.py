"""
Modelo base (baseline) de Machine Learning clásico: HOG + LinearSVC (Anexo A del informe).
Resultado reportado: accuracy 0.4270 sobre 2 492 imágenes de prueba.

Uso:  python src/baseline_hog_svm.py <ruta_del_dataset>
"""
import os
import sys

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/content/garbage/garbage-dataset"
CLASES = ["battery", "biological", "cardboard", "glass", "metal", "paper", "plastic", "trash"]

X, y = [], []
for categoria in CLASES:
    carpeta = os.path.join(dataset_path, categoria)
    for nombre in os.listdir(carpeta):
        img = cv2.imread(os.path.join(carpeta, nombre), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        X.append(hog(img, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), block_norm="L2-Hys"))
        y.append(categoria)
X, y = np.array(X), np.array(y)  # X: (12458, 8100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("Accuracy Baseline HOG + SVM:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
