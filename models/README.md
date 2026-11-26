# models/

`mobilenetv2_mejorado.keras` ya está incluido: se entrenó en CPU con `src/entrenar_local_cpu.py`
(métricas en `metricas_test.json`). Si entrenas la versión con fine-tuning en Colab
(`src/entrenar_mejorado.py`), reemplázalo con el archivo del mismo nombre.

Copia aquí los modelos entrenados en Colab (no se generan en este repositorio):

| Archivo | Origen | Uso en la app |
|---|---|---|
| `mobilenetv2_mejorado.keras` | Drive: `modelos_upa/entrenamiento_mejorado/` | **Modelo del sistema** (prueba 0.9294) |
| `mobilenetv2_residuos.h5` | Drive: `modelos_upa/` (ensayo original) | Comparación (validación 0.9168) |
| `efficientnetb3_residuos.h5` | Drive: `modelos_upa/` | Comparación (0.9032 al recargar) |

La app solo muestra en la barra lateral los modelos que encuentre en esta carpeta.
Guarda los `.h5` con `model.save(ruta, include_optimizer=False)`. Si un archivo pasa de
100 MB, usa Git LFS (`git lfs track "*.h5" "*.keras"`).
