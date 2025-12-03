# Motor de Redes Neuronales desde Cero

Este proyecto implementa un motor de redes neuronales totalmente desde cero usando únicamente **NumPy**, cumpliendo todos los requisitos del proyecto de Optimización y Heurística.

Incluye:
- Capas densas totalmente configurables
- Funciones de activación: Sigmoid, Tanh, ReLU, Softmax
- Funciones de pérdida: CrossEntropy y MSE
- Optimizadores: Adam (obligatorio), SGD
- Inicialización Xavier y He
- Entrenamiento con mini-batches
- División train/val/test
- Entrenador modular (`Trainer`)
- Notebook de experimentación
- Memoria en LaTeX

## 📂 Estructura del repositorio

```
OH_Proyecto/
 ├── src/
 │   ├── activations.py
 │   ├── dense.py
 │   ├── layers.py
 │   ├── losses.py
 │   ├── network.py
 │   ├── optimizer.py
 │   └── trainer.py
 ├── notebooks/
 │   └── Notebook_OH.ipynb
 ├── memoria/
 │   └── memoria_motor.tex
 ├── tests/
 ├── requirements.txt
 ├── .gitignore
 └── README.md
```

## 🚀 Instalación

```
pip install -r requirements.txt
```

## ▶️ Ejecución del notebook

Abre:

```
notebooks/Notebook_OH.ipynb
```

## 📊 Resultados

El motor aprende correctamente MNIST y puede ampliarse con nuevas funciones, capas y optimizadores.

## 📧 Autor
Proyecto generado con asistencia de ChatGPT.
