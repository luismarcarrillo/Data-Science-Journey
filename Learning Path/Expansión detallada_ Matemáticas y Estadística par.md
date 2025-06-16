# 🧮 Fase 2 — Fundamentos Matemáticos y Estadísticos para Ciencia de Datos

### 🎯 Objetivo General

Desarrollar un dominio sólido de las **matemáticas esenciales** aplicadas a ciencia de datos: **álgebra lineal**, **cálculo**, **probabilidad** y **estadística**, con orientación práctica en Python. Esta fase habilita la comprensión profunda de modelos y algoritmos de machine learning, más allá del uso mecánico de librerías.

---

## 🔸 Nivel 1 — Álgebra Lineal Computacional

> *“No hay machine learning sin vectores, matrices y transformaciones.”*

### 🎯 OKR

- Comprender estructuras fundamentales: vectores, matrices y transformaciones lineales.
- Aplicar operaciones matriciales para manipulación de datos y reducción de dimensionalidad.
- Implementar conceptos básicos con Python (NumPy).

### 📚 Recursos Clave

- 📘 *Essential Math for Data Science* – O’Reilly.
- 📘 *Mathematics for Machine Learning* – Deisenroth et al. (Cap. 1–3)
- 🔧 [Curso: Linear Algebra – 3Blue1Brown](https://www.3blue1brown.com/essence-of-linear-algebra)

### 🧱 Contenidos

- Espacios vectoriales y álgebra de vectores.
- Producto escalar y norma vectorial.
- Producto matricial, identidad, inversa, determinante.
- Sistemas de ecuaciones lineales.
- Eigenvectores y valores propios.
- Introducción a PCA (análisis de componentes principales).
- Visualización geométrica con Matplotlib y Python.

### 🛠 Mini-proyectos

| Proyecto                                        | Conceptos aplicados        |
| ---------------------------------------------- | -------------------------- |
| Reducción de imágenes en blanco y negro con PCA | Matrices, eigenvectores    |
| Simulador de transformaciones lineales en 2D    | Multiplicación de matrices |

---

## 🔸 Nivel 2 — Cálculo Aplicado a Machine Learning

> *“El descenso por gradiente se basa en derivadas. El ajuste de modelos, en optimización.”*

### 🎯 OKR

- Entender funciones, derivadas e integrales en contexto de optimización.
- Aplicar el cálculo a problemas básicos de minimización de funciones de pérdida.
- Programar funciones derivadas y analizar su comportamiento.

### 📚 Recursos Clave

- 📘 *Mathematics for Machine Learning* – Cap. 4–5.
- 📘 *Essential Math for Data Science* – Sección de cálculo diferencial.
- 📘 *Calculus Made Easy* – Thompson.
- 🔧 [Khan Academy: Cálculo diferencial e integral](https://es.khanacademy.org/math/calculus-1)

### 🧱 Contenidos

- Funciones: lineales, polinómicas, exponenciales.
- Límites, continuidad, derivadas.
- Derivadas parciales y gradientes.
- Reglas de la cadena y optimización.
- Concepto de integral como área bajo la curva.
- Aplicaciones del gradiente: descenso por gradiente y ajustes.

### 🛠 Mini-proyectos

| Proyecto                                                   | Conceptos aplicados   |
| ---------------------------------------------------------- | --------------------- |
| Visualización del descenso por gradiente en una parábola   | Derivadas, gradientes |
| Aproximación del área bajo una curva con métodos numéricos | Integrales            |

---

## 🔸 Nivel 3 — Fundamentos de Probabilidad y Estadística

> *“La estadística es el alma de los modelos predictivos.”*

### 🎯 OKR

- Modelar la incertidumbre con probabilidad.
- Describir datos, distribuciones y estimar parámetros con estadística.
- Aplicar conceptos de inferencia, estimación y pruebas de hipótesis.

### 📚 Recursos Clave

- 📘 *Estadística: 50 conceptos fundamentales con Python* – David Spiegelhalter.
- 📘 *Essential Math for Data Science* – Secciones 3 y 4.
- 📘 *Think Stats* – Allen B. Downey (gratuito en línea).
- 🔧 [Curso: Intro to Statistics – Stanford Online](https://online.stanford.edu/courses/sohs-ystatslearning-statistics)

### 🧱 Contenidos

#### 📌 Probabilidad

- Espacios muestrales y eventos.
- Reglas de la probabilidad (suma, producto, Bayes).
- Variables aleatorias (discretas y continuas).
- Distribuciones (Binomial, Normal, Poisson).

#### 📌 Estadística Descriptiva

- Media, mediana, moda.
- Varianza, desviación estándar.
- Visualización de datos: histogramas, boxplots.

#### 📌 Inferencia Estadística

- Estimación de parámetros.
- Intervalos de confianza.
- Pruebas de hipótesis (t, z, chi-cuadrado).
- Correlación y causalidad.

### 🛠 Mini-proyectos

| Proyecto                                   | Conceptos aplicados   |
| ------------------------------------------ | --------------------- |
| Simulación de dados y monedas              | Probabilidad discreta |
| Estudio de salarios en Colombia            | Medidas descriptivas  |
| Evaluación de diferencias entre dos grupos | Prueba t              |

---

## 🔸 Nivel 4 — Modelos Estadísticos Fundamentales

> *“Antes de usar una red neuronal, domina una regresión lineal.”*

### 🎯 OKR

- Entender modelos de regresión como herramientas de predicción.
- Implementar regresión lineal y logística en Python (desde cero y con `scikit-learn`).
- Evaluar y validar modelos con métricas estadísticas.

### 📚 Recursos Clave

- 📘 *An Introduction to Statistical Learning (ISLR)* – Cap. 3–4.
- 📘 *Hands-On ML with Scikit-learn* – Capítulos iniciales.
- 🔧 [Curso: Fundamentos de ML – fast.ai](https://course.fast.ai/)

### 🧱 Contenidos

- Regresión lineal simple y múltiple.
- Supuestos del modelo lineal.
- Regresión logística (clasificación binaria).
- Overfitting, regularización (Ridge, Lasso).
- Métricas: MAE, MSE, R², precisión, recall, F1-score.

### 🛠 Mini-proyectos

| Proyecto                                                  | Conceptos aplicados |
| --------------------------------------------------------- | ------------------- |
| Predicción del puntaje de matemáticas vs horas de estudio | Regresión lineal    |
| Clasificador de spam con texto vectorizado                | Regresión logística |

---

## 💡 Recomendaciones para Integrar Programación y Matemáticas

| Estrategia                                 | Ejemplo                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| 🧪 Experimentación activa                  | Visualizar cómo cambia la pendiente en una función al ajustar su derivada |
| 🔢 Código como herramienta de comprobación | Calcular media y varianza con NumPy y verificar manualmente               |
| 📊 Visualización constante                 | Ver histogramas, dispersión, transformaciones vectoriales                 |
| 🔁 Proyectos de cierre por nivel           | Terminar cada etapa con un análisis o simulación de datos reales          |

---

## 📘 Evaluación del Progreso y Consolidación

### ✅ Checkpoints por nivel

- **Álgebra lineal**: Implementar PCA básico.
- **Cálculo**: Implementar y graficar descenso por gradiente.
- **Estadística**: Interpretar distribuciones reales (e.g., salarios, COVID).
- **Modelos**: Predecir outcomes con regresión lineal + interpretación.

### 📁 Portafolio inicial

- Notebooks con visualizaciones, simulaciones, modelos.
- Reportes de análisis básico con Markdown.
- Mini-datasets procesados por ti.

---

## 🧭 Referencias

[^1]: [Essential Math for Data Science – O’Reilly](https://www.oreilly.com/library/view/essential-math-for/9781098115560/)
[^2]: [Mathematics for Machine Learning – Deisenroth et al.](https://mml-book.github.io/)
[^3]: [Estadística: 50 conceptos fundamentales con Python – Spiegelhalter](https://www.alianzaeditorial.es/libro/educacion/estadistica-50-conceptos-fundamentales-con-python-david-spiegelhalter-9788411482052/)
[^4]: [Think Stats – Allen B. Downey](https://greenteapress.com/wp/think-stats/)
[^5]: [Khan Academy – Cálculo y Estadística](https://es.khanacademy.org)
[^6]: [3Blue1Brown – Álgebra Lineal](https://www.3blue1brown.com/essence-of-linear-algebra)

---

