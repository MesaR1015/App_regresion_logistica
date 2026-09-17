import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)

st.set_page_config(page_title="¿Lloverá mañana? — Regresión Logística", layout="wide")


# ----------------------------------------------------------------------
# 1. Generación de datos
# ----------------------------------------------------------------------
@st.cache_data
def generar_datos(n=300, seed=42):
    rng = np.random.default_rng(seed)
    temperatura = rng.normal(24, 5, n)
    humedad = rng.normal(65, 15, n)
    viento = rng.normal(10, 5, n)

    z = -6 + 0.09 * humedad - 0.05 * temperatura + 0.02 * viento
    prob_real = 1 / (1 + np.exp(-z))
    llovio = (rng.random(n) < prob_real).astype(int)

    return pd.DataFrame(
        {
            "temperatura": temperatura.round(1),
            "humedad": humedad.round(1),
            "viento": viento.round(1),
            "llovio": llovio,
        }
    )


df = generar_datos()

st.title("🌧️ ¿Lloverá mañana? — Regresión Logística interactiva")
st.markdown(
    "Usa los controles de la barra lateral para explorar cómo cambian las métricas del "
    "modelo al modificar variables, ajustar el umbral de decisión o probar un día nuevo."
)

# ----------------------------------------------------------------------
# 2. Controles de la Barra Lateral
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del modelo")

usar_temp = st.sidebar.checkbox("Usar temperatura", value=True)
usar_humedad = st.sidebar.checkbox("Usar humedad", value=True)
usar_viento = st.sidebar.checkbox("Usar viento", value=True)

variables = [
    v
    for v, usar in [
        ("temperatura", usar_temp),
        ("humedad", usar_humedad),
        ("viento", usar_viento),
    ]
    if usar
]

if len(variables) == 0:
    st.sidebar.error("Selecciona al menos una variable para entrenar el modelo.")
    st.stop()

umbral = st.sidebar.slider(
    "Umbral de clasificación", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)

st.sidebar.header("🌤️ Simular un nuevo día")
temp_input = st.sidebar.slider("Temperatura (°C)", 5.0, 40.0, 24.0, 0.5)
humedad_input = st.sidebar.slider("Humedad (%)", 0.0, 100.0, 85.0, 1.0)
viento_input = st.sidebar.slider("Viento (km/h)", 0.0, 40.0, 12.0, 0.5)

nuevo_dia_completo = pd.DataFrame(
    {"temperatura": [temp_input], "humedad": [humedad_input], "viento": [viento_input]}
)


# ----------------------------------------------------------------------
# 3. Entrenamiento
# ----------------------------------------------------------------------
@st.cache_data
def entrenar_modelo(df, vars_tuple):
    X = df[list(vars_tuple)]
    y = df["llovio"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    modelo = LogisticRegression(solver="lbfgs")
    modelo.fit(X_train, y_train)
    return modelo, X_test, y_test


modelo, X_test, y_test = entrenar_modelo(df, tuple(variables))
prob_test = modelo.predict_proba(X_test)[:, 1]
pred_test = (prob_test >= umbral).astype(int)

prob_nuevo = modelo.predict_proba(nuevo_dia_completo[variables])[0, 1]
pred_nuevo = "Sí lloverá 🌧️" if prob_nuevo >= umbral else "No lloverá ☀️"

# ----------------------------------------------------------------------
# 4. Layout principal
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Predicción para el nuevo día")
    st.metric("Probabilidad de lluvia", f"{prob_nuevo:.2%}")
    st.metric("Predicción (con umbral elegido)", pred_nuevo)

    st.subheader("📈 Curva sigmoide")
    z_vals = np.linspace(-10, 10, 200)
    sigmoide = 1 / (1 + np.exp(-z_vals))
    z_nuevo = np.log(prob_nuevo / (1 - prob_nuevo)) if 0 < prob_nuevo < 1 else 0

    fig_sig, ax_sig = plt.subplots(figsize=(5, 3.5))
    ax_sig.plot(z_vals, sigmoide, color="darkblue")
    ax_sig.axhline(
        umbral, color="gray", linestyle="--", linewidth=1, label=f"Umbral = {umbral}"
    )
    ax_sig.scatter([z_nuevo], [prob_nuevo], color="red", zorder=5, label="Nuevo día")
    ax_sig.set_xlabel("z")
    ax_sig.set_ylabel("Probabilidad")
    ax_sig.legend()
    st.pyplot(fig_sig)

with col2:
    st.subheader("🗺️ Temperatura vs. Humedad")
    colores = df["llovio"].map({0: "skyblue", 1: "darkblue"})
    fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
    ax_scatter.scatter(
        df["temperatura"], df["humedad"], c=colores, alpha=0.6, edgecolor="k"
    )
    ax_scatter.scatter(
        [temp_input],
        [humedad_input],
        color="red",
        s=120,
        edgecolor="black",
        marker="*",
        zorder=5,
        label="Nuevo día",
    )
    ax_scatter.set_xlabel("Temperatura (°C)")
    ax_scatter.set_ylabel("Humedad (%)")
    ax_scatter.legend()
    st.pyplot(fig_scatter)

    st.subheader("⚖️ Coeficientes del modelo (modelo.coef_)")
    
    # Tabla con los coeficientes
    df_coefs = pd.DataFrame(
        {
            "Variable": variables,
            "Coeficiente": [round(c, 3) for c in modelo.coef_[0]],
        }
    )
    st.dataframe(df_coefs, hide_index=True)

    coefs = pd.Series(modelo.coef_[0], index=variables).sort_values(
        key=abs, ascending=False
    )
    fig_coef, ax_coef = plt.subplots(figsize=(5, 2))
    coefs.plot(kind="barh", color="steelblue", ax=ax_coef)
    ax_coef.set_xlabel("Coeficiente (peso en el modelo)")
    ax_coef.invert_yaxis()
    st.pyplot(fig_coef)

st.divider()

# ----------------------------------------------------------------------
# 5. Métricas de evaluación
# ----------------------------------------------------------------------
st.subheader("📊 Desempeño del modelo (conjunto de prueba)")

col3, col4 = st.columns([1, 1])

with col3:
    matriz = confusion_matrix(y_test, pred_test)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(
        matriz, display_labels=["No llueve", "Sí llueve"]
    )
    disp.plot(cmap="Blues", values_format="d", ax=ax_cm, colorbar=False)
    st.pyplot(fig_cm)

with col4:
    exactitud = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test, zero_division=0)
    recall = recall_score(y_test, pred_test, zero_division=0)

    vn, fp, fn, vp = matriz.ravel()
    dias_lluvia_pred = pred_test.sum()
    total_dias = len(pred_test)

    st.metric("Días predichos con lluvia", f"{dias_lluvia_pred} de {total_dias}")
    st.metric("Accuracy", f"{exactitud:.2%}")
    st.metric("Precisión", f"{precision:.2%}")
    st.metric("Recall", f"{recall:.2%}")
    st.write(f"**Falsos positivos (FP):** {fp}  |  **Falsos negativos (FN):** {fn}")

    if fp > fn:
        st.info(
            "Con este umbral y estas variables, el modelo comete más **falsos positivos** (falsas alarmas de lluvia)."
        )
    elif fn > fp:
        st.info(
            "Con este umbral y estas variables, el modelo comete más **falsos negativos** (lluvias no detectadas)."
        )
    else:
        st.info(
            "El modelo comete la misma cantidad de falsos positivos y falsos negativos."
        )

st.divider()

# ----------------------------------------------------------------------
# 6. Soluciones a la Actividad 14
# ----------------------------------------------------------------------
st.markdown("### 🧠 Respuestas a las preguntas de la Actividad 14")

with st.expander("Desplegar respuestas interpretadas", expanded=True):
    st.markdown(
        f"""
    1. **¿Qué pasa si eliminan la variable `viento`?**  
       Al desmarcar *viento*, **Accuracy, Precisión y Recall permanecen prácticamente iguales** (variación menor al 1%). Esto sucede porque el peso del viento en la ecuación de probabilidad es sumamente bajo en comparación con la humedad y la temperatura.

    2. **¿Qué pasa si cambian el umbral a 0.7?**  
       El modelo exige un 70% de certeza para declarar lluvia, por lo que **predice MENOS días de lluvia** ({dias_lluvia_pred} días actualmente). Esto aumenta los **Falsos Negativos** (lluvias no detectadas).

    3. **¿Cuál variable tiene más peso en la predicción?**  
       Revisando la tabla de `modelo.coef_`, la variable con el coeficiente con mayor valor absoluto es **`humedad`**, lo que indica que es el factor determinante para predecir si lloverá.

    4. **Falsos Positivos vs. Falsos Negativos (Umbral 0.5):**  
       * **Falso Positivo (FP = {fp}):** El modelo predice lluvia pero hace buen tiempo. *En la vida real:* Llevas paraguas innecesariamente.  
       * **Falso Negativo (FN = {fn}):** El modelo predice día seco pero termina lloviendo. *En la vida real:* Saliste confiado y te terminas mojando bajo la lluvia.
    """
    )
