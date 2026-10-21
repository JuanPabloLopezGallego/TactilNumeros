import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import pandas as pd

st.set_page_config(
    page_title="Detector de Dígitos MNIST",
    page_icon="✍️",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════
# ESTILOS — CYBER SKETCHPAD
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg:         #09090b;
        --surface:    #18181b;
        --border:     #27272a;
        --text:       #fafafa;
        --text-muted: #a1a1aa;
        --accent:     #d946ef; /* Fucsia Neón */
        --accent-2:   #8b5cf6; /* Violeta Neón */
    }

    html, body, [class*="css"], .stApp, button, input, textarea {
        font-family: 'Inter', sans-serif !important;
        color: var(--text) !important;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* ═══ FONDO ═══ */
    .stApp {
        background-color: var(--bg) !important;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.08), transparent 50%),
            radial-gradient(circle at 100% 100%, rgba(217, 70, 239, 0.05), transparent 40%);
        background-attachment: fixed;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stMarkdown p { font-family: 'Space Grotesk', sans-serif !important; font-size: 0.9rem; }

    /* ═══ HERO ═══ */
    .hero {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .hero h1 {
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #fff 30%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 1.1rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* ═══ CANVAS BOX (Estilo aplicado directamente al canvas) ═══ */
    [data-testid="stCanvas"] {
        background: #000000;
        border: 2px solid var(--border);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.1), inset 0 0 20px rgba(0,0,0,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        transition: border-color 0.3s ease;
    }
    [data-testid="stCanvas"]:hover {
        border-color: var(--accent);
    }
    [data-testid="stCanvas"] canvas {
        border-radius: 12px !important;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1) !important;
    }

    /* ═══ BOTONES ═══ */
    .stButton > button {
        width: 100%;
        border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.8rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }
    
    /* Botón Predecir */
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(217, 70, 239, 0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #e879f9 !important;
        box-shadow: 0 0 25px rgba(217, 70, 239, 0.6) !important;
        transform: translateY(-2px);
    }
    
    /* Botón Limpiar */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border: 1px solid var(--border) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--text) !important;
        color: var(--text) !important;
    }

    /* ═══ RESULTADOS ═══ */
    .result-hero {
        text-align: center;
        background: linear-gradient(180deg, rgba(24,24,27,0) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.5rem 1rem;
        margin-bottom: 2rem;
    }
    .result-digit {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 6rem;
        font-weight: 800;
        line-height: 1;
        color: var(--accent);
        text-shadow: 0 0 40px rgba(217, 70, 239, 0.5);
        margin: 0;
    }
    .result-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    /* ═══ MÉTRICAS ═══ */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        padding: 1.25rem !important;
        border-radius: 12px !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent-2) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2rem !important;
    }

    /* ═══ EXPANDER Y ALERTAS ═══ */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR (INFO DEL MODELO)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Info del Sistema")
    st.write(f"**TensorFlow:** `{tf.__version__}`")
    st.write(f"**Keras:** `{keras.__version__}`")
    st.markdown("---")
    st.info("Este modelo fue entrenado en Google Colab usando el dataset MNIST (dígitos del 0 al 9).")

# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
    <div class="hero">
        <h1>✍️ Detector de Dígitos</h1>
        <p>Dibuja un número del 0 al 9 en el lienzo y la IA adivinará cuál es.</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_mnist_model():
    model_files = ['mnist_model.keras', 'mnist_model.h5']
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                return model
            except Exception as e:
                st.error(f"❌ Error al cargar {model_path}: {str(e)}")
                continue
    
    st.error("❌ No se encontró el modelo `mnist_model.keras` o `.h5`.")
    st.info("Asegúrate de subir el archivo del modelo a la raíz de tu repositorio.")
    st.stop()

model = load_mnist_model()

# ═══════════════════════════════════════════════════════════════
# LAYOUT: CANVAS Y CONTROLES
# ═══════════════════════════════════════════════════════════════
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.markdown("#### 🎨 Lienzo")
    # Al quitar el with st.container(key="canvas_box"), el canvas recibe el CSS directamente
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("#### 🎮 Controles")
    st.write("Dibuja el número claramente en el centro del lienzo.")
    
    predict_btn = st.button("🔍 Predecir Dígito", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Limpiar Lienzo", type="secondary", use_container_width=True)
    
    if clear_btn:
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PREDICCIÓN Y RESULTADOS
# ═══════════════════════════════════════════════════════════════
if predict_btn:
    if canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) == 0:
            st.warning("⚠️ El lienzo está vacío. Dibuja un número primero.")
        else:
            with st.spinner("🤔 Analizando tu dibujo..."):
                # Procesar imagen
                image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Predicción
                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                alternative = np.argsort(prediction[0])[-2]
            
            st.markdown("---")
            
            # 1. Resultado Principal
            st.markdown(f"""
                <div class="result-hero">
                    <div class="result-label">Predicción del Modelo</div>
                    <div class="result-digit">{digit}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # 2. Métricas
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Dígito", digit)
            with col_m2:
                st.metric("Confianza", f"{confidence:.1f}%")
            with col_m3:
                st.metric("2ª Opción", alternative)
            
            st.markdown("---")
            
            # 3. Imágenes del procesamiento
            st.markdown("#### 🖼️ Procesamiento de Imagen")
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown("**Original (Dibujo)**")
                st.image(canvas_result.image_data, use_container_width=True)
            
            with col_img2:
                st.markdown("**Procesada (28x28 px)**")
                st.image(image_resized, use_container_width=True)
            
            # 4. Gráfico de barras
            st.markdown("#### 📊 Probabilidades por dígito")
            prob_df = pd.DataFrame({
                'Dígito': [str(i) for i in range(10)],
                'Probabilidad (%)': prediction[0] * 100
            })
            st.bar_chart(prob_df.set_index('Dígito'), color="#d946ef")
            
            # 5. Detalles exactos
            with st.expander("🔍 Ver porcentajes exactos"):
                for i, prob in enumerate(prediction[0]):
                    emoji = "🎯" if i == digit else "▫️"
                    st.write(f"{emoji} **Dígito {i}**: `{prob*100:.2f}%`")
    else:
        st.warning("⚠️ No se detectó ningún trazo en el lienzo.")
