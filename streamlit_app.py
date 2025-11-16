import streamlit as st
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Obtiene la llave (local o nube)
API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

client = Groq(api_key=API_KEY)


def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "customer_reviews.csv")
    return csv_path

@st.cache_data
def analyze_sentiment(text):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": f"Analiza la siguiente reseña de un cliente que está entre comillas sencillas triples y determina \
            si el sentimiento es positivo, negativo o neutral. Solo responde con la palabra correspondiente en ingles y minusculas: ''''{text}'''" }],
        temperature=0,
        max_completion_tokens=500,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )
    response = []
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return "".join(response)


st.title("Bienvenido al análisis d esentimientos")
st.write("Esta aplicación procesa datos con tecnología GenAI.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Cargar el conjunto datos"):
        try:
            csv_path = get_dataset_path()
            st.session_state['df'] = pd.read_csv(csv_path).head(20)
            st.success("Conjunto de datos cargados existosamente!")
        except FileNotFoundError:
            st.error("Conjunto de datos no encontradol. Por favor \
                verifica la ruta del archivo.")

with col2:
    if st.button("Análisis de sentimientos"):
        if "df" in st.session_state:
            st.session_state["df"]["ANALYZE_SENTIMENT"] = st.session_state["df"]["SUMMARY"].apply(analyze_sentiment)
            st.success("Análisis de sentimientos completado!")
        else:
            st.warning("Por favor cargar el conjunto de datos primero.")
            
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filtrar por producto")
    product = st.selectbox("Choose a product", ["Todos los productos"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reseñas para {product}")

    if product != "Todos los productos":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)
    if "df" in st.session_state and "ANALYZE_SENTIMENT" in st.session_state["df"].columns:
        st.subheader(f"Desglose de sentimientos para  {product}")
        grouped = st.session_state["df"].groupby(["PRODUCT", "ANALYZE_SENTIMENT"]).size().reset_index(name="QUANTITY")
        st.bar_chart(
            grouped,
            x="PRODUCT",
            y="QUANTITY",
            color="ANALYZE_SENTIMENT"
        ) 
    else:
        st.info("Carga el dataset y ejecuta el análisis primero.")
