import os
import openai
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from openai import OpenAI  # Importación del cliente OpenAI
import numpy as np
from umap import UMAP

# Configura tu clave de API de OpenAI desde un archivo
with open('C:/ProyectoChatOpenAI/Scripts/Api_key.txt') as f: 
    api_key = f.read().strip()

# Inicializa el cliente OpenAI
client = OpenAI(api_key=api_key)


# Función para generar embeddings con OpenAI
def generate_openai_embeddings(texts, model="text-embedding-ada-002"):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# Configuración de la barra lateral
with st.sidebar:
    st.title('🤖🧠 Modelos OpenAI')
    model_selection = st.selectbox('Selecciona el modelo', ['text-davinci-003', 'gpt-4'])
    embedding_model = st.selectbox('Modelo de Embedding', ['text-embedding-ada-002'])
    start_button = st.button('Iniciar')

# Carga del dataset
input_datapath = "C:/ProyectoChatOpenAI/Recomendacion/embedded_1k_reviews.csv"  # Ajusta la ruta a tu dataset
df = pd.read_csv(input_datapath, index_col=0)
st.write("## Vista del Dataset")
st.write(df.head(10))

# Generación de embeddings con OpenAI
st.write("## Generando Embeddings con OpenAI")
texts = df['reviews.text']  # Columna que contiene textos
matrix = generate_openai_embeddings(texts, model=embedding_model)

# Visualización con PCA
st.title("Visualización con PCA")
pca = PCA(n_components=2)
components = pca.fit_transform(matrix)
total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter(
    components, x=0, y=1, color=df['reviews.rating'].values,
    color_continuous_scale=px.colors.qualitative.Prism,
    title=f'PCA (2D) - Varianza Explicada Total: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2'}
)
st.plotly_chart(fig)

pca = PCA(n_components=3)
components = pca.fit_transform(matrix)
total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=df['reviews.rating'].values,
    color_continuous_scale=px.colors.qualitative.Prism,
    title=f'PCA (3D) - Varianza Explicada Total: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
st.plotly_chart(fig)

# Visualización con t-SNE
st.title("Visualización con t-SNE")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(matrix)
fig = px.scatter(
    x=tsne_results[:, 0], y=tsne_results[:, 1], color=df['reviews.rating'].values,
    color_continuous_scale=px.colors.qualitative.Prism,
    title="Visualización con t-SNE"
)
st.plotly_chart(fig)

# Visualización con UMAP
st.title("Visualización con UMAP")
umap = UMAP(n_components=2, random_state=42)
umap_results = umap.fit_transform(matrix)
fig = px.scatter(
    x=umap_results[:, 0], y=umap_results[:, 1], color=df['reviews.rating'].values,
    color_continuous_scale=px.colors.qualitative.Prism,
    title="Visualización con UMAP"
)
st.plotly_chart(fig)

##Chat Bot pruebas
# Función de recomendación del chatbot
def chatbot_recommendation(user_input, embeddings, model="text-davinci-003"):
    """
    Genera una recomendación utilizando el modelo de OpenAI.
    """
    prompt = (
        f"Tengo un dataset con reseñas de productos y sus puntuaciones. "
        f"Con base en los datos, sugiéreme productos relacionados o patrones que puedan ser útiles.\n\n"
        f"Historial del usuario: {user_input}\n\n"
        f"Embeddings calculados: {embeddings[:5].tolist()} (muestra de 5 puntos).\n\n"
        "¿Qué recomendaciones puedes darme?"
    )

    try:
        # Realiza la solicitud a la API de OpenAI
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7
        )
        # Extrae el texto de la respuesta
        return response['choices'][0]['text'].strip()
    except Exception as e:
        return f"Error al comunicarse con OpenAI: {e}"

# Interfaz de usuario para el chatbot
st.title("🤖 Chatbot de Recomendación de Productos")

# Campo de texto donde el usuario puede ingresar su solicitud
user_input = st.text_input("¿Sobre qué producto te gustaría obtener recomendaciones?", "")

# Cuando el usuario envía una solicitud
if user_input:
    # Filtrar las primeras 10 filas del dataset
    df_top_10 = df.head(10)

    # Generación de embeddings con OpenAI para las primeras 10 filas
    st.write("Generando Embeddings con OpenAI para las primeras 10 filas...")
    texts = df_top_10['reviews.text']  # Columna que contiene textos
    matrix_top_10 = generate_openai_embeddings(texts, model=embedding_model)

    # Genera recomendaciones con base en el input del usuario y los embeddings de las primeras 10 filas
    st.write("Generando recomendaciones...")
    recommendation = chatbot_recommendation(user_input, matrix_top_10)

    st.write("### Recomendación:")
    st.write(recommendation)



