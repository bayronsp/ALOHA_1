import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import base64

# Verificar si el archivo existe en el directorio
file_path = 'reglamento.xlsx'

if not os.path.isfile(file_path):
    st.error(f"El archivo {file_path} no se encontró.")
else:
    # Cargar el archivo Excel
    @st.cache_data
    def cargar_datos():
        return pd.read_excel(file_path, usecols="A:B", names=['Pregunta', 'Respuesta'])

    datos = cargar_datos()

    # Inicializar el vectorizador TF-IDF y ajustar con las preguntas
    @st.cache_resource
    def inicializar_vectorizador():
        vectorizer = TfidfVectorizer().fit(datos['Pregunta'])
        return vectorizer

    vectorizer = inicializar_vectorizador()

    # Función para obtener la respuesta basada en la pregunta del usuario
    def obtener_respuesta(pregunta):
        pregunta_vector = vectorizer.transform([pregunta])
        pregunta_tfidf = vectorizer.transform(datos['Pregunta'])
        similitudes = cosine_similarity(pregunta_vector, pregunta_tfidf).flatten()
        idx_max = similitudes.argmax()
        if similitudes[idx_max] > 0.1:  # Umbral para considerar una respuesta
            return datos['Respuesta'].iloc[idx_max]
        else:
            return "Lo siento, no encontré una respuesta en el reglamento."

    # Configuración de la interfaz de Streamlit
    st.title("ALOHA Virtual")
    st.write("Bienvenido al Chatbot ALOHA Virtual, ¿en qué puedo ayudarte hoy?")

    # Function to encode image to base64
    def img_to_bytes(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Encode the image to base64
    img_bytes = img_to_bytes("robot.png")
    img_html = f'<img src="data:image/png;base64,{img_bytes}" style="height: 50px; width: 50px;">'

    # Use st.markdown to create a button with an image
    if st.markdown(f'<button>{img_html}</button>', unsafe_allow_html=True):
        # Mostrar el chat
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []  # Lista para almacenar el historial de mensajes

        def mostrar_mensaje(mensaje, es_usuario=False):
            # Agregar mensaje al historial
            st.session_state.chat_history.append((mensaje, es_usuario))

            # Mostrar el historial de mensajes
            for mensaje, es_usuario in st.session_state.chat_history:
                if es_usuario:
                    st.chat_message(mensaje, is_user=True)
                else:
                    st.chat_message(mensaje)

        # Entrada del usuario y botón para enviar la pregunta
        pregunta = st.text_input("Escribe tu pregunta:").lower()
        
        if st.button("Enviar"):
            if pregunta == 'fin':
                st.write("Chat finalizado.")
            else:
                # Mostrar la pregunta del usuario
                mostrar_mensaje(pregunta, es_usuario=True)

                # Obtener la respuesta
                respuesta = obtener_respuesta(pregunta)

                # Mostrar la respuesta
                mostrar_mensaje(respuesta)
