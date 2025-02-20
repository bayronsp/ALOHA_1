import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

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

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Lista para almacenar el historial de mensajes

    # Mostrar el historial de mensajes
    for mensaje, es_usuario in st.session_state.chat_history:
        if es_usuario:
            st.write(f"**Usuario:** {mensaje}")
        else:
            st.write(f"**Bot:** {mensaje}")

    # Entrada del usuario y botón para enviar la pregunta
    pregunta = st.text_input("Escribe tu pregunta:")
    
    if st.button("Enviar"):
        if pregunta.lower() == 'fin':
            st.write("Chat finalizado.")
        else:
            # Mostrar la pregunta del usuario
            st.session_state.chat_history.append((pregunta, True))

            # Obtener la respuesta
            respuesta = obtener_respuesta(pregunta)

            # Mostrar la respuesta
            st.session_state.chat_history.append((respuesta, False))
            st.query_params(updated="true")
