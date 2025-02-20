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

    # Inicializar el historial de chat en la sesión
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Lista para almacenar el historial de mensajes

    # Inicializar el input del usuario en session_state
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""  # Campo vacío inicial

    # Mostrar el historial de mensajes
    for mensaje, es_usuario in st.session_state.chat_history:
        if es_usuario:
            st.write(f"**Usuario:** {mensaje}")
        else:
            st.write(f"**Bot:** {mensaje}")

    # Formulario para enviar la pregunta
    with st.form(key="form_pregunta"):
        pregunta = st.text_input("Escribe tu pregunta:", value=st.session_state.user_input, key="user_input")
        enviar = st.form_submit_button("Enviar")  # Botón para enviar

        if enviar:
            if pregunta.strip():  # Verificar que no esté vacío
                if pregunta.lower() == 'fin':
                    st.write("Chat finalizado.")
                else:
                    # Agregar la pregunta del usuario al historial
                    st.session_state.chat_history.append((pregunta, True))

                    # Obtener la respuesta del bot
                    respuesta = obtener_respuesta(pregunta)

                    # Agregar la respuesta al historial
                    st.session_state.chat_history.append((respuesta, False))
