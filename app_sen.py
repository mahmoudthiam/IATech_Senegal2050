from shiny import App, ui, render, reactive
import fitz  # PyMuPDF
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import speech_recognition as sr
import pyttsx3  # Lecteur vocal

# Charger le modèle NLP
model = SentenceTransformer("all-MiniLM-L6-v2")

# Lire les PDFs
def lire_pdfs(dossier):
    texte_total = []
    for fichier in os.listdir(dossier):
        if fichier.endswith(".pdf"):
            chemin = os.path.join(dossier, fichier)
            doc = fitz.open(chemin)
            for page in doc:
                texte_total.append(page.get_text("text"))
    return texte_total

# Charger les documents
dossier_pdfs = "C:/Users/thiam/Desktop/chatGPT/IA_SENEGALVISION2050"
documents = lire_pdfs(dossier_pdfs)
texte_corpus = " ".join(documents)

# Encoder les documents
doc_embeddings = model.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]

# Index FAISS
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Fonction de réponse
def repondre_question(question):
    try:
        documents_list = texte_corpus.split(". ")
        vect = TfidfVectorizer()
        tfidf_matrix = vect.fit_transform(documents_list + [question])
        scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        top_indices = np.argsort(scores[0])[-20:][::-1]  # Sélection des 50 meilleures réponses
        meilleure_reponse = ".\n".join([documents_list[i] for i in top_indices]) + '.'
        return meilleure_reponse
    except Exception as e:
        return "Je n'ai pas compris la question."

# Fonction pour la reconnaissance vocale
def reconnaitre_vocal():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Dites quelque chose...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='fr-FR')
            print("Vous avez dit: " + text)
            return text
        except sr.UnknownValueError:
            print("Je n'ai pas pu comprendre")
            return None
        except sr.RequestError:
            print("Erreur de la demande à l'API de reconnaissance vocale")
            return None

# Fonction de synthèse vocale
def lire_reponse_vocale(reponse):
    engine = pyttsx3.init()
    # Configuration de la vitesse et de la voix
    engine.setProperty('rate', 150)  # Vitesse de la parole
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Choisir une voix (0 = Masculine, 1 = Féminine)
    engine.say(reponse)
    engine.runAndWait()

# Interface Shiny
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .chat-container {
                max-width: 900px;
                margin: 0 auto;
                height: 80vh;
                display: flex;
                flex-direction: column;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .chat-header {
                background-color: #f9fafb;
                padding: 16px;
                border-bottom: 1px solid #e5e7eb;
                text-align: center;
                font-weight: bold;
            }
            .chat-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                background-color: #ffffff;
            }
            .message {
                margin-bottom: 16px;
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 18px;
                line-height: 1.5;
            }
            .user-message {
                background-color: #3b82f6;
                color: white;
                margin-left: auto;
                border-bottom-right-radius: 4px;
            }
            .bot-message {
                background-color: #f3f4f6;
                color: #111827;
                margin-right: auto;
                border-bottom-left-radius: 4px;
            }
            .input-area {
                padding: 16px;
                background-color: #f9fafb;
                border-top: 1px solid #e5e7eb;
                display: flex;
                gap: 8px;
            }
            .input-text {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #e5e7eb;
                border-radius: 24px;
                outline: none;
            }
            .input-text:focus {
                border-color: #3b82f6;
            }
            .send-button {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 24px;
                padding: 12px 24px;
                cursor: pointer;
                font-weight: bold;
            }
            .send-button:hover {
                background-color: #2563eb;
            }
            .timestamp {
                font-size: 0.75rem;
                color: #6b7280;
                margin-top: 4px;
                text-align: right;
            }
            .icon {
                cursor: pointer;
                padding: 5px;
                background-color: #3b82f6;
                color: white;
                border-radius: 50%;
                margin-top: 10px;
            }
            .icon:hover {
                background-color: #2563eb;
            }
            .voice-icon {
                display: inline-block;
                margin-left: 10px;
                cursor: pointer;
            }
        """)
    ),
    ui.div(
        ui.div(
            ui.div("Assistant IA", class_="chat-header"),
            ui.div(
                ui.output_ui("chat_messages"),
                id="chat-messages",
                class_="chat-messages"
            ),
            ui.div(
                ui.input_text("question", "", placeholder="Envoyez un message..."),
                ui.input_action_button("send", "Envoyer"),
                ui.input_action_button("voice_input", "🎤", class_="icon"),  # Icône pour la reconnaissance vocale
                class_="input-area"
            ),
            class_="chat-container"
        ),
        style="padding: 20px; height: 100vh; background-color: #f3f4f6;"
    )
)

# Backend
def server(input, output, session):
    messages = reactive.Value([{
        "content": "Bonjour ! Je suis un assistant IA. Posez-moi vos question sur la vision SENEGAL 2050.",
        "sender": "bot",
        "time": datetime.now().strftime("%H_%M_%S")  # Ajout des secondes pour plus de granularité
    }])

    @reactive.Effect
    @reactive.event(input.send)
    def handle_message():
        question = input.question()
        if not question.strip():
            return

        # Ajouter le message de l'utilisateur
        new_user_message = {
            "content": question,
            "sender": "user",
            "time": datetime.now().strftime("%H_%M_%S")  # Ajout des secondes pour plus de granularité
        }
        messages.set(messages() + [new_user_message])
        ui.update_text("question", value="")

        # Obtenir et ajouter la réponse du bot
        response = repondre_question(question)
        new_bot_message = {
            "content": response,
            "sender": "bot",
            "time": datetime.now().strftime("%H_%M_%S"),  # Ajout des secondes pour plus de granularité
            "voice_icon": True  # Ajouter un champ pour indiquer que le bot a une icône de voix
        }
        messages.set(messages() + [new_bot_message])

    @reactive.Effect
    @reactive.event(input.voice_input)
    def voice_input_event():
        # Reconnaître la question via la voix
        question = reconnaitre_vocal()
        if question:
            new_user_message = {
                "content": question,
                "sender": "user",
                "time": datetime.now().strftime("%H_%M_%S")  # Ajout des secondes pour plus de granularité
            }
            messages.set(messages() + [new_user_message])

            # Obtenir et ajouter la réponse du bot
            response = repondre_question(question)
            new_bot_message = {
                "content": response,
                "sender": "bot",
                "time": datetime.now().strftime("%H_%M_%S"),  # Ajout des secondes pour plus de granularité
                "voice_icon": True  # Ajouter un champ pour indiquer que le bot a une icône de voix
            }
            messages.set(messages() + [new_bot_message])

    @reactive.Effect
    @reactive.event(input.voice_icon)
    def play_audio():
        # Lancer la lecture vocale de la réponse du bot
        message_id = input.voice_icon()  # ID unique de l'icône
        message = next((msg for msg in messages() if f"voice_icon_{msg['time']}" == message_id), None)
        if message:
            lire_reponse_vocale(message['content'])  # Lire le message du bot

    @output
    @render.ui
    def chat_messages():
        return ui.TagList(
            *[
                ui.div(
                    ui.div(
                        msg["content"],
                        class_="message " + ("user-message" if msg["sender"] == "user" else "bot-message")
                    ),
                    # Suppression de l'heure
                    # ui.div(
                    #    msg["time"],
                    #    class_="timestamp"
                    # ),
                    # Si le message est du bot, ajouter un bouton de voix
                    ui.div(
                        ui.input_action_button(f"voice_icon_{msg['time']}", "🔊", class_="voice-icon") if msg["sender"] == "bot" else "",
                        style="display: flex; justify-content: flex-end;"
                    ),
                    style="display: flex; flex-direction: column;"
                )
                for msg in messages()
            ]
        )

# App
app = App(app_ui, server)
