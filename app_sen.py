from shiny import App, ui, render, reactive
import fitz  # PyMuPDF
import requests
from io import BytesIO
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import time

# Charger le modèle NLP
model = SentenceTransformer("all-MiniLM-L6-v2")

# Télécharger et lire les PDFs depuis GitHub
def lire_pdfs_depuis_github(urls):
    texte_total = []
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Ouvrir le PDF directement depuis le contenu téléchargé
                doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
                for page in doc:
                    texte_total.append(page.get_text("text"))
            else:
                print(f"Erreur lors du téléchargement : {url}")
        except Exception as e:
            print(f"Erreur avec {url} : {e}")
    return texte_total

# Liste des liens vers les PDF hébergés sur GitHub
pdf_urls = [
    "https://raw.githubusercontent.com/mahmoudthiam/IATech_Senegal2050/main/Strategie-Nationale-de-Developpement-2025-2029.pdf"
]

# Charger les documents
documents = lire_pdfs_depuis_github(pdf_urls)
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
        "content": "Bonjour ! Je suis un assistant IA. Posez-moi des questions sur la vision SENEGAL 2050",
        "sender": "bot",
        "time": datetime.now().strftime("%H:%M")
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
            "time": datetime.now().strftime("%H:%M")
        }
        messages.set(messages() + [new_user_message])
        ui.update_text("question", value="")

        # Obtenir et ajouter la réponse du bot
        response = repondre_question(question)
        new_bot_message = {
            "content": response,
            "sender": "bot",
            "time": datetime.now().strftime("%H:%M")
        }
        messages.set(messages() + [new_bot_message])

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
                    ui.div(
                        msg["time"],
                        class_="timestamp"
                    ),
                    style="display: flex; flex-direction: column;"
                )
                for msg in messages()
            ]
        )

# App
app = App(app_ui, server)
