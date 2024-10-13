import customtkinter as ctk
from PIL import Image, ImageTk
import pyglet
from tkinter import font as tkfont
from langchain_google_community import TextToSpeechTool
import pygame
from langchain_google_community import SpeechToTextLoader
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ReadOnlySharedMemory
from threading import Thread
from functools import partial
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os

QA_GENERATION_TEMPLATE = """Task: Genera una risposta completa per rispondere a {question} usando TUTTI i risultati contenuti in {context}, senza escluderne nessuno
    """
CYPHER_USER_GENERATION_TEMPLATE = """Task: In base alla seguente domanda: {question} stabilisci se esprime una preferenza o no.
    Consideriamo come preferenza tutte le volte in cui l'utente chiede un consiglio su un film, su un attore o su un genere.
    Esempio di preferenza: Suggeriscimi un film di [attore]. Consigliami un film di [attore]. Consigliami un film [genere] etc...
    Se viene espressa una preferenza: crea una query Cypher per interrogare un database grafico, rispondi solo con la query generata senza aggiungere altri commenti o #.
    esempio di query:
    #Consigliami un film di Tom Hanks
    MATCH (u:User)
    WHERE u.name = "User"
    MATCH (p:Person)
    WHERE p.name = "Tom Hanks"
    MERGE (u)-[r:LIKES]->(p)
    ON CREATE SET r.count = 1
    ON MATCH SET r.count = r.count + 1

    #Consigliami un film d'Azione  
    MATCH (u:User)
    WHERE u.name = "User"
    MATCH (g:Genre)
    WHERE g.name = "Action"
    MERGE (u)-[r:LIKES]->(g)
    ON CREATE SET r.count = 1
    ON MATCH SET r.count = r.count + 1

    I generi sono: 
    Adventure, Animation, Children, Comedy, Fantasy, Romance, Drama, Action, Crime, Thriller, Horror, Mystery, Sci-Fi, Documentary, IMAX, War, Musical, Western, Film-Noir.

    Se non esprime una preferenza: rispondi NO
    """
CYPHER_GENERATION_TEMPLATE = """Task:Continua la seguente conversazione dove Human rappresentano le mie domande e AI rappresentano le tue risposte: {chat_history}
    Human: {question}
    I messaggi più in basso sono i più recenti, quindi i più importanti.
    Devi rispondere generando una dichiarazione Cypher per interrogare un database grafico.
    Le entità sono: Actor, Director, Genre, Movie, Person, User
    Le relazioni sono: ACTED_IN, DIRECTED, IN_GENRE, LIKES
    Attributi di Movie: budget, countries, imdbRating, languages, plot, poster, released, runtime, title, year
    Attributi di Person, Actor, Director: name, url, bio, born, bornIn, died, poster

Esempi:Ecco alcuni esempi di dichiarazioni Cypher generate per domande specifiche:

    # Quanti attori hanno recitato in Top Gun?
    MATCH (m:Movie {{title:"Top Gun"}})-[:ACTED_IN]-(person:Person)
    RETURN count(person) AS QUANTI_ATTORI_HANNO_RECITATO_IN_TOP_GUN

    # Chi ha recitato in Film?
    MATCH (p:Person)-[:ACTED_IN]-(m:Movie)
    WHERE m.title = "Film"
    RETURN p.name AS RECITATO_IN_FILM

    # Film con Attore?
    MATCH (m:Movie)-[:ACTED_IN]-(p:Person)
    WHERE p.name = "Attore"
    RETURN m.title AS QUALI_FILM_CON_ATTORE

    #Film di Regista?
    MATCH (m:Movie)-[:DIRECTED]-(p:Person)
    WHERE p.name = "Regista"
    RETURN m.title AS QUALI_FILM_DIRETTI_DA_REGISTA

    #Chi ha diretto A League of Their Own?
    MATCH (m:Movie)-[:DIRECTED]-(p:Person)
    WHERE m.title = "A League of Their Own"
    RETURN p.name AS REGISTA_DI_A_LEAGUE_OF_THEIR_OWN

    #Di cosa parla Bicentennial Man?
    MATCH (m:Movie)
    WHERE m.title="Bicentennial Man"
    RETURN m.tagline AS DI_COSA_PARLA_BICENTENNIAL_MAN

    #In quanti film ha recitato Tom Hanks?
    MATCH (p:Person)-[:ACTED_IN]-(m:Movie)
    WHERE p.name="Tom Hanks"
    RETURN count(m) AS QUANTI_FILM_HA_RECITATO_TOM_HANKS

    #Di che genere è Apollo 13?
    MATCH (m:Movie)-[:IN_GENRE]-(g:Genre)
    WHERE m.title = "Apollo 13"
    RETURN g.name AS GENERE_DI_APOLLO_13

    #Film del genere Azione
    #Suggerisci film di Azione
    MATCH (m:Movie)-[:IN_GENRE]-(g:Genre)
    WHERE g.name = "Action"
    RETURN m.title AS SUGGERIMENTI_AZIONE
    ORDER BY m.imdbRating DESC

    #Quali sono i film di James Cameron con l'attore Di Caprio?
    MATCH (m:Movie)-[:DIRECTED]-(p:Person), (m)-[:ACTED_IN]-(a:Person)
    WHERE p.name = "James Cameron" AND a.name = "Leonardo DiCaprio"
    RETURN DISTINCT m.title AS FILM_DI_JAMES_CAMERON_CON_DI_CAPRIO

    #Parlami di Attore
    #Chi è Attore
    MATCH (p:Person) WHERE p.name='Attore' return p.bio AS INFO_DI_TOM_HANKS

    #Film con Attore1 e Attore2
    MATCH (p1:Person), (p2:Person), (m:Movie)
    WHERE p1.name = "Attore1" AND p2.name = "Attore2" AND (p1)-[:ACTED_IN]-(m) AND (p2)-[:ACTED_IN]-(m)
    RETURN m.title AS FILM_CONDIVISI

    #Attore1 ha mai recitato con Attore2?
    MATCH (p1:Person), (p2:Person), (m:Movie)
    WHERE p1.name = "Attore1" AND p2.name = "Attore1" AND (p1)-[:ACTED_IN]-(m) AND (p2)-[:ACTED_IN]-(m)
    RETURN count(m) AS NUMERO_DI_FILM_CON_ATTORE1_E_ATTORE2

    #Con quali attori ha lavorato Di Caprio?
    MATCH (p1:Person)-[:ACTED_IN]-(m:Movie), (p2:Person)-[:ACTED_IN]-(m)
    WHERE p1.name = "Leonardo DiCaprio" AND p2.name <> "Leonardo DiCaprio"
    RETURN p2.name AS ATTORI_CHE_HANNO_RECITATO_CON_DI_CAPRIO
    ORDER BY m.imdbRating DESC

CONSIGLIO/SUGGERIMENTO:
Se viene richiesto un consiglio o un suggerimento su un film PARTENDO DALL'ATTORE, per esempio, CONSIGLIAMI UN FILM CON TOM HANKS, esegui questa query:
    MATCH (u:User)
    WHERE u.name = "User"
    OPTIONAL MATCH (u)-[r:LIKES]->(g1:Genre)
    WITH u, g1, COALESCE(SUM(r.count), 0) AS direct_count
    OPTIONAL MATCH (u)-[r:LIKES]->(p:Person)-[:ACTED_IN]->(m:Movie)-[:IN_GENRE]->(g2:Genre)
    WITH u, g1, direct_count, g2, COALESCE(SUM(r.count), 1) AS indirect_count
    WITH g1 AS genre, direct_count, SUM(indirect_count) AS total_indirect_count
    WITH genre.name AS genre_name, direct_count + total_indirect_count AS total_likes
    ORDER BY total_likes DESC
    LIMIT 1
    MATCH (m:Movie)-[:IN_GENRE]->(genre)
    WHERE genre.name = genre_name
    MATCH (p:Person)-[:ACTED_IN]->(m)
    WHERE p.name = "Tom Hanks"
    RETURN m.title AS TITOLO_FILM
    ORDER BY RAND()
    LIMIT 1

Se viene richiesto un consiglio o un suggerimento su un film PARTENDO DAL GENERE, per esempio: CONSIGLIAMI UN FILM D'AZIONE, esegui questa query:
    MATCH (u:User)
    WHERE u.name = "User"
    OPTIONAL MATCH (u)-[r:LIKES]->(p1:Person)
    WITH u, p1, COALESCE(SUM(r.count), 0) AS direct_count
    OPTIONAL MATCH (u)-[r:LIKES]-(g:Genre)-[:IN_GENRE]-(m:Movie)-[:ACTED_IN]-(p2:Person)
    WITH u, p1, direct_count, p2, COALESCE(SUM(r.count), 1) AS indirect_count
    WITH p1 AS actor, direct_count, SUM(indirect_count) AS total_indirect_count
    WITH actor.name AS actor_name, direct_count + total_indirect_count AS total_likes
    ORDER BY total_likes DESC
    LIMIT 1
    MATCH (m:Movie)-[:ACTED_IN]-(actor)
    WHERE actor.name = actor_name
    MATCH (g:Genre)-[:IN_GENRE]-(m)
    WHERE g.name = "Action"
    RETURN m.title AS TITOLO_FILM
    ORDER BY RAND()
    LIMIT 1


    Se viene richiesta la trama/descrizione/di cosa parla, RETURN m.plot
    Se viene richiesto il voto di un film, RETURN m.imdbRating
    Quando viene richiesto un genere usa [:IN_GENRE] fai il RETURN e ordina in base al voto se presente
    AS [nome], [nome] non può contenere spazi all'interno per esempio:


    I generi sono: 
    Adventure, Animation, Children, Comedy, Fantasy, Romance, Drama, Action, Crime, Thriller, Horror, Mystery, Sci-Fi, Documentary, IMAX, War, Musical, Western, Film-Noir.




    La risposta è:
    {question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "chat_history"], template=CYPHER_GENERATION_TEMPLATE
)
QA_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "context", "chat_history"], template=QA_GENERATION_TEMPLATE
)
CYPHER_USER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question"], template=CYPHER_USER_GENERATION_TEMPLATE
)

api_key = 'api_key'
project_id = "project_id"
sample_rate = 16000  # Campionamento a 16 kHz
channels = 1  # Numero di canali
tts = TextToSpeechTool()
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
memory = ConversationBufferMemory(memory_key="chat_history", k=2)
readonlymemory = ReadOnlySharedMemory(memory=memory)
chain = GraphCypherQAChain.from_llm(
    cypher_llm=GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key), graph=graph, verbose=True,
    cypher_llm_kwargs={"prompt": CYPHER_GENERATION_PROMPT, "memory": readonlymemory, "verbose": False},
    qa_llm=GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key),
    qa_llm_kwargs={"prompt": QA_GENERATION_PROMPT, "memory": readonlymemory}, memory=memory
)
chain_user = GraphCypherQAChain.from_llm(
    GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key), graph=graph, verbose=True,
    cypher_llm_kwargs={"prompt": CYPHER_USER_GENERATION_PROMPT, "verbose": False}
)

# Configurazione di base
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Chatbot con Registrazione Audio")
root.geometry("450x700")
root.resizable(False, False)
root.config(bg="#584CD7")
root.configure(background="#584CD7")

# Carica l'immagine
header_img_path = ("./assets/topchat.png")  # Inserisci il percorso dell'immagine
header_img = Image.open(header_img_path)
window_width = 562

# Ridimensiona l'immagine alla larghezza della finestra mantenendo le proporzioni
aspect_ratio = header_img.width / header_img.height
new_height = int(window_width / aspect_ratio)
header_img_resized = header_img.resize((window_width, new_height), Image.LANCZOS)
header_img_tk = ImageTk.PhotoImage(header_img_resized)



# Crea un widget Label per visualizzare l'immagine
header_label = ctk.CTkLabel(root, image=header_img_tk, bg_color="#584CD7", fg_color="#584CD7", text="")
header_label.pack(side="top", pady=0)

# Frame principale che contiene la scrollbar e le chat bubbles
chat_frame = ctk.CTkFrame(root, fg_color="#584CD7", bg_color="#584CD7", border_color="#584CD7")
chat_frame.pack(pady=20, padx=0, fill=ctk.BOTH, expand=True)
chat_frame.configure(fg_color="#584CD7", border_color="#584CD7")

# Canvas e scrollbar
canvas = ctk.CTkCanvas(chat_frame, bg="#584CD7", highlightthickness=0)
canvas.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)
canvas.configure(bg="#584CD7")

scrollbar = ctk.CTkScrollbar(chat_frame, command=canvas.yview, bg_color="#584CD7", fg_color="#584CD7")
scrollbar.pack(side=ctk.RIGHT, fill=ctk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

# Frame interno per contenere i messaggi
inner_frame = ctk.CTkFrame(canvas, fg_color="#584CD7", bg_color="#584CD7")
canvas.create_window((0, 0), window=inner_frame, anchor='nw', width=canvas.winfo_width())


def resize_frame(event):
    canvas.itemconfig(inner_frame_id, width=event.width)


canvas.bind('<Configure>', resize_frame)
inner_frame_id = canvas.create_window((0, 0), window=inner_frame, anchor='nw')


def update_scrollregion():
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
    canvas.yview_moveto(1.0)



audio_data = []


def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_data.append(indata.copy())
stream = sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16', callback=callback)


#SPEECH TO TEXT
def record_audio(event):
    global audio_data
    audio_data = []
    stream.start()


def stop_recording(event, frame_param, widget_param):
    stream.stop()
    Thread(target=process_audio, args=(frame_param,widget_param)).start()


def process_audio(frame_param,widget_param):
    msg_frame_bot = ctk.CTkFrame(frame_param, corner_radius=30, fg_color="#7B70EC", border_color="#7B70EC",
                                 border_width=2)
    message_label = ctk.CTkLabel(msg_frame_bot, text="Sto ascoltando...", text_color='white', font=("", 14), anchor='w',
                                 wraplength=300, corner_radius=50)
    message_label.pack(padx=8, pady=8, fill=ctk.BOTH, expand=True)
    msg_frame_bot.pack(padx=18, pady=5, anchor='w')
    audio_data_np = np.concatenate(audio_data, axis=0)
    file_name = "registrazione.wav"
    write(file_name, sample_rate, audio_data_np)
    loader = SpeechToTextLoader(project_id=project_id, file_path=file_name)
    docs = loader.load()
    question = docs[0].page_content
    msg_frame_bot.destroy()
    send(question,frame_param,widget_param)

#TEXT TO SPEECH
def on_frame_click(event,custom_param):
    print("cliccato")
    #play_response(custom_param)
    Thread(target=play_response, args=(custom_param,)).start()

def play_response(response):
    speech_file = tts.run(response)
    pygame.mixer.init()
    pygame.mixer.music.load(speech_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(speech_file)

def send(msg, frame, entry_widget):
    question = msg
    entry_widget.delete(0, ctk.END)
    message_frame_bot = add_message(frame, question, "user", None)
    Thread(target=invoke_model, args=(question, frame, message_frame_bot)).start()

def invoke_model(question, frame, message_frame_bot):
    result = {'result': "Nessuna risposta disponibile"}  # Inizializza con un valore predefinito
    try:
        result = chain.invoke({"query": question})  # question "Suggeriscimi un film romantico"})
        print(f"Final answer: {result['result']}")
    except Exception as e:
        result['result'] = "Scusa penso di non aver capito, puoi ripetere?"
        print(f"Final answer: {result['result']}")
    add_message(frame, result['result'], "bot", message_frame_bot)
    try:
        question = "Domanda: " + question
        chain_user.invoke({"query": question})
    except Exception as e:
        print("")


def add_message(frame, msg, tag, msg_frame_bot=None):
    message_frame_user = ctk.CTkFrame(frame, corner_radius=30, fg_color="white", border_color="white", border_width=2)

    if tag == "user":
        message_frame_bot = ctk.CTkFrame(frame, corner_radius=30, fg_color="#7B70EC", border_color="#7B70EC",
                                         border_width=2)
        message_label = ctk.CTkLabel(message_frame_user, text=msg, text_color='#584CD7', font=("", 14), anchor='e',
                                     wraplength=300)
        message_label.pack(padx=8, pady=8, fill=ctk.BOTH, expand=True)
        message_frame_user.pack(padx=18, pady=5, anchor='e')
        if msg_frame_bot is None:
            message_label = ctk.CTkLabel(message_frame_bot, text="Sto elaborando la risposta...", text_color='white',
                                         font=("", 14), anchor='w',
                                         wraplength=300)
            message_label.pack(padx=8, pady=8, fill=ctk.BOTH, expand=True)
            message_frame_bot.pack(padx=15, pady=5, anchor='w')
            message_frame_bot.update_idletasks()
        message_frame_user.update_idletasks()
        update_scrollregion()
        return message_frame_bot

    elif tag == "bot":
        if msg_frame_bot is not None:
            msg_frame_bot.destroy()
        msg_frame_bot = ctk.CTkFrame(frame, corner_radius=30, fg_color="#7B70EC", border_color="#7B70EC",
                                     border_width=2)
        message_label = ctk.CTkLabel(msg_frame_bot, text=msg, text_color='white', font=("", 14), anchor='w',
                                     wraplength=300, corner_radius=50)
        message_label.pack(padx=8, pady=8, fill=ctk.BOTH, expand=True)
        custom_param = msg
        message_label.bind("<Button-1>", partial(on_frame_click, custom_param=custom_param))
        msg_frame_bot.pack(padx=15, pady=5, anchor='w')

    msg_frame_bot.update_idletasks()
    message_frame_user.update_idletasks()
    update_scrollregion()

    # Scorri verso il basso quando viene aggiunto un nuovo messaggio


# Esempio di aggiunta di messaggi
add_message(inner_frame, "Ciao, come posso aiutarti?", "bot")

# Input frame
input_frame = ctk.CTkFrame(root, bg_color="#584CD7", fg_color="#584CD7")
input_frame.pack(pady=10, fill=ctk.X)

rec_img_path = "./assets/microphone.png"
rec_img_red_path = "./assets/button_2_red.png"
send_img_path = "./assets/send.png"
rec_img = Image.open(rec_img_path)
rec_red_img = Image.open(rec_img_red_path)
send_img = Image.open(send_img_path)
rec_img_tk = ImageTk.PhotoImage(rec_img)
rec_img_red_tk = ImageTk.PhotoImage(rec_red_img)
send_img_tk = ImageTk.PhotoImage(send_img)

rec_button = ctk.CTkButton(input_frame, image=rec_img_tk, bg_color="#584CD7",
                           fg_color="#584CD7", text="", width=30)
rec_button.grid(row=0, column=0, padx=5)


entry_widget = ctk.CTkEntry(input_frame, width=300, height=40, font=("", 14,"bold"), bg_color="#584CD7",
                            fg_color="white", border_color="#584CD7", corner_radius=30,text_color="#584CD7")
entry_widget.grid(row=0, column=1, padx=1, sticky="ew")


rec_button.bind("<ButtonPress-1>", record_audio)
rec_button.bind("<ButtonRelease-1>", partial(stop_recording,frame_param=inner_frame,widget_param=entry_widget))

send_button = ctk.CTkButton(input_frame, image=send_img_tk,
                            command=lambda: send(entry_widget.get().strip(), inner_frame, entry_widget),
                            bg_color="#584CD7",
                            fg_color="#584CD7", text="", width=30)
send_button.grid(row=0, column=2, padx=2)

input_frame.grid_columnconfigure(1, weight=1)

root.mainloop()
