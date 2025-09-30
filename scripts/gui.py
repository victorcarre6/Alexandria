import threading
import platform
import tkinter as tk
import time
import json
import os
from pathlib import Path
from tkinter import ttk, scrolledtext, messagebox
import networkx as nx
from neo4j import GraphDatabase
import networkx as nx
import sys
import webbrowser

from metadatas.fetch_ss import get_last_papers_from_query, get_unpaywall_pdf, insert_articles_into_sqlite
#from metadatas.fetch_subject import insert_articles_into_sqlite
from metadatas import downloader
from metadatas import docling
from connections.sim_graph import generate_graph

# -----------------------
#     Initialisation
# -----------------------

# === Connexions ===

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

keys_path = ROOT_DIR / "keys.json"
with open(keys_path) as f:
    keys = json.load(f)

NEO4J_URI = keys["NEO4J_URI"]
NEO4J_USERNAME = keys["NEO4J_USERNAME"]
NEO4J_PASSWORD = keys["NEO4J_PASSWORD"]
email = keys["EMAIL"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
G: nx.Graph = nx.Graph()

DB_PATH = "datas/bibliography.db"

# === Paramètres ===

PATIENT_ZERO = ""

# --- Fetch ---

max_results = 20
year_from = 2015
year_to = 2026

email = "victorcarre@icloud.com"
divided_max_results = round(max_results/3)


# --- Graph ---

DEPTH = 4
FETCH_MAX_DOIS = 20
GRAPH_MAX_DOIS = 500


PAUSE = 0.5
BATCH_SIZE = 100
MAX_WORKER = 10

EDGE_MODE = "SIMILAR"   # "CITES", "SIMILAR"
WEIGHT_THRESHOLD = 10
TOP_N = 1
TOP_N_CITES = 30
TOP_N_SCORE = 10
TOP_N_COCIT = 10
TOP_N_COUPLING = 10

root = tk.Tk()
total_analyzed = 0

# -----------------------
#        Fonctions
# -----------------------

# === Principales ===

def on_sim_graph():

    patient_zero =  input.get("1.0", tk.END).strip()

    generate_graph(driver, DB_PATH, patient_zero, DEPTH, FETCH_MAX_DOIS, GRAPH_MAX_DOIS, 
                    PAUSE, BATCH_SIZE, MAX_WORKER, 
                    EDGE_MODE, WEIGHT_THRESHOLD, 
                    TOP_N, TOP_N_CITES, TOP_N_SCORE, TOP_N_COCIT, TOP_N_COUPLING)

def fetcher_ss(limit=FETCH_MAX_DOIS, db_path=DB_PATH):
    patient_zero = input.get("1.0", tk.END).strip()
    print(f"[fetcher_ss] Starting fetch for: '{patient_zero}' with limit={limit}")

    print("[fetcher_ss] Calling get_last_papers_from_query...")
    raw_results = get_last_papers_from_query(patient_zero, limit)  # dict {doi: {...}}
    print(f"[fetcher_ss] get_last_papers_from_query returned {len(raw_results)} results.")

    articles = []
    for doi, data in raw_results.items():
        # Combine patient_zero with field from API
        field_list = []
        # In fetch_ss.py, 'field' is a string (comma-joined fields)
        if data.get("field"):
            # Split the string into a list, stripping whitespace
            field_list.extend([f.strip() for f in data["field"].split(",") if f.strip()])
        if patient_zero:
            field_list.append(patient_zero)
        print(f"[fetcher_ss] Built fields for DOI {doi}: {field_list}")
        articles.append({
            "field": "; ".join(field_list) if field_list else patient_zero,
            "paper_type": data.get("publicationTypes", ""),
            "title": data.get("title", ""),
            "authors": str(data.get("authors", "")),
            "doi": data.get("doi", doi),
            "journal": data.get("journal", ""),
            "year": data.get("year", ""),
            "volume": data.get("volume", ""),
            "issue": data.get("issue", ""),  # Not provided by SS, fallback to ""
            "pages": data.get("pages", ""),
            "source": "SemanticScholar",
            "pdf_link": data.get("pdf_link", ""),
            "abstract": data.get("abstract", "")
        })

    print(f"[fetcher_ss] Prepared {len(articles)} articles for insertion.")
    print("[fetcher_ss] Calling insert_articles_into_sqlite...")
    insert_articles_into_sqlite(articles, email=email, db_path=db_path)
    print("[fetcher_ss] insert_articles_into_sqlite finished.")
    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, f"[SemanticScholar] {len(articles)} articles insérés.\n\n")

    for i, article in enumerate(articles, start=1):
        text_output.insert(tk.END, f"{i}. {article['title']}\n")
        text_output.insert(tk.END, f"   {article['authors']}\n")
        journal_line = f"   {article['journal']} ({article['year']})"
        if article['volume']:
            journal_line += f", {article['volume']}"
        if article['issue']:
            journal_line += f", {article['issue']}"
        if article['pages']:
            journal_line += f", {article['pages']}."
        text_output.insert(tk.END, journal_line + "\n")
        text_output.insert(tk.END, f"   doi:{article['doi']}\n")
        if article['pdf_link']:
            text_output.insert(tk.END, f"   Open Access available at {article['pdf_link']}\n\n")
        if article['abstract']:
            text_output.insert(tk.END, f"{article['abstract']}\n\n")

def parser():
    pass  # TODO: implement context parsing
    
# === Affichage ===

def bring_to_front():
    root.update()
    root.deiconify()            
    root.lift()               
    root.attributes('-topmost', True)
    root.after(200, lambda: root.attributes('-topmost', False)) 

def show_help():
    help_window = tk.Toplevel(root)
    help_window.title("Help")
    help_window.geometry("500x500")
    help_window.configure(bg="#323232")
    help_window.resizable(False, False)

    frame = tk.Frame(help_window, bg="#323232")
    frame.place(x=0, y=0, width=960, height=450)

    title_label = tk.Label(
        frame,
        text="Alexandria — Don't panic !",
        font=("Segoe UI", 12, "bold"),
        bg="#323232",
        fg="white",
        justify=tk.CENTER
    )
    title_label.pack(fill=tk.X, pady=(0, 2))

    help_text = (
            "• Generate (▲): Ask a question and get an answer using the memory system.\n\n"
            "• \n"
            "github.com/victorcarre6/llm-memorization."
        )
    
    label = tk.Label(
        frame,
        text=help_text,
        font=("Segoe UI", 12),
        bg="#323232",
        fg="white",
        justify=tk.LEFT
    )
    label.pack(fill=tk.BOTH, expand=True)    
    
def update_status(message, error=False, success=False):
    label_status.config(text=message)
    if error:
        label_status.config(foreground='#ff6b6b')
    elif success:
        label_status.config(foreground='#599258')
    else:
        label_status.config(foreground='white')

def open_github(event):
    webbrowser.open_new("https://github.com/victorcarre6")
    

# -----------------------
#       Interface
# -----------------------


root.title("Alexandria")
root.geometry("900x750")
root.configure(bg="#323232")

style = ttk.Style(root)
style.theme_use('clam')

style_config = {
    'Green.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 12),
        'padding': 2
    },
    'Bottom.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 12),
        'padding': 2
    },
    'Blue.TLabel': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 11, 'italic underline'),
        'padding': 0
    },
    'TLabel': {
        'background': '#323232',
        'foreground': 'white',
        'font': ('Segoe UI', 13)
    },
    'TEntry': {
        'fieldbackground': '#FDF6EE',
        'foreground': 'black',
        'font': ('Segoe UI', 13)
    },
    'TFrame': {
        'background': '#323232'
    },
    'Status.TLabel': {
        'background': '#323232',
        'font': ('Segoe UI', 13)
    },
    'TNotebook': {
        'background': '#323232',
        'borderwidth': 0
    },
    'TNotebook.Tab': {
        'background': '#2a2a2a',
        'foreground': 'white',
        'padding': (10, 4)
    },
    'Custom.Treeview': {
        'background': '#2a2a2a',
        'foreground': 'white',
        'fieldbackground': '#2a2a2a',
        'font': ('Segoe UI', 12),
        'bordercolor': '#323232',
        'borderwidth': 0,
    },
    'Custom.Treeview.Heading': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 13, 'bold'),
        'relief': 'flat'
    }
}

for style_name, app_config in style_config.items():
    style.configure(style_name, **app_config)

style.map('Green.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map("TNotebook.Tab",
          background=[("selected", "#323232"), ("active", "#2a2a2a")],
          foreground=[("selected", "white"), ("active", "white")])

style.map('Bottom.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map('TCheckbutton',
          background=[('active', '#323232'), ('pressed', '#323232')],
          foreground=[('active', 'white'), ('pressed', 'white')])

# === WIDGETS PRINCIPAUX ===

main_frame = ttk.Frame(root, padding=10, style='TFrame')
main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Input frame
input_frame = tk.Frame(main_frame, bg="#323232")
input_frame.pack(fill=tk.X, expand=False)

input = tk.Text(input_frame, height=1.5, width=10, wrap="word", font=('Segoe UI', 13))
input.pack(side="left", fill="both", expand=True)

style.configure("Vertical.TScrollbar",
                troughcolor='#FDF6EE',
                background='#C0C0C0',
                darkcolor='#C0C0C0',
                lightcolor='#C0C0C0',
                bordercolor='#FDF6EE',
                arrowcolor='black',
                relief='flat')

input.bind("<Return>", lambda event: (fetcher_ss(), "break"))

input_button_frame = tk.Frame(input_frame, bg="#323232")
input_button_frame.pack(side="right", fill=tk.Y)

input_button_bottom_frame = tk.Frame(input_frame, bg="#323232")
input_button_bottom_frame.pack(side="right", fill=tk.Y)

btn_list = ttk.Button(
    input_button_frame,
    text="Get last publications",
    command=fetcher_ss,
    style='Green.TButton'
)

btn_graph = ttk.Button(
    input_button_frame,
    text="Generate a semantic graph",
    command=on_sim_graph,
    style='Green.TButton'
)

btn_db = ttk.Button(
    input_button_bottom_frame,
    text="Get publications from the database",
    command=fetcher_ss,
    style='Green.TButton'
)

btn_graph.pack(side="left", padx=(0, 5), pady=(0, 0), fill=tk.X, expand=True)
btn_list.pack(side="left", padx=(0, 5), pady=(0, 0), fill=tk.X, expand=True)
btn_db.pack(side="left", padx=(0, 5), pady=(0, 0), fill=tk.X, expand=True)

# --- Placeholder / texte par défaut ---
PLACEHOLDER_TEXT = "Insert a DOI (10.{...}), keywords, field ..."

def add_placeholder():
    input.insert("1.0", PLACEHOLDER_TEXT)
    input.tag_add("placeholder", "1.0", "end")
    input.tag_config("placeholder", foreground="gray")

def remove_placeholder(event):
    current_text = input.get("1.0", "end-1c")
    if current_text == PLACEHOLDER_TEXT:
        input.delete("1.0", "end")
        input.tag_remove("placeholder", "1.0", "end")

def restore_placeholder(event):
    current_text = input.get("1.0", "end-1c")
    if current_text.strip() == "":
        add_placeholder()

# Initialiser le placeholder
add_placeholder()

# Bind events pour gérer le focus
input.bind("<FocusIn>", remove_placeholder)
input.bind("<FocusOut>", restore_placeholder)

# === ZONE DE SORTIE ÉTENDABLE ===
output_expanded = tk.BooleanVar(value=False)

output_frame = ttk.Frame(main_frame, style='TFrame')
output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

text_output = scrolledtext.ScrolledText(
    output_frame,
    width=100,
    height=20,
    font=('Segoe UI', 13),
    wrap=tk.WORD,
    insertbackground="black"
)

text_output.pack(fill=tk.BOTH, expand=True)

# Context menu (right click)
if platform.system() == "Darwin":
    right_click_event = "<Button-2>"
else:
    right_click_event = "<Button-3>"


# Menu contextuels pour input/output (zone de question)
question_context_menu = tk.Menu(input, tearoff=0)
question_context_menu.add_command(label="Copy", command=lambda: input.event_generate("<<Copy>>"))
question_context_menu.add_command(label="Paste", command=lambda: input.event_generate("<<Paste>>"))
question_context_menu.add_command(label="Select all", command=lambda: input.tag_add("sel", "1.0", "end"))

output_context_menu = tk.Menu(text_output, tearoff=0)
output_context_menu.add_command(label="Copier", command=lambda: text_output.event_generate("<<Copy>>"))
output_context_menu.add_command(label="Coller", command=lambda: text_output.event_generate("<<Paste>>"))
output_context_menu.add_command(label="Tout sélectionner", command=lambda: text_output.tag_add("sel", "1.0", "end"))

def show_question_context_menu(event):
    try:
        question_context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        question_context_menu.grab_release()

input.bind(right_click_event, show_question_context_menu)

def show_output_context_menu(event):
    try:
        output_context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        output_context_menu.grab_release()

text_output.bind(right_click_event, show_output_context_menu)


input_frame.bind(right_click_event, show_question_context_menu)
output_frame.bind(right_click_event, show_output_context_menu)

# === BARRE DE STATUT ET BOUTONS ===

status_buttons_frame = ttk.Frame(main_frame, style='TFrame')
status_buttons_frame.pack(fill=tk.X, pady=(5, 2))

# Buttons frame below output, horizontal, fill width, fixed height
buttons_frame = ttk.Frame(main_frame, style='TFrame')
buttons_frame.pack(fill=tk.X, pady=(0, 10))

btn_select_fetch = ttk.Button(buttons_frame, text="Use as reference", command=fetcher_ss, style='Bottom.TButton')
btn_select_fetch.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

btn_Select_graph = ttk.Button(buttons_frame, text="Generate a graph", command=on_sim_graph, style='Bottom.TButton')
btn_Select_graph.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

btn_select_rag = ttk.Button(buttons_frame, text="Use as context", command=parser, style='Bottom.TButton')
btn_select_rag.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

# New frame for status label and Help button on same horizontal line
status_help_frame = ttk.Frame(main_frame, style='TFrame')
status_help_frame.pack(fill=tk.X, pady=(5, 2))

label_status = ttk.Label(
    status_help_frame,
    text="Ready",
    style='Status.TLabel',
    foreground='white',
    anchor='w'
)
label_status.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor='w')

btn_help = ttk.Button(status_help_frame, text="Help", style='Bottom.TButton', command=show_help, width=8)
btn_help.pack(side=tk.RIGHT, padx=(0, 0))

# === FOOTER ===
footer_bottom_frame = ttk.Frame(root, style='TFrame')
footer_bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

dev_label = ttk.Label(footer_bottom_frame, text="Developped by Victor Carré —", style='TLabel', font=('Segoe UI', 10))
dev_label.pack(side=tk.LEFT)

github_link = ttk.Label(footer_bottom_frame, text="GitHub", style='Blue.TLabel', cursor="hand2")
github_link.pack(side=tk.LEFT)
github_link.bind("<Button-1>", open_github)

bring_to_front()

root.mainloop()