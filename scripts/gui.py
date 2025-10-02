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
import sqlite3

from metadatas.fetch_ss import get_last_papers_from_query, get_unpaywall_pdf, insert_articles_into_sqlite
#from metadatas.fetch_subject import insert_articles_into_sqlite
from metadatas import downloader
from metadatas import docling
from connections.sim_graph import generate_graph, generate_graph_from_closed_pool

# -----------------------
#     Initialisation
# -----------------------

# === Connexions ===

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

keys_path = ROOT_DIR / "resources" / "keys.json"
with open(keys_path) as f:
    keys = json.load(f)

NEO4J_URI = keys["NEO4J_URI"]
NEO4J_USERNAME = keys["NEO4J_USERNAME"]
NEO4J_PASSWORD = keys["NEO4J_PASSWORD"]
EMAIL = keys["EMAIL"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
G: nx.Graph = nx.Graph()

def load_config(config_path):
    config_path = expand_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    config = raw_config
    return config

def expand_path(value):
    if isinstance(value, Path):
        return str(value.resolve())
    expanded = Path(os.path.expanduser(value))
    if not expanded.is_absolute():
        expanded = ROOT_DIR / expanded
    return str(expanded.resolve())

CONFIG_PATH = ROOT_DIR / "resources" / "config.json"
config = load_config(CONFIG_PATH)
DB_PATH = ROOT_DIR / "datas" / "bibliography.db"

# === Paramètres ===

# --- Graph ---
root = tk.Tk()

total_analyzed = 0
displayed_dois = []
initial_articles = []  # liste complète chargée depuis la DB ou fetch
displayed_articles = []  # liste affichée après filtre/tri
selected_doi = None

PATIENT_ZERO = ""

fetch_conf = config.get("fetch", {})
graph_conf = config.get("graph", {})
database_conf = config.get("database", {})
DEPTH = tk.IntVar(value=fetch_conf.get("DEPTH", 3))
FETCH_MAX_DOIS = tk.IntVar(value=fetch_conf.get("FETCH_MAX_DOIS", 50))
GRAPH_MAX_DOIS = tk.IntVar(value=fetch_conf.get("GRAPH_MAX_DOIS", 500))
DATES = tk.StringVar(value=fetch_conf.get("DATES", "2010-"))
SORT_BY = tk.StringVar(value=fetch_conf.get("SORT_BY", "relevance"))
PAUSE = tk.IntVar(value=fetch_conf.get("PAUSE", 0.5))
BATCH_SIZE = tk.IntVar(value=fetch_conf.get("BATCH_SIZE", 100))
MAX_WORKER = tk.IntVar(value=fetch_conf.get("MAX_WORKER", 10))
EDGE_MODE = tk.StringVar(value=graph_conf.get("EDGE_MODE", "SIMILAR"))
WEIGHT_THRESHOLD = tk.IntVar(value=graph_conf.get("WEIGHT_THRESHOLD", 10))
TOP_N = tk.IntVar(value=graph_conf.get("TOP_N", 1))
TOP_N_CITES = tk.IntVar(value=graph_conf.get("TOP_N_CITES", 30))
TOP_N_SCORE = tk.IntVar(value=graph_conf.get("TOP_N_SCORE", 10))
TOP_N_COCIT = tk.IntVar(value=graph_conf.get("TOP_N_COCIT", 10))
TOP_N_COUPLING = tk.IntVar(value=graph_conf.get("TOP_N_COUPLING", 10))
EPHEMERAL_MODE = tk.BooleanVar(value=database_conf.get("EPHEMERAL", False))

def reset_to_defaults(settings_window=None):
    # Valeurs par défaut des variables
    DEFAULTS = {
        "DEPTH": 3,
        "FETCH_MAX_DOIS": 50,
        "GRAPH_MAX_DOIS": 500,
        "PAUSE": 0.5,
        "BATCH_SIZE": 100,
        "MAX_WORKER": 10,
        "EDGE_MODE": "SIMILAR", # "CITES", "SIMILAR"
        "WEIGHT_THRESHOLD": 10,
        "TOP_N": 1,
        "TOP_N_CITES": 30,
        "TOP_N_SCORE": 10,
        "TOP_N_COCIT": 10,
        "TOP_N_COUPLING": 10,
        "EPHEMERAL": False
    }

    # Appliquer les valeurs par défaut aux variables dans le config.json
    DEPTH.set(DEFAULTS["DEPTH"])
    GRAPH_MAX_DOIS.set(DEFAULTS["GRAPH_MAX_DOIS"])
    EDGE_MODE.set(DEFAULTS["EDGE_MODE"])
    TOP_N.set(DEFAULTS["TOP_N"])
    DATES.set(DEFAULTS["DATES"])
    SORT_BY.set(DEFAULTS["SORT_BY"])
    EPHEMERAL_MODE.set(DEFAULTS["EPHEMERAL"])
    
    update_status("Settings reset to defaults.", success=True)
    root.update()

    if settings_window is not None:
        settings_window.destroy()

def save_gui_config(*args):
    config["fetch"]["DEPTH"] = DEPTH.get()
    config["fetch"]["GRAPH_MAX_DOIS"] = GRAPH_MAX_DOIS.get()
    config["fetch"]["DATES"] = DATES.get()
    config["fetch"]["SORT_BY"] = SORT_BY.get()
    config["graph"]["EDGE_MODE"] = EDGE_MODE.get()
    config["graph"]["TOP_N"] = TOP_N.get()
    config["database"]["EPHEMERAL"] = EPHEMERAL_MODE.get()
    with open(expand_path(CONFIG_PATH), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# Sauvegarde automatique des paramètres
DEPTH.trace_add("write", save_gui_config)
GRAPH_MAX_DOIS.trace_add("write", save_gui_config)
EDGE_MODE.trace_add("write", save_gui_config)
TOP_N.trace_add("write", save_gui_config)
DATES.trace_add("write", save_gui_config)
SORT_BY.trace_add("write", save_gui_config)
EPHEMERAL_MODE.trace_add("write", save_gui_config)

# -----------------------
#        Fonctions
# -----------------------

def generate_graph_with_selection():
    global selected_doi
    if selected_doi:
        generate_graph_selected()
    else:
        on_sim_graph_pool()
        
def select_doi(doi):
    global selected_doi
    selected_doi = doi
    update_status(f"Selected DOI: {doi}")  # Affiche le DOI sélectionné dans la barre de statut

def delete_selected_publication():
    global selected_doi, displayed_dois
    if not selected_doi:
        update_status("No publication selected for deletion.", error=True)
        return

    confirm = messagebox.askyesno(
        "Delete Publication",
        f"Are you sure you want to delete DOI {selected_doi} from the database?"
    )
    if not confirm:
        return

    try:
        # Suppression depuis SQLite
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM paper_data WHERE doi=?", (selected_doi,))
        c.execute("DELETE FROM metadatas WHERE doi=?", (selected_doi,))
        conn.commit()
        conn.close()

        update_status(f"Publication {selected_doi} deleted from database.")

        # Réinitialiser la sélection
        selected_doi = None

    except Exception as e:
        update_status(f"Error deleting publication: {e}", error=True)

def generate_graph_selected():
    global selected_doi
    if selected_doi:
        # Appeler la fonction classique on_sim_graph avec le DOI sélectionné
        input.delete("1.0", tk.END)
        input.insert(tk.END, selected_doi)
        on_sim_graph()
    else:
        update_status("No DOI selected", error=True)
          
def fetcher_selected():
    global selected_doi
    if selected_doi:
        # Appeler la fonction classique on_sim_graph avec le DOI sélectionné
        input.delete("1.0", tk.END)
        input.insert(tk.END, selected_doi)
        
        fetcher_ss()
    else:
        update_status("No DOI selected", error=True)

def update_displayed_articles(articles_list):
    """
    Remplit text_output à partir d'une liste d'articles.
    """
    text_output.delete("1.0", tk.END)
    displayed_dois.clear()
    
    for i, article in enumerate(articles_list, start=1):
        doi = article["doi"]
        displayed_dois.append(doi)
        start_index = text_output.index(tk.END)
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
        text_output.insert(tk.END, f"   doi:{doi}\n")
        if article['pdf_link']:
            text_output.insert(tk.END, f"   Open Access available at {article['pdf_link']}\n\n")
        if article['abstract']:
            text_output.insert(tk.END, f"{article['abstract']}\n\n")
        # Tag sur le DOI
        tag_name = f"doi_{i}"
        end_index = text_output.index(tk.END)
        text_output.tag_add(tag_name, start_index, end_index)
        text_output.tag_bind(tag_name, "<Button-1>", lambda e, d=doi: select_doi(d))
        text_output.tag_bind(tag_name, right_click_event, lambda e, d=doi: select_doi(d))

# === Principales ===

def on_sim_graph():
    patient_zero =  input.get("1.0", tk.END).strip()
    _DEPTH = DEPTH.get()
    _GRAPH_MAX_DOIS = GRAPH_MAX_DOIS.get()
    _EDGE_MODE = EDGE_MODE.get()
    _TOP_N = TOP_N.get()

    _DATES = DATES.get()
    _SORT_BY = SORT_BY.get()
    _EPHEMERAL_MODE = EPHEMERAL_MODE.get()
    update_status("Generating semantic graph ...")
    root.update()
    results = generate_graph(driver, DB_PATH, patient_zero, _DEPTH, FETCH_MAX_DOIS.get(), _GRAPH_MAX_DOIS, 
                   PAUSE.get(), BATCH_SIZE.get(), MAX_WORKER.get(), 
                   _EDGE_MODE, WEIGHT_THRESHOLD.get(), 
                   _TOP_N, TOP_N_CITES.get(), TOP_N_SCORE.get(), TOP_N_COCIT.get(), TOP_N_COUPLING.get())
    articles = []
    for doi, data in results.items():
        field_list = []
        field_val = data.get("field")
        if isinstance(field_val, str) and field_val:
            field_list.extend([f.strip() for f in field_val.split(",") if f.strip()])
        elif isinstance(field_val, list):
            field_list.extend([str(f).strip() for f in field_val if str(f).strip()])
        if patient_zero:
            field_list.append(patient_zero)

        articles.append({
            "field": "; ".join(field_list) if field_list else patient_zero,
            "paper_type": data.get("paper_type", data.get("publicationTypes", "")),
            "title": data.get("title", ""),
            "authors": str(data.get("authors", "")),
            "doi": data.get("doi", doi),
            "journal": data.get("journal", ""),
            "year": data.get("year", ""),
            "volume": data.get("volume", ""),
            "issue": data.get("issue", ""),
            "pages": data.get("pages", ""),
            "source": "SemanticScholar",
            "pdf_link": data.get("pdf_link", ""),
            "abstract": data.get("abstract", "")
        })

    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, f"[Graph] {len(articles)} articles affichés.\n\n")

    for i, article in enumerate(articles, start=1):
        doi = article["doi"]
        displayed_dois.append(doi)
        start_index = text_output.index(tk.END)
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

        # Tag sur le DOI
        tag_name = f"doi_{i}"
        end_index = text_output.index(tk.END)
        text_output.tag_add(tag_name, start_index, end_index)
        text_output.tag_bind(tag_name, "<Button-1>", lambda e, d=doi: select_doi(d))
        text_output.tag_bind(tag_name, right_click_event, lambda e, d=doi: select_doi(d))
    update_status("Graph generation completed")
    root.update()


def on_sim_graph_pool():
    global displayed_dois
    print("DEBUG displayed_dois avant appel:", displayed_dois)
    batch_size = 100
    _EDGE_MODE = EDGE_MODE.get()
    _TOP_N = TOP_N.get()

    _DATES = DATES.get()
    _SORT_BY = SORT_BY.get()
    _EPHEMERAL_MODE = EPHEMERAL_MODE.get()
    
    update_status("Generating semantic graph from selected publications ...")
    root.update()
    generate_graph_from_closed_pool(driver, DB_PATH, displayed_dois, PAUSE.get(), batch_size, 
                                    _EDGE_MODE, WEIGHT_THRESHOLD.get(), 
                                    _TOP_N, TOP_N_CITES.get(), TOP_N_SCORE.get(), 
                                    TOP_N_COCIT.get(), TOP_N_COUPLING.get(), 
                                    graph_max_dois=500, depth=2)
    update_status("Graph generation completed")
    root.update()

def fetcher_ss(limit=None, email=None, db_path=str(DB_PATH)):
    global initial_articles, displayed_articles
    patient_zero = input.get("1.0", tk.END).strip()
    if limit is None:
        limit = FETCH_MAX_DOIS.get()
    if email is None:
        email = EMAIL


    update_status("Fetching publications from Semantic Scholar ...")
    root.update()
    print(f"[fetcher_ss] Starting fetch for: '{patient_zero}' with limit={limit}")

    raw_results = get_last_papers_from_query(patient_zero, limit)
    update_status(f"{len(raw_results)} publications retrieved from Semantic Scholar")
    root.update()

    articles = []
    for doi, data in raw_results.items():
        # Combine patient_zero with field from API
        field_list = []
        field_val = data.get("field")
        if isinstance(field_val, str) and field_val:
            # Split the string into a list, stripping whitespace
            field_list.extend([f.strip() for f in field_val.split(",") if f.strip()])
        elif isinstance(field_val, list):
            # Already a list, just strip each element
            field_list.extend([str(f).strip() for f in field_val if str(f).strip()])
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

    update_status("Inserting publications into local database ...")
    root.update()
    insert_articles_into_sqlite(articles, email, db_path)
    update_status(f"{len(articles)} publications fetched and added to the database")
    root.update()
    for idx, art in enumerate(articles):
                art["_initial_pos"] = idx  # position relative dans la liste initiale
    initial_articles = articles.copy()  # sauvegarde pour refinement
    displayed_articles = articles.copy() 
    update_displayed_articles(displayed_articles)
    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, f"[SemanticScholar] {len(articles)} articles insérés.\n\n")

    for i, article in enumerate(articles, start=1):
        doi = article["doi"]
        displayed_dois.append(article["doi"])
        start_index = text_output.index(tk.END)
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
            text_output.insert(tk.END, f"{article['abstract']}")
        text_output.insert(tk.END, f"\n\n")
        # Tag sur le DOI
        tag_name = f"doi_{i}"
        end_index = text_output.index(tk.END)
        text_output.tag_add(tag_name, start_index, end_index)
        text_output.tag_bind(tag_name, "<Button-1>", lambda e, d=doi: select_doi(d))
        text_output.tag_bind(tag_name, right_click_event, lambda e, d=doi: select_doi(d))

def get_publication_by_doi(doi: str, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("""
        SELECT m.doi, m.field, m.title, m.authors, m.year, m.journal, 
               m.paper_type, m.volume, m.issue, m.pages, p.abstract, p.pdf_link
        FROM metadatas m
        LEFT JOIN paper_data p ON m.doi = p.doi
        WHERE m.doi = ?
    """, (doi,))
    
    row = c.fetchone()
    conn.close()
    
    if row:
        keys = ["doi", "field", "title", "authors", "year", "journal", 
                "paper_type", "volume", "issue", "pages", "abstract", "pdf_link"]
        return dict(zip(keys, row))
    else:
        return None

def get_publication_by_keywords(keywords, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    query = """
        SELECT m.doi, m.field, m.title, m.authors, m.year, m.journal, 
               m.paper_type, m.volume, m.issue, m.pages, p.abstract, p.pdf_link
        FROM metadatas m
        LEFT JOIN paper_data p ON m.doi = p.doi
        WHERE m.title LIKE ? OR m.authors LIKE ? OR m.field LIKE ?
    """
    
    pattern = f"%{keywords}%"
    c.execute(query, (pattern, pattern, pattern))
    rows = c.fetchall()
    conn.close()
    
    keys = ["doi", "field", "title", "authors", "year", "journal", 
            "paper_type", "volume", "issue", "pages", "abstract", "pdf_link"]
    return [dict(zip(keys, row)) for row in rows]

def get_from_base(db_path=DB_PATH):
    global displayed_articles, initial_articles
    
    query = input.get("1.0", tk.END).strip()
    text_output.delete("1.0", tk.END)
    

    if query.startswith("10."):  # cas DOI
        update_status(f"Retrieving publication informations from database with DOI {query}")
        root.update()
        result = get_publication_by_doi(query, db_path)
        update_status(f"Publications DOI:{query} retrieved from the database")
        root.update()
        if result:
            results = [result]
            for idx, art in enumerate(results):
                art["_initial_pos"] = idx  # position relative dans la liste initiale
            initial_articles = results.copy()
            displayed_articles = results.copy() 
        else:
            results = []
    else:  # cas mots-clés
        update_status(f'Retrieving publications informations from database with keyword "{query}"')
        root.update()
        results = get_publication_by_keywords(query, db_path)
        for idx, art in enumerate(results):
                art["_initial_pos"] = idx  # position relative dans la liste initiale
        initial_articles = results.copy()  # sauvegarde pour refinement
        displayed_articles = results.copy() 
        update_status(f"{len(results)} publications retrieved from the database")
        root.update()

    update_displayed_articles(displayed_articles)

    for i, article in enumerate(results, start=1):
        doi = article["doi"]
        displayed_dois.append(article["doi"])
        start_index = text_output.index(tk.END)
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
            text_output.insert(tk.END, f"{article['abstract']}")
        text_output.insert(tk.END, f"\n\n")
        # Tag sur le DOI
        tag_name = f"doi_{i}"
        end_index = text_output.index(tk.END)
        text_output.tag_add(tag_name, start_index, end_index)
        text_output.tag_bind(tag_name, "<Button-1>", lambda e, d=doi: select_doi(d))
        text_output.tag_bind(tag_name, right_click_event, lambda e, d=doi: select_doi(d))
        
    
def open_in_browser():
    global selected_doi
    if not selected_doi:
        update_status("No publication selected for opening in browser.", error=True)
        return

    record = get_publication_by_doi(selected_doi, DB_PATH)
    if record and record.get("doi"):
        doi_url = f"https://doi.org/{record['doi']}"
        try:
            webbrowser.open_new_tab(doi_url)
            update_status(f"{doi_url} opened in web browser.", success=True)
        except Exception as e:
            update_status(f"Error opening browser: {e}", error=True)
    else:
        update_status("DOI not found in the database.", error=True)
    

def download_selected_pdf():
    global selected_doi
    if not selected_doi:
        update_status("No publication selected for download.", error=True)
        return
    
    record = get_publication_by_doi(selected_doi, DB_PATH)
    if record and record.get("pdf_link"):
        result = downloader.download(record)
        local_path = result.get("local_path")
        if local_path:
            update_status(f"PDF downloaded to: {local_path}", success=True)
        else:
            update_status("PDF could not be downloaded.", error=True)
    else:
        update_status("No Open Access PDF available for the selected publication.", error=True)   

def parser():
    pass  # TODO: implement context parsing
    
# === Affichage ===

def open_quickstart():
    sp_window = tk.Toplevel(root)
    sp_window.title("System Prompt")
    sp_window.geometry("450x325")
    sp_window.configure(bg="#323232")
    sp_window.resizable(False, False)

    sp_frame = ttk.Frame(sp_window, padding=10, style="TFrame")
    sp_frame.pack(fill=tk.BOTH, expand=False)
    
def open_routines():
    sp_window = tk.Toplevel(root)
    sp_window.title("System Prompt")
    sp_window.geometry("450x325")
    sp_window.configure(bg="#323232")
    sp_window.resizable(False, False)

    sp_frame = ttk.Frame(sp_window, padding=10, style="TFrame")
    sp_frame.pack(fill=tk.BOTH, expand=False)


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
    
def open_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")
    settings_window.geometry("250x350")
    settings_window.configure(bg="#323232")
    settings_window.resizable(False, False)

    settings_frame = ttk.Frame(settings_window, padding=10, style='TFrame')
    settings_frame.pack(fill=tk.BOTH, expand=True)

    label_depth_count = ttk.Label(settings_frame, text=f"Depth: {DEPTH.get()}", style='TLabel')
    label_depth_count.pack(anchor='center')

    slider_depth = ttk.Scale(
        settings_frame,
        from_=1,
        to=8,
        orient="horizontal",
        variable=DEPTH,
        length=150,
        command=lambda val: label_depth_count.config(text=f"Depth: {int(float(val))}")
    )
    slider_depth.pack(anchor='center', pady=(0,5))


    label_max_dois_count = ttk.Label(settings_frame, text=f"Max pre-filtered results: {GRAPH_MAX_DOIS.get()}", style='TLabel')
    label_max_dois_count.pack(anchor='center')

    slider_max_dois = ttk.Scale(
        settings_frame,
        from_=100,
        to=5000,
        orient=tk.HORIZONTAL,
        variable=GRAPH_MAX_DOIS,
        length=150,
        command=lambda val: label_max_dois_count.config(text=f"Max pre-filtered results: {int(float(val))}")
    )
    slider_max_dois.pack(anchor='center', pady=(0,10))

    label_top_n_count = ttk.Label(settings_frame, text=f"Retrieved publications: {TOP_N.get():1}", style='TLabel')
    label_top_n_count.pack(anchor='center')

    slider_top_n = ttk.Scale(
        settings_frame,
        from_=1,
        to=100,
        orient=tk.HORIZONTAL,
        variable=TOP_N,
        length=150,
        command=lambda val: label_top_n_count.config(text=f"Retrieved publications: {int(float(val))}")
    )
    slider_top_n.pack(anchor='center', pady=(0,5))

    """
    Ici menu déroulant vers le bas, qui permet de choisir deux fois soit un tiret soit un entier (1980 à 2026).
    Le but est que ce soit une phrase qui soit affichée, telle que "Get publications from {premier menu} to {second menu}.
    La variable est pour l'instant appelée dates_count, voir pour l'utiliser tel quel en adaptant les deux dates ou alors en faisant deux variables distinctes.
    Attention, il faut penser au cas où l'utilisateur choisi une date de la borne inférieure correspondant à un date ultérieure à la borne supérieure. 
    Dans ce cas, prévoir un message d'erreur type "The dates that you selected aren't possible", ou alors tout simplement retourner les deux bornes.
    """

    """
    Ici menu déroulant vers le bas, qui permet de choisir le type de tri effectué sur les publications récupérées et affichées dans l'output.
    Le but est que ce soit une phrase qui soit affichée, telle que "Sort publications by {variable}.
    La variable est pour l'instant appelée sort_by, et est passée dans l'API semantic scholar, au sein de "query". Elle peut valoir "relevance" ou "date".
    """

    chk_edge_mode = ttk.Checkbutton(
        settings_frame,
        text="Citing-only graph",
        variable=EDGE_MODE,
        style='Custom.TCheckbutton'
    )
    chk_edge_mode.pack(anchor='center', pady=2)

    chk_ephemeral_mode = ttk.Checkbutton(
        settings_frame,
        text="Don't save new publications",
        variable=EPHEMERAL_MODE,
        style='Custom.TCheckbutton'
    )
    chk_ephemeral_mode.pack(anchor='center', pady=(5,5))

    spacer = ttk.Label(settings_frame, text="")
    spacer.pack(pady=(5,0))

    buttons_frame = ttk.Frame(settings_frame)
    buttons_frame.pack(anchor='center', pady=(0,10))

    quickstart_btn = ttk.Button(
        buttons_frame,
        text="Quickstart",
        command=open_quickstart,
        style="Settings.TButton",
        cursor="hand2"
    )
    quickstart_btn.pack(side="left", padx=5)

    routines_btn = ttk.Button(
        buttons_frame,
        text="Routines",
        command=open_routines,
        style="Settings.TButton",
        cursor="hand2"
    )
    routines_btn.pack(side="left", padx=5)

    reset_btn = ttk.Button(
        settings_frame,
        text="Reset settings",
        command=lambda: reset_to_defaults(settings_window),
        style="Reset.TButton",
        cursor="hand2"
    )
    reset_btn.pack(anchor='center', pady=(20,10))
    
    def get_total_publications(db_path=DB_PATH):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM metadatas")
            count = c.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            update_status(f"[ERROR] Unable to count publications: {e}")
            return 0

    total_pubs = get_total_publications()

    label_total_pubs = ttk.Label(
        settings_frame,
        text=f"Total publications in database: {total_pubs}",
        style='TLabel'
    )
    label_total_pubs.pack(anchor='center', pady=(15,0))

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
    'Refine.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 12),
        'padding': 0.5
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
    },'Reset.TButton': {
        'background': '#A52A2A',      
        'foreground': 'white',
        'font': ('Segoe UI', 12, 'bold'),
        'padding': 2
    },
    'Settings.TButton': {
        'background': '#599258',      
        'foreground': 'white',
        'font': ('Segoe UI', 12, 'bold'),
        'padding': 2
    },
    'TCheckbutton': {
        'background': '#323232',
        'foreground': 'white',
        'font': ('Segoe UI', 13),
        'focuscolor': '#323232',
        'indicatorcolor': '#599258'
    }
}

for style_name, app_config in style_config.items():
    style.configure(style_name, **app_config)

style.map('Green.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map('Refine.TButton',
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

style.map("Reset.TButton",
          background=[('active', '#5C0000'), ('pressed', '#3E0000')],
          foreground=[('disabled', '#d9d9d9')])

style.map('Settings.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

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

btn_list = ttk.Button(
    input_button_frame,
    text="Search online",
    command=fetcher_ss,
    style='Green.TButton'
)

btn_graph = ttk.Button(
    input_button_frame,
    text="Generate a graph",
    command=on_sim_graph,
    style='Green.TButton'
)

btn_db = ttk.Button(
    input_button_frame,
    text="Get from the database",
    command=get_from_base,
    style='Green.TButton'
)

btn_list.pack(side="left", padx=(0, 5), pady=(0, 0), fill=tk.X, expand=True)
btn_graph.pack(side="left", padx=(0, 5), pady=(0, 0), fill=tk.X, expand=True)
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
# Refine

style.configure("Refine.TEntry",
                fieldbackground="#1E1E1E",  # fond noir identique aux Text
                foreground="white",          # texte blanc
                font=('Segoe UI', 13),
                bordercolor="#1E1E1E",
                borderwidth=3)

refine_frame = ttk.Frame(main_frame, style='TFrame')
refine_frame.pack(fill=tk.X, pady=(5,5))

# Champ texte pour mot-clé
refine_keyword = tk.StringVar()
refine_entry = tk.Entry(refine_frame, textvariable=refine_keyword,
                        width=30,
                        bg="#1E1E1E", fg="white",
                        insertbackground="white",
                        relief="flat",  # supprime le contour
                        font=('Segoe UI', 13))
refine_entry.pack(side=tk.LEFT, padx=(0,5))


# Tri par année
refine_year_order = tk.StringVar(value="Relevance")

year_sort_menu = tk.OptionMenu(refine_frame, refine_year_order, "Relevance", "Oldest first", "Newest first")
year_sort_menu.config(
    bg="#323232",
    fg="white",
    font=('Segoe UI', 13),
    activebackground="#323232",
    activeforeground="white",
    relief="flat",
    highlightthickness=0
)
year_sort_menu["menu"].config(
    bg="#1E1E1E",
    fg="white",
    font=('Segoe UI', 13)
)
year_sort_menu.pack(side=tk.LEFT, padx=(0,5))

# Bouton appliquer
def apply_refine_filters():
    global displayed_articles
    filtered = initial_articles.copy()  # toujours partir de la liste complète
    
    keyword = refine_keyword.get().strip().lower()
    if keyword:
        filtered = [a for a in filtered if keyword in a["title"].lower() or keyword in a["field"].lower() or keyword in a["abstract"].lower()]

    year_order = refine_year_order.get()
    if year_order == "Oldest first":
        filtered.sort(key=lambda x: int(x["year"]) if x["year"].isdigit() else 0)
    elif year_order == "Newest first":
        filtered.sort(key=lambda x: int(x["year"]) if x["year"].isdigit() else 0, reverse=True)
    elif year_order == "Relevance":
        filtered.sort(key=lambda x: x["_initial_pos"])
    
    displayed_articles = filtered.copy()
    update_displayed_articles(displayed_articles)

refine_btn = ttk.Button(refine_frame, text="Refine", command=apply_refine_filters, style="Refine.TButton")
refine_btn.pack(side=tk.LEFT, padx=(5,0))


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

output_context_menu = tk.Menu(text_output, tearoff=0)
output_context_menu.add_command(label="Use as reference for a new search", command=fetcher_selected)
output_context_menu.add_command(label="Use as reference for a new graph", command=generate_graph_with_selection)
output_context_menu.add_command(label="Use as context for the assistant ––WIP––", command=parser)
output_context_menu.add_command(label="Open in external browser", command=open_in_browser)
output_context_menu.add_command(label="Download PDF from Open Access", command=download_selected_pdf)
output_context_menu.add_command(label="Delete from the database", command=delete_selected_publication)

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

btn_settings = ttk.Button(status_help_frame, text="Settings", command=open_settings, style='Bottom.TButton', width=8)
btn_settings.pack(side=tk.LEFT, padx=(0, 5))

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