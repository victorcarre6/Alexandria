from neo4j import GraphDatabase
import networkx as nx
from typing import Optional, List, Set, Tuple, Dict
import asyncio
import aiohttp
import json
from pathlib import Path
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests, time
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import tkinter as tk

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from scripts.metadatas.fetch_sources import get_crossref_metadata, get_unpaywall_pdf, generate_unique_id
# === Connexion bases ===

DB_PATH = "datas/bibliography.db"

keys_path = ROOT_DIR / "keys.json"
with open(keys_path) as f:
    keys = json.load(f)

NEO4J_URI = keys["NEO4J_URI"]
NEO4J_USERNAME = keys["NEO4J_USERNAME"]
NEO4J_PASSWORD = keys["NEO4J_PASSWORD"]
email = keys["EMAIL"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

PATIENT_ZERO = "10.1002/cssc.201900519"
MAX_CITED = 50
MAX_CITING = 50
DEPTH = 4
PAUSE = 1
BATCH_SIZE = 10
MAX_THREADS = 100  # For OpenCitations API concurrency
TOP_N = 40

G: nx.Graph = nx.Graph()

# === Neo4j insertion ===
def insert_relations(tx, doi: str, citations_out: List[str], citations_in: List[str]):
    # Assurer le noeud de base
    tx.run("MERGE (p:Paper {doi: $doi})", doi=doi)


+ Test COVID non-positif. Plutot cool, donc.

    # Références citées
    for ref in citations_out:
        tx.run(
            """
            MERGE (cited:Paper {doi: $cited_doi})
            MERGE (p:Paper {doi: $doi})
            MERGE (p)-[:CITES]->(cited)
            """,
            doi=doi, cited_doi=ref
        )

    # Entrants (papiers citant ce DOI)
    for citer in citations_in:
        tx.run(
            """
            MERGE (citer:Paper {doi: $citer_doi})
            MERGE (p:Paper {doi: $doi})
            MERGE (citer)-[:CITES]->(p)
            """,
            doi=doi, citer_doi=citer
        )

# === OpenCitations API (async) ===
async def get_citations_out(session: aiohttp.ClientSession, doi: str) -> List[str]:
    """DOIs cités par le DOI donné"""
    url = f"https://opencitations.net/index/coci/api/v1/references/{doi}"
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with session.get(url, timeout=timeout) as r:
            r.raise_for_status()
            data = await r.json()
            return [item["cited"] for item in data if "cited" in item][:MAX_CITED]
    except Exception as e:
        print(f"[OpenCitations] Erreur (out) {doi}: {e}")
        return []

async def get_citations_in(session: aiohttp.ClientSession, doi: str) -> List[str]:
    """DOIs citant le DOI donné"""
    url = f"https://opencitations.net/index/coci/api/v1/citations/{doi}"
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with session.get(url, timeout=timeout) as r:
            r.raise_for_status()
            data = await r.json()
            return [item["citing"] for item in data if "citing" in item][:MAX_CITING]
    except Exception as e:
        print(f"[OpenCitations] Erreur (in) {doi}: {e}")
        return []

# === Graphe récursif avec batching séquentiel, collecte OpenCitations en parallèle ===
async def build_citation_graph(seed_doi: str, depth: int = DEPTH, pause: float = PAUSE) -> None:
    import time as _time
    seen: Set[str] = set()
    frontier: List[Tuple[str, int]] = [(seed_doi, 0)]
    
    async with aiohttp.ClientSession() as aio_session:
        with driver.session() as session:

            sem = asyncio.Semaphore(MAX_THREADS)

            async def fetch_citations(doi: str):
                async with sem:
                    out_refs = await get_citations_out(aio_session, doi)
                    in_refs = await get_citations_in(aio_session, doi)
                    await asyncio.sleep(pause)
                    return doi, out_refs, in_refs

            while frontier:
                batch = []
                next_frontier = []
                for _ in range(BATCH_SIZE):
                    if not frontier:
                        break
                    doi, level = frontier.pop(0)
                    if doi not in seen and level <= depth:
                        batch.append((doi, level))

                if not batch:
                    break

                tasks = [fetch_citations(doi) for doi, _ in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                dois_to_insert, out_refs_batch, in_refs_batch, levels_batch = [], [], [], []

                for (i, result) in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"[OpenCitations] Erreur lors de la récupération : {result}")
                        continue
                    # Ici result est bien un tuple (doi, out_refs, in_refs)
                    doi_fetched, out_refs, in_refs = result # type: ignore
                    
                    level_fetched = batch[i][1]
                    seen.add(doi_fetched)
                    print(f"[Niveau {level_fetched}] {doi_fetched} → {len(out_refs)} cités, {len(in_refs)} citants")

                    dois_to_insert.append(doi_fetched)
                    out_refs_batch.append(out_refs)
                    in_refs_batch.append(in_refs)
                    levels_batch.append(level_fetched)

                    # Ajouter les nouveaux DOIs à la prochaine frontière
                    for ref in out_refs + in_refs:
                        if ref not in seen and level_fetched < depth:
                            next_frontier.append((ref, level_fetched + 1))

                # Insérer en batch dans une seule transaction (séquentiel)
                def batch_insert(tx):
                    for doi, citations_out, citations_in in zip(dois_to_insert, out_refs_batch, in_refs_batch):
                        insert_relations(tx, doi, citations_out, citations_in)
                session.execute_write(batch_insert)

                frontier = next_frontier
                
def compute_similarity_in_batches(driver, batch_size: int = 1000, alpha: float = 0.5, beta: float = 0.5, min_weight: float = 0.1):
    """
    Calcule la similarité entre articles et crée des relations :SIMILAR_TO en batch
    alpha : poids pour co-citation
    beta  : poids pour bibliographic coupling
    min_weight : seuil minimal pour créer une relation
    """

    # === 1. Co-citation en batch ===
    cocite_query = """
    MATCH (a:Paper)
    WITH a
    ORDER BY a.doi
    SKIP $skip LIMIT $limit
    MATCH (a)<-[:CITES]-(x:Paper)-[:CITES]->(b:Paper)
    WHERE a.doi < b.doi
    WITH a,b,count(x) AS cocite_count
    MERGE (a)-[r:SIMILAR_TO]->(b)
    SET r.co_citation = cocite_count
    """

    # === 2. Bibliographic coupling en batch ===
    bib_query = """
    MATCH (a:Paper)
    WITH a
    ORDER BY a.doi
    SKIP $skip LIMIT $limit
    MATCH (a)-[:CITES]->(ref:Paper)<-[:CITES]-(b:Paper)
    WHERE a.doi < b.doi
    WITH a,b,count(ref) AS bib_count
    MERGE (a)-[r:SIMILAR_TO]->(b)
    SET r.bib_coupling = bib_count
    """

    # Fonction utilitaire pour exécuter une requête en batch
    def run_batched_query(query: str):
        skip = 0
        while True:
            with driver.session() as session:
                result = session.run(query, skip=skip, limit=batch_size)
                summary = result.consume()
                # Si rien de nouveau n’est écrit → on arrête
                if (summary.counters.properties_set == 0 
                    and summary.counters.relationships_created == 0):
                    break
            skip += batch_size
            print(f"[BATCH] Traité jusqu’à {skip} nœuds.")

    print("[INFO] Calcul des co-citations...")
    run_batched_query(cocite_query)

    print("[INFO] Calcul du bibliographic coupling...")
    run_batched_query(bib_query)

    # === 3. Score global ===
    with driver.session() as session:
        session.run("""
        MATCH (a:Paper)-[r:SIMILAR_TO]->(b:Paper)
        WITH r,
             coalesce(r.co_citation,0) AS co,
             coalesce(r.bib_coupling,0) AS bib
        WITH r, $alpha*co + $beta*bib AS score
        SET r.weight = score
        """, alpha=alpha, beta=beta)

        # === 4. Supprimer les faibles liens ===
        session.run("""
        MATCH ()-[r:SIMILAR_TO]->()
        WHERE r.weight < $min_weight
        DELETE r
        """, min_weight=min_weight)

    print("[INFO] Relations SIMILAR_TO calculées et insérées (batch).")

# === Ajout des TOP_N publications en base SQLite ===



def insert_top_n_into_sqlite(top_edges: List[Tuple[str, str, float]], email: str, db_path=DB_PATH):
    """
    Ajoute dans SQLite les publications présentes dans les TOP_N du graphe de similarité.
    top_edges : liste de tuples (source_doi, target_doi, weight)
    Applique la même logique d'insertion/enrichissement que fetch_sources.py.
    """
    

    # 1. Collecte tous les DOI uniques
    dois = set()
    for src, tgt, _ in top_edges:
        dois.add(src)
        dois.add(tgt)
    dois = list(dois)

    # 2. Connexion SQLite et création table si besoin
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS metadatas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_id TEXT UNIQUE,
            subject TEXT,
            title TEXT,
            authors TEXT,
            doi TEXT,
            journal TEXT,
            year TEXT,
            volume TEXT,
            issue TEXT,
            pages TEXT,
            source TEXT,
            local_key INTEGER DEFAULT 0,
            pdf_link TEXT,
            file_name TEXT
        );
    """)
    conn.commit()

    # 3. Vérifie les DOI déjà présents (local_key=1 ou non)
    c.execute("SELECT doi, local_key FROM metadatas WHERE doi IN ({})".format(",".join("?"*len(dois))), dois)
    present = {row[0]: row[1] for row in c.fetchall() if row[0]}
    # DOI à enrichir: pas présent OU local_key=0
    dois_to_enrich = [doi for doi in dois if doi not in present or present[doi] == 0]

    # 4. Prépare la liste des articles à enrichir (par DOI)
    articles_for_enrich: List[Dict] = []
    for doi in dois_to_enrich:
        articles_for_enrich.append({"doi": doi, "source": "similarity_graph"})

    # 5. Enrichissement Crossref + Unpaywall en parallèle
    def enrich_article(article):
        doi = article.get("doi")
        crossref_data = get_crossref_metadata(doi)
        unpaywall_data = get_unpaywall_pdf(doi, email)
        # Extraction des champs Crossref
        title = None
        authors = None
        year = None
        journal = None
        volume = None
        issue = None
        pages = None
        if crossref_data:
            # crossref_data peut contenir 'title', 'author', 'issued', etc.
            # On refait la requête pour plus de champs si possible
            # get_crossref_metadata retourne déjà journal/volume/issue/pages
            # Pour le titre et les auteurs, on refait la requête complète
            url = f"https://api.crossref.org/works/{doi}"
            
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                time.sleep(0.05)
                data = r.json().get("message", {})
                title = data.get("title", [None])[0]
                authors = ", ".join([
                    a.get("family", "") + (" " + a.get("given", "") if a.get("given") else "")
                    for a in data.get("author", [])]) if data.get("author") else None
                issued = data.get("issued", {}).get("date-parts", [[None]])
                year = str(issued[0][0]) if issued and issued[0][0] else None
                journal = crossref_data.get("journal")
                volume = crossref_data.get("volume")
                issue = crossref_data.get("issue")
                pages = crossref_data.get("pages")
            except Exception:
                pass
        pdf_link = unpaywall_data.get("pdf_link") if unpaywall_data else None
        if not pdf_link:
            print(f"[WARN] Aucun PDF trouvé pour {doi}")
        # unique_id: basé sur le titre si dispo, sinon DOI
        if title:
            uid = generate_unique_id({"title": title})
        else:
            # fallback: hash du DOI
            uid = hashlib.md5((doi or "").encode("utf-8")).hexdigest()
        return {
            "unique_id": uid,
            "subject": None,
            "title": title,
            "authors": authors,
            "doi": doi,
            "journal": journal,
            "year": year,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "source": "similarity_graph",
            "pdf_link": pdf_link,
            "file_name": None
        }

    results = []
    if articles_for_enrich:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(enrich_article, art) for art in articles_for_enrich]
            for future in as_completed(futures):
                results.append(future.result())

    # 6. Insertion ou mise à jour dans SQLite
    for article in results:
        # Insertion ou update (si déjà en base mais local_key=0)
        c.execute("""
            INSERT OR IGNORE INTO metadatas
            (unique_id, subject, title, authors, doi, journal, year, volume, issue, pages, source, pdf_link, local_key, file_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article["unique_id"],
            article["subject"],
            article["title"],
            article["authors"],
            article["doi"],
            article["journal"],
            article["year"],
            article["volume"],
            article["issue"],
            article["pages"],
            article["source"],
            article["pdf_link"],
            1,
            article["file_name"]
        ))
        # Si déjà présent mais local_key=0, on met à jour tous les champs
        c.execute("""
            UPDATE metadatas
            SET subject=?, title=?, authors=?, journal=?, year=?, volume=?, issue=?, pages=?, source=?, pdf_link=?, local_key=1, file_name=?
            WHERE doi=? AND (local_key=0 OR local_key IS NULL)
        """, (
            article["subject"],
            article["title"],
            article["authors"],
            article["journal"],
            article["year"],
            article["volume"],
            article["issue"],
            article["pages"],
            article["source"],
            article["pdf_link"],
            article["file_name"],
            article["doi"]
        ))
    conn.commit()
    conn.close()
    print(f"[INFO] {len(dois)} publications ajoutées ou enrichies dans SQLite depuis Neo4j.")

# === Main ===

if __name__ == "__main__":

    # Vider la base avant chaque essai
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("[INFO] Base Neo4j vidée pour un nouvel essai")

    print(f"[INFO] Exploration OpenCitations profondeur : {DEPTH}")
    
    asyncio.run(build_citation_graph(seed_doi=PATIENT_ZERO, depth=DEPTH))
    compute_similarity_in_batches(driver)

    # Récupération des relations depuis Neo4j  

    with driver.session() as session:
        result = session.run("""
        MATCH (a:Paper)-[r:SIMILAR_TO]->(b:Paper)
        RETURN a.doi AS source, b.doi AS target, r.weight AS weight
        """)
        edges = [(rec["source"], rec["target"], rec["weight"]) for rec in result]
        
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:TOP_N]

    G = nx.Graph()
    for u,v,w in edges:
        G.add_edge(u,v,weight=w)
        
    insert_top_n_into_sqlite(edges, email=email)

    print(f"[INFO] Graphe de similarité construit : "
        f"{len(list(G.nodes()))} noeuds, {len(list(G.edges()))} arêtes (top {TOP_N}).")

    # === Visualisation Matplotlib dans Tkinter ===
    root = tk.Tk()
    root.title("Graphe de similarité (Tkinter + Matplotlib)")

    # Connexion SQLite pour métadonnées
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Layout du graphe
    pos = nx.spring_layout(G, weight="weight", k=0.5, iterations=100)

    # Louvain pour la couleur des communautés
    partition = community_louvain.best_partition(G, weight='weight')
    cmap = cm.get_cmap('tab20', max(partition.values())+1)
    node_colors = [cmap(partition[n]) for n in G.nodes()]

    # Taille des nœuds selon la force
    strength = {node: sum(d["weight"] for _,_,d in G.edges(node, data=True)) for node in G.nodes()}
    max_strength = max(strength.values()) if strength else 1
    sizes = [150 + 350*(strength[n]/max_strength) for n in G.nodes()]

    # Bordures
    node_border_colors = []
    node_border_widths = []
    for n in G.nodes():
        if n == PATIENT_ZERO:
            node_border_colors.append("red")
            node_border_widths.append(3)
        else:
            node_border_colors.append("black")
            node_border_widths.append(0.5)

    # Récupérer les métadonnées pour chaque nœud
    node_metadatas = {}
    for n in G.nodes():
        c.execute("SELECT title, authors, year, journal FROM metadatas WHERE doi = ?", (n,))
        row = c.fetchone()
        if row:
            title, authors, year, journal = row
            node_metadatas[n] = {
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal
            }
        else:
            node_metadatas[n] = {}

    # Libellé du nœud: premier auteur + année
    node_labels = {}
    for n in G.nodes():
        meta = node_metadatas[n]
        label = ""
        if meta.get("authors"):
            first_author = meta["authors"].split(",")[0].strip()
            label = first_author
        if meta.get("year"):
            label += f" {meta['year']}" if label else meta["year"]
        node_labels[n] = label if label else n[:8]

    # Figure matplotlib
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    # Edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.2, edge_color="#888888")
    # Nodes
    nodes_artist = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=sizes,
        edgecolors=node_border_colors, linewidths=node_border_widths
    )
    # Labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()

    # Tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    # Label pour info détaillée (désormais inutilisé)
    info_label = tk.Label(root, text="Survolez un nœud pour voir les détails.", anchor="w", justify="left", wraplength=700)
    info_label.pack(fill=tk.X, padx=10, pady=5)

    # Tooltip flottant à côté de la souris
    tooltip = None

    # Mapping matplotlib (pixels) <-> nœud pour survol/clic
    node_coords = {}
    for n in G.nodes():
        x, y = pos[n]
        # matplotlib scatter: data coordinates, need to transform to display coords
        node_coords[n] = (x, y)

    # Utilitaire: trouver le nœud le plus proche d'un point (en axes coords)
    def get_node_under_cursor(event):
        if event.inaxes != ax:
            return None
        # event.xdata, event.ydata: axes coords
        min_dist = float("inf")
        closest = None
        for n, (x, y) in node_coords.items():
            dx = event.xdata - x if event.xdata else 999
            dy = event.ydata - y if event.ydata else 999
            dist = (dx*dx + dy*dy)**0.5
            r = (sizes[list(G.nodes()).index(n)]**0.5)/140 # approx radius in axes
            if dist < r and dist < min_dist:
                min_dist = dist
                closest = n
        return closest

    # Affichage des infos au survol avec cadre flottant
    def on_motion(event):
        global tooltip
        n = get_node_under_cursor(event)
        if n:
            meta = node_metadatas[n]
            txt = f"DOI: {n}"
            if meta.get("title"):
                txt += f"\nTitre: {meta['title']}"
            if meta.get("authors"):
                txt += f"\nAuteurs: {meta['authors']}"
            if meta.get("year"):
                txt += f"\nAnnée: {meta['year']}"
            if meta.get("journal"):
                txt += f"\nJournal: {meta['journal']}"
            txt += f"\nStrength: {strength[n]:.2f}"
            # Afficher le tooltip flottant près du curseur
            # Obtenir la position du curseur dans la fenêtre racine
            if event.guiEvent is not None:
                cursor_x = event.guiEvent.x_root
                cursor_y = event.guiEvent.y_root
            else:
                # Fallback: placer au centre
                cursor_x = root.winfo_pointerx()
                cursor_y = root.winfo_pointery()
            # Créer ou mettre à jour le tooltip
            if tooltip is None or not tooltip.winfo_exists():
                tooltip = tk.Toplevel(root)
                tooltip.wm_overrideredirect(True)
                tooltip.attributes("-topmost", True)
                # Style modifié: texte noir sur fond blanc, contour noir
                label = tk.Label(
                    tooltip,
                    text=txt,
                    justify="left",
                    bg="white",
                    fg="black",
                    relief="solid",
                    borderwidth=1,
                    wraplength=400,
                    font=("Arial", 10)
                )
                label.pack(ipadx=4, ipady=2)
                tooltip.label = label
            else:
                tooltip.label.config(text=txt)
            # Positionner le tooltip à côté de la souris (offset pour éviter de cacher le curseur)
            tooltip.geometry(f"+{cursor_x+15}+{cursor_y+10}")
        else:
            # Masquer le tooltip si visible
            if tooltip is not None and tooltip.winfo_exists():
                tooltip.destroy()
                tooltip = None

    # Clic pour mettre en évidence le chemin vers PATIENT_ZERO
    highlight_artists = []
    def on_click(event):
        n = get_node_under_cursor(event)
        # Nettoyer le précédent chemin
        for artist in highlight_artists:
            try:
                artist.remove()
            except Exception:
                pass
        highlight_artists.clear()
        if n and n != PATIENT_ZERO:
            try:
                path = nx.shortest_path(G, source=n, target=PATIENT_ZERO, weight='weight')
                # Tracer le chemin
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    artist, = ax.plot([x0, x1], [y0, y1], color="red", linewidth=3, zorder=5)
                    highlight_artists.append(artist)
                fig.canvas.draw_idle()
            except nx.NetworkXNoPath:
                info_label.config(text=info_label.cget("text") + "\nAucun chemin vers PATIENT_ZERO")

    # Connecter événements
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Fermer la connexion SQLite à la fermeture
    def on_closing():
        try:
            conn.close()
        except Exception:
            pass
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()

"""
Ca fonctionne bien dans l'idée, en fournissant les top_n papiers les plus pertinents en fonction du patient_zero.
Les noeud sont ensuite regroupées par clusters, avec un poids (pondération similarité+citations communes)
Il manque une optimisation de la visualisation:
 	1. Premier auteur + année sur les noeuds
 	2. Affichage des métadonnées (titre, auteurs, journal, abstract?) au hover
 	3. Couleur du nœud = année de publication
Interactivité nécessaire entre Neo4j et SQLite : intégrer les top_n papiers dans la base via les fetchers.
"""