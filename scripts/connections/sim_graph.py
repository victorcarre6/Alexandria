import networkx as nx
from typing import Optional, List, Set, Dict, Union, Any
import requests, time
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sqlite3
import tkinter as tk

# === Neo4j insertion ===
def insert_relations(tx, doi: str, citations_in: List[str]):
    tx.run("MERGE (p:Paper {doi: $doi})", doi=doi)

    for citer in citations_in:
        tx.run(
            """
            MERGE (citer:Paper {doi: $citer_doi})
            MERGE (p:Paper {doi: $doi})
            MERGE (citer)-[:CITES]->(p)
            """,
            doi=doi, citer_doi=citer
        )

# --- SemanticScholar batch ---

def pages_formatting(pages):
    if not pages:
        return
    return pages.replace("\n", "").replace(" ", "")

def extract_authors(authors_field):
    if not authors_field:
        return []
    return [a.get("name") for a in authors_field if a.get("name")]



tooltip = None

def get_data_in_batch(dois: List[str], batch_size: int = 200, max_workers: int = 4) -> Dict[str, Dict[str, Union[str, List[Any], None]]]:
    """
    Récupère les citations des DOI en batch via Semantic Scholar, en utilisant ThreadPoolExecutor pour paralléliser.
    batch_size : nombre de DOI par requête API (max 500)
    max_workers : nombre de threads simultanés
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {"fields": "externalIds,citations.externalIds"}
    results: Dict[str, Dict[str, Union[str, List[Any], None]]] = {}

    # Split DOI list into batches
    batches = [dois[i:i+batch_size] for i in range(0, len(dois), batch_size)]

    def fetch_batch(batch_dois: List[str]) -> Dict[str, Dict[str, Union[str, List[Any], None]]]:
        batch_results: Dict[str, Dict[str, Union[str, List[Any], None]]] = {}
        try:
            resp = requests.post(url, params=params, json={"ids": batch_dois}, timeout=20)
            resp.raise_for_status()
            papers = resp.json()

            for paper in papers:
                orig_doi = paper.get("externalIds", {}).get("DOI")
                if not orig_doi:
                    continue
                batch_results[orig_doi] = {
                    "doi": orig_doi,
                    "citations": [
                        cited.get("externalIds", {}).get("DOI")
                        for cited in paper.get("citations", [])
                        if cited.get("externalIds", {}).get("DOI")
                    ],
                    "file_name": None
                }
        except Exception as e:
            print(f"[SemanticScholar] Erreur (batch in): {e}")

        # S'assurer que tous les DOI du batch sont présents même s'ils n'ont pas été récupérés
        for d in batch_dois:
            if d not in batch_results:
                batch_results[d] = {"doi": d, "citations": []}

        return batch_results

    # ThreadPoolExecutor pour paralléliser les batches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(fetch_batch, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            batch_res = future.result()
            results.update(batch_res)

    return results


def get_data_from_query(input: str, limit: int = 10) -> Dict[str, Dict[str, Union[str, List[Any], None]]]:
    """
    Récupère les résultats d'une requête textuelle via Semantic Scholar.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query = {"query": input, "limit": limit, "yearFilter": "2010-", "sort": "relevance"}
    params = {"fields": "externalIds,citations.externalIds"}
    results: Dict[str, Dict[str, Union[str, List[Any], None]]] = {}

    try:
        resp = requests.get(url, params={**query, **params}, timeout=20)
        resp.raise_for_status()
        papers = resp.json()

        for paper in papers.get("data", []):
            orig_doi = paper.get("externalIds", {}).get("DOI")
            if not orig_doi:
                continue
            results[orig_doi] = {
                "doi": orig_doi,
                "citations": [
                    cited.get("externalIds", {}).get("DOI")
                    for cited in paper.get("citations", [])
                    if cited.get("externalIds", {}).get("DOI")
                ],
                "file_name": None
            }
    except Exception as e:
        print(f"[SemanticScholar] Erreur (query): {e}")

    return results

def insert_in_base(dois: List[str], db_path) -> Dict[str, Dict[str, Union[str, List[Any], None]]]:
    """
    Récupère les métadonnées des DOI en batch via Semantic Scholar et les insère dans SQLite.
    """
    conn = sqlite3.connect(db_path)
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {"fields": "title,fieldsOfStudy,authors,journal,year,abstract,publicationTypes,externalIds,openAccessPdf"}
    data = {"ids": dois}
    results: Dict[str, Dict[str, Union[str, List[Any], None]]] = {}
    
    c = conn.cursor()
    # Crée les tables si nécessaire
    c.execute("""
        CREATE TABLE IF NOT EXISTS metadatas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field TEXT,
            paper_type TEXT,
            title TEXT,
            authors TEXT,
            doi TEXT UNIQUE,
            journal TEXT,
            year TEXT,
            volume TEXT,
            issue TEXT,
            pages TEXT,
            source TEXT
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doi TEXT UNIQUE,
            local_key INTEGER DEFAULT 0,
            pdf_link TEXT,
            file_name TEXT,
            abstract TEXT
        );
    """)
    conn.commit()

    try:
        resp = requests.post(url, params=params, json=data, timeout=20)
        resp.raise_for_status()
        papers = resp.json()

        inserted_count = 0

        for paper in papers:
            orig_doi = paper.get("externalIds", {}).get("DOI")
            if not orig_doi:
                continue

            authors_list = extract_authors(paper.get("authors"))
            fields = paper.get("fieldsOfStudy") or []
            if not isinstance(fields, list):
                fields = []
            publication_types = paper.get("publicationTypes") or []
            if not isinstance(publication_types, list):
                publication_types = []

            pdf_link = paper.get("openAccessPdf", {}).get("url")
            pages = pages_formatting(paper.get("journal", {}).get("pages"))

            # Conversion listes -> chaînes
            fields_str = ", ".join(fields)
            publication_types_str = ", ".join(publication_types)
            authors_str = ", ".join(authors_list)

            entry = {
                "doi": orig_doi,
                "title": paper.get("title") or "",
                "authors": authors_str,
                "field": fields_str,
                "journal": paper.get("journal", {}).get("name") or "",
                "volume": paper.get("journal", {}).get("volume") or "",
                "pages": pages or "",
                "year": str(paper.get("year") or ""),
                "abstract": paper.get("abstract") or "",
                "publicationTypes": publication_types_str,
                "pdf_link": pdf_link or "",
                "file_name": None
            }

            results[orig_doi] = entry

            # Insertion dans SQLite
            try:
                c.execute("""
                    INSERT OR IGNORE INTO metadatas
                    (field, paper_type, title, authors, doi, journal, year, volume, issue, pages, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry["field"],
                    entry["publicationTypes"],
                    entry["title"],
                    entry["authors"],
                    entry["doi"],
                    entry["journal"],
                    entry["year"],
                    entry["volume"],
                    None,
                    entry["pages"],
                    "semantic_scholar"
                ))
                c.execute("""
                    INSERT OR IGNORE INTO paper_data
                    (doi, pdf_link, local_key, file_name, abstract)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entry["doi"],
                    entry["pdf_link"],
                    0,
                    entry["file_name"],
                    entry["abstract"]
                ))
                inserted_count += 1
            except Exception as e:
                print(f"[ERROR] DOI {orig_doi} non inséré: {e}")

        conn.commit()
        print(f"[INFO] {inserted_count} DOI insérés dans SQLite")
    except Exception as e:
        print(f"[SemanticScholar] Erreur (batch in): {e}")
    finally:
        conn.close()

    # S'assurer que tous les DOI sont dans le dict même s'ils n'ont pas été récupérés
    for d in dois:
        if d not in results:
            results[d] = {
                "doi": d,
                "title": None,
                "authors": None,
                "field": None,
                "journal": None,
                "volume": None,
                "pages": None,
                "year": None,
                "publicationTypes": "",
                "pdf_link": None,
                "file_name": None,
                "abstract": None
            }

    return results

def compute_similarity_in_batches(driver, batch_size: int = 1000, alpha: float = 0.5, beta: float = 0.5, min_weight: float = 1.0):
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
    MERGE (a)-[r:SIMILAR_COCIT]->(b)
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
    MERGE (a)-[r:SIMILAR_COUPLING]->(b)
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
    run_batched_query(cocite_query)
    run_batched_query(bib_query)

    # === 3. Score global ===
    with driver.session() as session:
        session.run("""
        MATCH (a:Paper)-[c:SIMILAR_COCIT]->(b:Paper)
        OPTIONAL MATCH (a)-[brel:SIMILAR_COUPLING]->(b)
        WITH a, b,
             coalesce(c.co_citation,0) AS co,
             coalesce(brel.bib_coupling,0) AS bib
        MERGE (a)-[r:SIMILAR_TO]->(b)
        SET r.co_citation = co,
            r.bib_coupling = bib,
            r.weight = $alpha*co + $beta*bib
        """, alpha=alpha, beta=beta)

def build_citation_graph_from_seed(seed_doi: str, depth, pause,
                         driver=None, batch_size=50, max_dois: Optional[int] = None):
    """
    Explore les citations à partir d'un DOI seed, jusqu'à une profondeur donnée.
    Crée les relations CITES dans Neo4j en batch.
    max_dois : nombre maximal de DOI à analyser (None = pas de limite)
    """
    

    total_analyzed = 0
    start_time = time.time()
    citations_dict: Dict[str, Dict[str, Optional[List[str]]]] = {}
    frontier = deque([(seed_doi, 0)])
    seen: Set[str] = set()

    while frontier:
        batch = []
        while frontier and len(batch) < batch_size:
            doi, level = frontier.popleft()
            if doi not in seen and level <= depth:
                # Vérifier la limite max_dois
                if max_dois is not None and total_analyzed >= max_dois:
                    frontier.clear()  # stop exploration
                    break
                batch.append((doi, level))

        if not batch:
            break

        batch_dois = [doi for doi, _ in batch]
        batch_levels = [lvl for _, lvl in batch]

        # Récupération des données en batch
        s_time = time.time()
        batch_data = get_data_in_batch(batch_dois)
        e_time = time.time()
        print(f"[DEBUG] Batch de {len(batch_dois)} DOI récupéré en {e_time - s_time:.2f}s")

        # Préparer les insertions et la prochaine frontière
        dois_to_insert = []
        in_refs_batch = []
        next_frontier = []

        for doi, level in zip(batch_dois, batch_levels):
            t0 = time.time()
            refs = batch_data[doi].get("citations") or []  # <- s'assure que refs est une liste
            seen.add(doi)
            citations_dict[doi] = {"in": [r for r in refs if isinstance(r, str)]}
            dois_to_insert.append(doi)
            in_refs_batch.append(refs)
            total_analyzed += 1
            t1 = time.time()
            
            # Ajouter à la prochaine frontière si profondeur non atteinte
            if level < depth:
                for ref in refs:
                    if ref not in seen:
                        next_frontier.append((ref, level + 1))

            #print(f"[Niveau {level}] {doi} → {len(refs)} citants [{total_analyzed} analyses en {t1-t_start:.1f}s].")

        # Insertion dans Neo4j en batch optimisée avec UNWIND/FOREACH
        if driver and dois_to_insert:
            # Préparer la structure des données pour Cypher
            papers = []
            for doi, citations_in in zip(dois_to_insert, in_refs_batch):
                papers.append({"doi": doi, "citations_in": [c for c in citations_in if c]})
            cypher = """
            UNWIND $papers AS paper
            MERGE (p:Paper {doi: paper.doi})
            FOREACH (citer_doi IN paper.citations_in |
                MERGE (citer:Paper {doi: citer_doi})
                MERGE (citer)-[:CITES]->(p)
            )
            """
            with driver.session() as session:
                session.run(cypher, papers=papers)

        # Pause pour limiter le rythme des requêtes API
        time.sleep(pause)
        
        frontier.extend(next_frontier)
    end_time = time.time()
    print(f"[INFO] Récupération des DOI terminée pour {total_analyzed} publications analysées en {end_time-start_time:.1f}s.")
    return citations_dict
                
                

def build_citation_graph_from_query(seed: str, depth, pause,
                         driver=None, batch_size=50,  max_dois: Optional[int] = None):
    """
    Explore les citations à partir d'un terme de recherche.
    Crée les relations CITES dans Neo4j en batch.
    max_dois : nombre maximal de DOI à analyser (None = pas de limite)
    """
    
    total_analyzed = 0
    start_time = time.time()
    citations_dict: Dict[str, Dict[str, Optional[List[str]]]] = {}
    
    seen: Set[str] = set()

    # Initialisation du batch avec les DOI du premier batch, niveau 0
    first_doi_batch = get_data_from_query(seed, limit=10)
    initial_batch = [(doi, 0) for doi in first_doi_batch.keys()]
    frontier = deque()

    while frontier or initial_batch:
        if initial_batch:
            batch = initial_batch
            initial_batch = []
        else:
            batch = []
            while frontier and len(batch) < batch_size:
                doi, level = frontier.popleft()
                if doi not in seen and level <= depth:
                    # Vérifier la limite max_dois
                    if max_dois is not None and total_analyzed >= max_dois:
                        frontier.clear()  # stop exploration
                        break
                    batch.append((doi, level))

        if not batch:
            break

        batch_dois = [doi for doi, _ in batch]
        batch_levels = [lvl for _, lvl in batch]

        # Récupération des données en batch
        s_time = time.time()
        
        batch_data = get_data_in_batch(batch_dois)
        e_time = time.time()
        print(f"[DEBUG] Batch de {len(batch_dois)} DOI récupéré en {e_time - s_time:.2f}s")

        # Préparer les insertions et la prochaine frontière
        dois_to_insert = []
        in_refs_batch = []
        next_frontier = []

        for doi, level in zip(batch_dois, batch_levels):
            t0 = time.time()
            refs = batch_data[doi].get("citations") or []  # <- s'assure que refs est une liste
            seen.add(doi)
            citations_dict[doi] = {"in": [r for r in refs if isinstance(r, str)]}
            dois_to_insert.append(doi)
            in_refs_batch.append(refs)
            total_analyzed += 1
            t1 = time.time()
            
            # Ajouter à la prochaine frontière si profondeur non atteinte
            if level < depth:
                for ref in refs:
                    if ref and isinstance(ref, str) and ref not in seen:
                        next_frontier.append((ref, level + 1))

            #print(f"[Niveau {level}] {doi} → {len(refs)} citants [{total_analyzed} analyses en {t1-t_start:.1f}s].")

        # Insertion dans Neo4j en batch optimisée avec UNWIND/FOREACH
        if driver and dois_to_insert:
            # Préparer la structure des données pour Cypher
            papers = []
            for doi, citations_in in zip(dois_to_insert, in_refs_batch):
                papers.append({"doi": doi, "citations_in": [c for c in citations_in if c]})
            cypher = """
            UNWIND $papers AS paper
            MERGE (p:Paper {doi: paper.doi})
            FOREACH (citer_doi IN paper.citations_in |
                MERGE (citer:Paper {doi: citer_doi})
                MERGE (citer)-[:CITES]->(p)
            )
            """
            with driver.session() as session:
                session.run(cypher, papers=papers)

        # Pause pour limiter le rythme des requêtes API
        time.sleep(pause)
        
        frontier.extend(next_frontier)
    end_time = time.time()
    print(f"[INFO] Récupération des DOI terminée pour {total_analyzed} publications analysées en {end_time-start_time:.1f}s.")
    return citations_dict


def generate_graph(driver, db_path, seed, depth, max_dois, 
                   pause, batch_size, max_worker, 
                   edge_mode, weight_threshold, 
                   top_n, top_n_cites, top_n_score, top_n_cocit, top_n_coupling):
# Vider la base avant chaque essai
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("[INFO] Base Neo4j vidée pour un nouvel essai")

    if seed.startswith("10."):
        print(f"[INFO-DOI] Generating a semantic graph from {seed}")
        build_citation_graph_from_seed(
            seed_doi=seed,
            depth=depth,
            pause=pause,
            driver=driver,
            batch_size=batch_size,
            max_dois=max_dois
        )
    else:
        print(f"[INFO-QUERY] Generating a semantic graph of {seed}")
        build_citation_graph_from_query(
            seed=seed,
            depth=depth,
            pause=pause,
            driver=driver,
            batch_size=batch_size,
            max_dois=max_dois
        )

    compute_similarity_in_batches(driver)


    # --- Comptage des nœuds Neo4j ---
    with driver.session() as session:
        result = session.run("MATCH (p:Paper) RETURN count(p) AS total_nodes")
        record = result.single()
        total_nodes = record["total_nodes"] if record else 0
        print(f"[INFO] Nombre total de nœuds dans Neo4j : {total_nodes}")

    # === Récupération des relations en fonction du mode ===
    edges = []

    with driver.session() as session:
        if edge_mode in ("CITES", "SIMILAR"):
            cites_result = session.run("""
                MATCH (a:Paper)-[:CITES]->(b:Paper)
                RETURN a.doi AS source, b.doi AS target
                LIMIT $top_n
            """, top_n=top_n_cites)
            edges.extend([(rec["source"], rec["target"], {"type": "CITES"}) for rec in cites_result])
            
        if edge_mode in ("SIMILAR"):
            sim_result = session.run("""
                MATCH (a:Paper)-[r:SIMILAR_TO]->(b:Paper)
                RETURN a.doi AS source, b.doi AS target, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT $top_n
            """, top_n=top_n_score)
            cocit_result = session.run("""
                MATCH (a:Paper)-[:SIMILAR_COCIT]->(b:Paper)
                RETURN a.doi AS source, b.doi AS target
                LIMIT $top_n
            """, top_n=top_n_cocit)
            coupling_result = session.run("""
                MATCH (a:Paper)-[:SIMILAR_COUPLING]->(b:Paper)
                RETURN a.doi AS source, b.doi AS target
                LIMIT $top_n
            """, top_n=top_n_coupling)
            edges.extend([(rec["source"], rec["target"], {"type": "SIMILAR_TO", "weight": rec["weight"]})
                            for rec in sim_result])
            edges.extend([(rec["source"], rec["target"], {"type": "COCIT"}) for rec in cocit_result])
            edges.extend([(rec["source"], rec["target"], {"type": "COUPLING"}) for rec in coupling_result])
            
            

        gathered_edges = list(edges)
        print(f"[DEBUG] edges récupérées: {len(gathered_edges)}")
        
        # --- Filtrage des arêtes et nœuds selon le seuil de poids ---
        
        filtered_edges = []
        for u, v, d in edges:
            if d.get("type") == "SIMILAR_TO":
                weight = d.get("weight")
                if weight is not None and isinstance(weight, (int, float)) and weight >= weight_threshold:
                    filtered_edges.append((u, v, d))
            elif d.get("type") == "CITES" or "COCIT" or "COUPLING":
                # On garde toujours les CITES
                filtered_edges.append((u, v, d))
            
    filtered_edges_list = list(filtered_edges)
    print(f"[DEBUG] edges après filtration: {len(filtered_edges_list)}")

    # 2. Extraire tous les DOI uniques
    filtered_dois = set([u for u, v, d in filtered_edges] + [v for u, v, d in filtered_edges])
    filtered_dois = list(filtered_dois)

    # 3. Récupérer et insérer les métadatas dans SQLite
    top_results = insert_in_base(filtered_dois, db_path)

    # 4. Créer le MultiGraph final avec ces filtered_dois
    G = nx.Graph()
    for u, v, d in filtered_edges:
        G.add_edge(u, v, **d)

    print(f"[INFO] Graphe (MultiGraph) construit (filtré) : "
        f"{len(list(G.nodes()))} noeuds, {len(list(G.edges()))} arêtes.")


    # === Visualisation Matplotlib dans Tkinter ===
    root = tk.Tk()
    root.title(f'Semantic bibliography graph starting from "{seed}"')

    # Connexion SQLite pour métadonnées
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Layout du graphe
    pos = nx.spring_layout(G, weight="weight", k=0.5, iterations=100)

    # Louvain pour la couleur des communautés (uniquement sur arêtes présentes)
    if len(G.nodes()) > 0:
        partition = community_louvain.best_partition(G, weight='weight')
        cmap = cm.get_cmap('tab20', max(partition.values())+1)
        node_colors = [cmap(partition[n]) for n in G.nodes()]
    else:
        partition = {}
        node_colors = []

    # Taille des nœuds selon la force
    strength = {node: sum(d.get("weight", 1.0) for _,_,d in G.edges(node, data=True)) for node in G.nodes()}
    max_strength = max(strength.values()) if strength else 1
    sizes = [150 + 350*(strength[n]/max_strength) for n in G.nodes()]

    # Bordures
    node_border_colors = []
    node_border_widths = []
    for n in G.nodes():
        if n == seed:
            node_border_colors.append("#319A31")
            node_border_widths.append(2)
        else:
            node_border_colors.append("black")
            node_border_widths.append(0.5)

    # Récupérer les métadonnées pour chaque nœud
    node_metadatas = {}
    for n in G.nodes():
        c.execute("""
            SELECT m.title, m.authors, m.year, m.journal, m.paper_type, p.abstract
            FROM metadatas m
            LEFT JOIN paper_data p ON m.doi = p.doi
            WHERE m.doi = ?
        """, (n,))
        row = c.fetchone()
        if row:
            title, authors, year, journal, paper_type, abstract = row
            node_metadatas[n] = {
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "paper_type": paper_type,
                "abstract": abstract
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

    # === Matplotlib ===
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)

    # Edges selon EDGE_MODE
    cites_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "CITES"]
    sim_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "SIMILAR_TO"]
    cocit_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "COCIT"]
    coupling_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "COUPLING"]

    if cites_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=cites_edges,
                                width=1, edge_color="#333333", style="solid")
    if edge_mode == "SIMILAR" and sim_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=sim_edges,
                            width=1, edge_color="#821717", style="solid")
    if edge_mode == "SIMILAR" and cocit_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=cocit_edges,
                            width=0.5, edge_color="#888888", style="solid", alpha=0.8)
    if edge_mode == "SIMILAR" and coupling_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=coupling_edges,
                            width=1, edge_color="#821717", style="solid")

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
            dx = event.xdata - x if event.xdata is not None else 999
            dy = event.ydata - y if event.ydata is not None else 999
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
            if meta.get("type"):
                txt += f"\nType: {meta['type']}"
            if meta.get("abstract"):
                txt += f"\nAbstract: {meta['abstract']}"
            txt += f"\nWeight: {strength[n]:.2f}"
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
                tooltip.label = tk.Label(
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
                tooltip.label.pack(ipadx=4, ipady=2)

            else:
                tooltip.label.config(text=txt)
            # Positionner le tooltip à côté de la souris (offset pour éviter de cacher le curseur)
            tooltip.geometry(f"+{cursor_x+5}+{cursor_y+5}")
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
        if n and n != seed:
            try:
                path = nx.shortest_path(G, source=n, target=seed, weight='weight')
                # Tracer le chemin
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    artist, = ax.plot([x0, x1], [y0, y1], color="#319A31", linewidth=2, zorder=5)
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