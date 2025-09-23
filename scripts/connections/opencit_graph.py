import requests
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from typing import Optional, List, Set, Tuple
import itertools
import asyncio
import aiohttp

# === Configuration ===
NEO4J_URI = NEO4J_URI
NEO4J_USER = NEO4J_USER
NEO4J_PASSWORD = NEO4J_PASSWORD

PATIENT_ZERO = "10.1002/cssc.201900519"
MAX_CITED = 30
MAX_CITING = 30
DEPTH = 6
PAUSE = 1
BATCH_SIZE = 10
MAX_THREADS = 50  # For OpenCitations API concurrency

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    max_connection_lifetime=3600,  # 1 hour, adjust as needed
    max_connection_pool_size=20,   # adjust as needed for your workload
    keep_alive=True
)

# === Neo4j insertion ===
def insert_relations(tx, doi: str, citations_out: List[str], citations_in: List[str]):
    # Assurer le noeud de base
    tx.run("MERGE (p:Paper {doi: $doi})", doi=doi)

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

# === OpenCitations API ===

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

async def build_citation_graph(seed_doi: str, depth: int = DEPTH, pause: float = PAUSE):
    import time as _time
    seen: Set[str] = set()
    frontier: List[Tuple[str, int]] = [(seed_doi, 0)]
    start_time = _time.time()

    async with aiohttp.ClientSession() as aio_session:
        with driver.session() as session:
            while frontier:
                # Prendre un batch de BATCH_SIZE DOIs à traiter
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

                dois_to_insert = []
                out_refs_batch = []
                in_refs_batch = []
                levels_batch = []

                # Asynchronous fetching
                sem = asyncio.Semaphore(MAX_THREADS)
                async def fetch_citations(doi, level):
                    async with sem:
                        out_refs = await get_citations_out(aio_session, doi)
                        in_refs = await get_citations_in(aio_session, doi)
                        await asyncio.sleep(pause)
                        return (doi, level, out_refs, in_refs)

                tasks = [fetch_citations(doi, level) for doi, level in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        print(f"[OpenCitations] Erreur lors de la récupération : {result}")
                        continue
                    doi_fetched, level_fetched, out_refs, in_refs = result
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

    elapsed = _time.time() - start_time
    print(f"[Graphe OpenCitations] Construction terminée. Temps total : {elapsed:.2f} secondes.")

# === Main ===
if __name__ == "__main__":
    print(f"[INFO] Exploration OpenCitations profondeur : {DEPTH}")
    asyncio.run(build_citation_graph(seed_doi=PATIENT_ZERO, depth=DEPTH))

    # Récupération des relations depuis Neo4j
    with driver.session() as session:
        query = """
        MATCH (p:Paper)-[r:CITES]->(q:Paper)
        RETURN p.doi AS from, q.doi AS to
        """
        result = session.run(query)

        # Construire le graphe orienté
        G = nx.DiGraph()
        edge_colors = []
        edge_list = []
        for record in result:
            f = record["from"]
            t = record["to"]
            G.add_node(f)
            G.add_node(t)
            # Ajouter l'edge DOI → référence (cité), vert
            G.add_edge(f, t)
            edge_list.append((f, t))
            edge_colors.append('green')
            # Ajouter l'edge référence → DOI (citant), rouge
            G.add_edge(t, f)
            edge_list.append((t, f))
            edge_colors.append('red')

    # Visualisation avec Tkinter et matplotlib
    root = tk.Tk()
    root.title("Graphe de citations (OpenCitations/Neo4j)")

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=150, node_color="#ccccff")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    # Les edges sont colorés en fonction de la liste edge_colors
    nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=edge_colors, arrows=True, ax=ax, arrowsize=12, width=1.3)
    ax.set_axis_off()
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    root.mainloop()