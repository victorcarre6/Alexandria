import requests
from neo4j import GraphDatabase
from pyvis.network import Network
import time
from typing import Optional, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Configuration ===
NEO4J_URI = NEO4J_URI
NEO4J_USER = NEO4J_USER
NEO4J_PASSWORD = NEO4J_PASSWORD

PATIENT_ZERO = "10.1002/anie.202402964"

DEPTH = 2
PAUSE = 1
MAX_REFS_PER_DOI = 10
MAX_WORKERS = 5

# === Neo4j Driver ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === Neo4j Functions ===
def insert_papers_and_citations(tx, doi: str, title: Optional[str], references: List[Tuple[str, Optional[str]]]):
    # Insert the main paper with title
    tx.run(
        """
        MERGE (p:Paper {doi: $doi})
        SET p.title = $title
        """,
        doi=doi, title=title
    )
    # Insert referenced papers and relationships in batch with titles
    for cited_doi, cited_title in references:
        tx.run(
            """
            MERGE (cited:Paper {doi: $cited_doi})
            SET cited.title = coalesce(cited.title, $cited_title)
            MERGE (p:Paper {doi: $doi})
            MERGE (p)-[:CITES]->(cited)
            """,
            doi=doi, cited_doi=cited_doi, cited_title=cited_title
        )

# === CrossRef API ===
def get_crossref_references(doi: str) -> Tuple[Optional[str], List[Tuple[str, Optional[str]]]]:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 404:
            print(f"[CrossRef] 404 Not Found pour {doi}, passage au suivant.")
            return None, []
        r.raise_for_status()
        data = r.json()
        message = data.get("message", {})
        title_list = message.get("title", [])
        main_title = title_list[0] if title_list else None
        refs = message.get("reference", [])
        dois_titles = []
        for ref in refs:
            ref_doi = ref.get("DOI")
            ref_title = ref.get("article-title") or ref.get("series-title") or ref.get("volume-title") or None
            if ref_doi:
                dois_titles.append((ref_doi.lower(), ref_title))
        return main_title, dois_titles[:MAX_REFS_PER_DOI]
    except Exception as e:
        print(f"[CrossRef] Erreur pour {doi}: {e}")
        return None, []

# === Graph Building with ThreadPoolExecutor ===
def build_citation_graph(seed_doi: str, depth: int = DEPTH, pause: float = PAUSE):
    seen: Set[str] = set()
    frontier: List[Tuple[str, int]] = [(seed_doi, 0)]

    with driver.session() as session:
        while frontier:
            # Prepare batch for this level
            batch = []
            next_frontier = []
            # Collect all DOIs to process at this iteration
            while frontier:
                doi, level = frontier.pop(0)
                if doi not in seen and level <= depth:
                    batch.append((doi, level))
            if not batch:
                break

            # Parallel fetch references
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_doi = {executor.submit(get_crossref_references, doi): (doi, level) for doi, level in batch}
                results = []
                for future in as_completed(future_to_doi):
                    doi, level = future_to_doi[future]
                    title, references = future.result()
                    results.append((doi, level, title, references))
                    time.sleep(pause)

            # Insert nodes and relationships in Neo4j in batch per DOI
            for doi, level, title, references in results:
                if doi in seen or level > depth:
                    continue
                seen.add(doi)
                print(f"[Niveau {level}] Traitement de {doi} avec {len(references)} références")
                session.execute_write(insert_papers_and_citations, doi, title, references)
                # Prepare next frontier
                if level < depth:
                    for ref_doi, _ in references:
                        if ref_doi not in seen:
                            next_frontier.append((ref_doi, level + 1))

            frontier = next_frontier

    print("[Graphe] Construction terminée.")

# === Main Entrypoint ===
if __name__ == "__main__":
    print(f"[INFO] Profondeur d'exploration : {DEPTH}")
    build_citation_graph(seed_doi=PATIENT_ZERO, depth=DEPTH)

    net = Network(height="750px", width="100%", notebook=True)

    with driver.session() as session:
        query = "MATCH (p:Paper)-[:CITES]->(c:Paper) RETURN p.doi AS from_doi, p.title AS from_title, c.doi AS to_doi, c.title AS to_title LIMIT 100"
        result = session.run(query)
        for record in result:
            from_label = record["from_title"] if record["from_title"] else record["from_doi"]
            to_label = record["to_title"] if record["to_title"] else record["to_doi"]
            net.add_node(record["from_doi"], label=from_label)
            net.add_node(record["to_doi"], label=to_label)
            net.add_edge(record["from_doi"], record["to_doi"])

    net.show("citation_graph.html")
    driver.close()