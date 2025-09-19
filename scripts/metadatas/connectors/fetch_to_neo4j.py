from neo4j import GraphDatabase
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2].parent)) 

from scripts.metadatas.gathering.fetch_sources import fetch_pubmed, fetch_arxiv, fetch_EuropePMC
from scripts.metadatas.gathering.fetch_sources import generate_unique_id, get_doi_from_title

# Paramètres de connexion Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_paper_node(tx, article):
    tx.run(
        """
        MERGE (p:Paper {unique_id: $unique_id})
        SET p.title = $title,
            p.authors = $authors,
            p.year = $year,
            p.journal = $journal,
            p.doi = $doi,
            p.source = $source
        """,
        unique_id=article["unique_id"],
        title=article["title"],
        authors=", ".join(article.get("authors", [])),
        year=article.get("year"),
        journal=article.get("journal"),
        doi=article.get("doi"),
        source=article.get("source")
    )

def fetch_and_store(subject="photochemistry", year_from=2015, year_to=2026, max_results=30):
    divided_max_results = round(max_results / 3)
    sources = [
        fetch_pubmed(subject, year_from, year_to, limit=divided_max_results),
        fetch_EuropePMC(subject, year_from, year_to, limit=divided_max_results),
        fetch_arxiv(subject, year_from, year_to, limit=divided_max_results)
    ]

    articles = []
    for source_gen in sources:
        for article in source_gen:
            # Générer unique_id
            article["unique_id"] = generate_unique_id(article)
            # Remplir DOI si manquant
            if not article.get("doi"):
                article["doi"] = get_doi_from_title(article["title"])
            articles.append(article)

    # Insertion dans Neo4j
    with driver.session() as session:
        for article in articles:
            session.write_transaction(create_paper_node, article)
    print(f"{len(articles)} articles insérés dans Neo4j.")

if __name__ == "__main__":
    fetch_and_store()