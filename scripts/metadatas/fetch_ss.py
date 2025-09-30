from typing import Optional, List, Set, Dict, Union, Any
import requests, time
import sqlite3
import xml.etree.ElementTree as ET
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

def pages_formatting(pages):
    if not pages:
        return
    return pages.replace("\n", "").replace(" ", "")

def extract_authors(authors_field):
    if not authors_field:
        return []
    return [a.get("name") for a in authors_field if a.get("name")]


def get_last_papers_from_query(input: str, limit: int = 10) -> Dict[str, Dict[str, Union[str, List[Any], None]]]:
    """
    Récupère les résultats d'une requête textuelle via Semantic Scholar.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query = {"query": input, "limit": limit, "yearFilter": "2010-", "sort": "relevance"}
    params = {"fields": "title,fieldsOfStudy,authors,journal,year,abstract,publicationTypes,externalIds,openAccessPdf"}
    results: Dict[str, Dict[str, Union[str, List[Any], None]]] = {}

    try:
        resp = requests.get(url, params={**query, **params}, timeout=360)
        resp.raise_for_status()
        papers = resp.json()

        for paper in papers.get("data", []):
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
            
            results[orig_doi] = {
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
    except Exception as e:
        print(f"[SemanticScholar] Erreur (query): {e}")

    return results

def get_unpaywall_pdf(doi, email):
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        time.sleep(0.05)
        data = r.json()

        pdf_url = None
        if data.get("best_oa_location"):
            pdf_url = data["best_oa_location"].get("url_for_pdf")
        return {"pdf_link": pdf_url}

    except Exception as e:
        return None

def insert_articles_into_sqlite(articles, email, db_path="datas/bibliography.db"):
    """
    Insère les articles directement dans SQLite et enrichit avec Crossref.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
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
            doi TEXT,
            local_key INTEGER DEFAULT 0,
            pdf_link TEXT,
            file_name TEXT,
            abstract TEXT
        );
    """)
    conn.commit()

    # Insertion initiale
    for article in articles:
        c.execute("""
            INSERT OR IGNORE INTO metadatas
            (field, paper_type, title, authors, doi, journal, year, volume, issue, pages, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article.get('field'),
            article.get('paper_type'),
            article.get('title'),
            str(article.get('authors')),
            article.get('doi'),
            article.get('journal'),
            article.get('year'),
            article.get('volume'),
            article.get('issue'),
            article.get('pages'),
            article.get('source')
        ))
        c.execute("""
            INSERT OR IGNORE INTO paper_data
            (doi, local_key, pdf_link, file_name, abstract)
            VALUES (?, ?, ?, ?, ?)
        """, (
            article.get('doi'),
            0,
            article.get('pdf_link'),
            None,
            article.get('abstract')
        ))
    conn.commit()

    # Récupération des pdf_link si absents
    c.execute("""
        SELECT m.doi, m.source
        FROM metadatas m
        JOIN paper_data p ON m.doi = p.doi
        WHERE m.doi IS NOT NULL AND p.local_key = 0
    """)
    doi_source_pairs = c.fetchall()
    articles_for_enrich = [{"doi": row[0], "source": row[1]} for row in doi_source_pairs]

    # Enrichissement avec Unpaywall
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(get_unpaywall_pdf, article["doi"], email): article["doi"]
            for article in articles_for_enrich
        }
        for future in as_completed(futures):
            doi = futures[future]
            try:
                unpaywall_data = future.result()
                results.append((doi, unpaywall_data))
            except Exception as e:
                print(f"[Unpaywall] Erreur pour {doi}: {e}")

    # Mise à jour SQLite dans le thread principal
    for doi, unpaywall_data in results:
        if unpaywall_data:
            pdf_url = unpaywall_data.get("pdf_link")
            c.execute("""
                UPDATE paper_data
                SET pdf_link=?, local_key=1
                WHERE doi=?
            """, (pdf_url, doi))
            
    conn.commit()
    conn.close()