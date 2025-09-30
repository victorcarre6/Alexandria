import requests
import sqlite3
import time
import xml.etree.ElementTree as ET
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

max_results = 15
divided_max_results = round(max_results/3)
field = "4CzIPN"
year_from = 2015
year_to = 2026
email = "victorcarre@icloud.com"

# --- RECUPERATION DES PUBLICATIONS BRUTES ---


"""#TODO
Ajout d'une fonction de fetch depuis HAL
"""

def fetch_pubmed(field, year_from, year_to, limit=None, page_size=100):
    """
    Récupère les articles PubMed par mot-clé 'field' avec pagination.
    - limit : nombre total d'articles souhaité
    - page_size : nombre d'articles par requête
    """
    # Connexion à la base pour vérifier les articles existants
    db_path = "datas/bibliography.db"
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
    c.execute("SELECT COUNT(*) FROM metadatas WHERE field = ? AND source = 'PubMed'", (field,))
    existing_count = c.fetchone()[0]
    conn.close()
    limit = limit or divided_max_results
    limit = int(limit)
    target_count = limit + existing_count
    fetch_limit = target_count * 2
    print(f"[PubMed] {existing_count} articles déjà présents en base pour '{field}'. Nouvelle limite de récupération : {fetch_limit}.")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    total_retrieved = 0
    start = 0
    articles_yielded = 0
    while articles_yielded < limit:
        retmax = min(page_size, fetch_limit - total_retrieved)
        params = {
            "db": "pubmed",
            "term": field,
            "retmax": retmax,
            "retstart": start,
            "retmode": "json",
            "sort": "relevance",
            "mindate": year_from,
            "maxdate": year_to
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            time.sleep(0.05)
        except Exception as e:
            print(f"[PubMed] Erreur durant la requête: {e}")
            break
        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            break
        for pmid in id_list:
            try:
                summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                sum_params = {"db": "pubmed", "id": pmid, "retmode": "json"}
                sum_resp = requests.get(summary_url, params=sum_params)
                sum_resp.raise_for_status()
                time.sleep(0.05)
                summary = sum_resp.json().get("result", {}).get(pmid, {})
            except Exception as e:
                print(f"[PubMed] Erreur récupération résumé pour PMID {pmid}: {e}")
                continue
            doi = summary.get("elocationid")
            if doi:
                doi = doi.replace("doi:", "").strip()
            year = summary.get("pubdate")
            if year:
                year = year[:4]
            article = {
                "title": summary.get("title"),
                "authors": [a["name"] for a in summary.get("authors", [])],
                "field": field,
                "doi": doi,
                "year": year,
                "journal": None,
                "volume": None,
                "issue": None,
                "pages": None,
                "source": "PubMed",
                "pdf_link": None
            }
            # Vérifier si déjà en base (par titre)
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM metadatas WHERE field = ? AND source = 'PubMed' AND title = ?", (field, article["title"]))
            if c.fetchone()[0] == 0:
                yield article
                articles_yielded += 1
                if articles_yielded >= limit:
                    conn.close()
                    break
            conn.close()
        retrieved_count = len(id_list)
        total_retrieved += retrieved_count
        start += retrieved_count
        if articles_yielded >= limit:
            break

def fetch_arxiv(field, year_from, year_to, limit=divided_max_results):
    """
    Récupère les articles ArXiv par mot-clé 'field' et plage d'années.
    """
    db_path = "datas/bibliography.db"
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
    c.execute("SELECT COUNT(*) FROM metadatas WHERE field = ? AND source = 'ArXiv'", (field,))
    existing_count = c.fetchone()[0]
    conn.close()
    limit = limit or divided_max_results
    limit = int(limit)
    target_count = limit + existing_count
    fetch_limit = target_count * 2
    print(f"[ArXiv] {existing_count} articles déjà présents en base pour '{field}'. Nouvelle limite de récupération : {fetch_limit}.")
    base_url = "http://export.arxiv.org/api/query"
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }

    total_retrieved = 0
    start = 0
    page_size = min(100, fetch_limit)  # arXiv limite max 100 par requête
    articles_yielded = 0

    while articles_yielded < limit:
        search_query = f"all:{field} AND submittedDate:[{year_from}0101 TO {year_to}1231]"
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": page_size
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            time.sleep(0.05)
        except Exception as e:
            print(f"[ArXiv] Erreur durant la requête: {e}")
            break

        root = ET.fromstring(response.content)
        entries = root.findall("atom:entry", ns)
        if not entries:
            print("[ArXiv] Aucun article trouvé ou fin des résultats.")
            break

        for entry in entries:
            title = entry.find("atom:title", ns).text.strip()
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]

            pdf_link = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"]

            doi_elem = entry.find("arxiv:doi", ns)
            doi = doi_elem.text.strip() if doi_elem is not None else None

            pub_date_elem = entry.find("atom:published", ns)
            date = pub_date_elem.text.split("T")[0] if pub_date_elem is not None else None
            year = date[:4] if date else None

            journal_elem = entry.find("arxiv:journal_ref", ns)
            journal_ref = journal_elem.text.strip() if journal_elem is not None else None

            article = {
                "title": title,
                "authors": authors,
                "field": field,
                "doi": doi,
                "date": date,
                "year": year,
                "journal": journal_ref,
                "volume": None,
                "issue": None,
                "pages": None,
                "source": "ArXiv",
                "pdf_link": pdf_link
            }
            # Vérifier si déjà en base (par titre)
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM metadatas WHERE field = ? AND source = 'ArXiv' AND title = ?", (field, title))
            if c.fetchone()[0] == 0:
                yield article
                articles_yielded += 1
                if articles_yielded >= limit:
                    conn.close()
                    break
            conn.close()

        retrieved_count = len(entries)
        total_retrieved += retrieved_count
        start += retrieved_count
        if articles_yielded >= limit:
            break

def fetch_EuropePMC(field, year_from, year_to, limit=None, retries=3, delay=5):
    """
    Récupère les articles depuis Europe PMC avec retry automatique.
    """
    db_path = "datas/bibliography.db"
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
    c.execute("SELECT COUNT(*) FROM metadatas WHERE field = ? AND source = 'EuropePMC'", (field,))
    existing_count = c.fetchone()[0]
    conn.close()
    limit = limit or divided_max_results
    limit = int(limit)
    target_count = limit + existing_count
    fetch_limit = target_count * 2
    print(f"[EuropePMC] {existing_count} articles déjà présents en base pour '{field}'. Nouvelle limite de récupération : {fetch_limit}.")
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    page_size = min(1000, fetch_limit)
    articles_yielded = 0
    page = 1
    while articles_yielded < limit:
        params = {
            "query": f"{field} AND PUB_YEAR:[{year_from} TO {year_to}]",
            "format": "json",
            "pageSize": page_size,
            "page": page
        }
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                time.sleep(0.05)
                break
            except requests.exceptions.HTTPError as e:
                if response.status_code == 503 and attempt < retries - 1:
                    print(f"[EuropePMC] Service indisponible, retry dans {delay} sec...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"[EuropePMC] Erreur HTTP: {e}")
                    return
            except Exception as e:
                print(f"[EuropePMC] Erreur: {e}")
                return
        data = response.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break
        for result in results:
            fulltext_urls = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
            pdf_link = None
            for f in fulltext_urls:
                if f.get("documentStyle") == "pdf":
                    pdf_link = f.get("url")
                    break
            article = {
                "title": result.get("title"),
                "authors": result.get("authorString").split(", ") if result.get("authorString") else [],
                "field": field,
                "doi": result.get("doi"),
                "year": str(result.get("pubYear")) if result.get("pubYear") else None,
                "journal": result.get("journalTitle"),
                "volume": result.get("journalVolume"),
                "issue": result.get("issue"),
                "pages": result.get("pageInfo"),
                "source": "EuropePMC",
                "pdf_link": pdf_link
            }
            # Vérifier si déjà en base (par titre)
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM metadatas WHERE field = ? AND source = 'EuropePMC' AND title = ?", (field, article["title"]))
            if c.fetchone()[0] == 0:
                yield article
                articles_yielded += 1
                if articles_yielded >= limit:
                    conn.close()
                    break
            conn.close()
        if articles_yielded >= limit or len(results) < page_size:
            break
        page += 1

# --- TRAITEMENT ---

## Protection contre les duplicats

def generate_unique_id(article):
    title = article.get("title", "").strip().lower()
    uid = hashlib.md5(title.encode("utf-8")).hexdigest()
    return uid

## Récupération des métadonnées

def get_doi_from_title(title):
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": 1}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        time.sleep(0.05)
        items = r.json().get("message", {}).get("items", [])
        if items:
            doi = items[0].get("DOI")
            return doi
    except Exception as e:
        print(f"[Crossref] Erreur pour '{title[:50]}...': {e}")
    return None

def get_crossref_metadata(doi):
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        time.sleep(0.05)
        data = r.json().get("message", {})

        journal = data.get("container-title", [None])[0]
        volume = data.get("volume")
        issue = data.get("issue")
        pages = data.get("page")
        return {
            "journal": journal,
            "volume": volume,
            "issue": issue,
            "pages": pages
        }
    except Exception as e:
        return {}

## Recherche des fichiers .pdf manquants

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

# --- INSERTION EN BASE ---

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
        """, (article['field'], None, article['title'], str(article['authors']), article['doi'],
              article['journal'], article['year'], article['volume'], article['issue'], article['pages'],
              article['source']))
        # Insert into paper_data as well
        c.execute("""
            INSERT OR IGNORE INTO paper_data
            (doi, local_key, pdf_link, file_name, abstract)
            VALUES (?, ?, ?, ?, ?)
        """, (article['doi'], 0, article['pdf_link'], None, None))
    conn.commit()

    # Après insertion, pour ArXiv, mettre local_key=1 si pdf_link non NULL
    c.execute("""
        UPDATE paper_data
        SET local_key = 1
        WHERE doi IN (
            SELECT doi FROM metadatas WHERE source = 'ArXiv'
        ) AND pdf_link IS NOT NULL AND local_key = 0
    """)
    conn.commit()

    # Récupération DOI non encore enrichis
    c.execute("""
        SELECT m.doi
        FROM metadatas m
        JOIN paper_data p ON m.doi = p.doi
        WHERE m.doi IS NOT NULL AND p.local_key = 0
    """)
    new_dois = [row[0] for row in c.fetchall()]

    # Appel Crossref et Unpaywall en parallèle, mais Unpaywall seulement si source != "ArXiv"
    def enrich_article(article):
        crossref_data = get_crossref_metadata(article.get("doi"))
        unpaywall_data = None if article.get("source") == "ArXiv" else get_unpaywall_pdf(article.get("doi"), email)
        return article.get("doi"), crossref_data, unpaywall_data, article.get("source")

    # On doit récupérer les sources pour chaque DOI pour savoir si c'est ArXiv
    # On va récupérer les articles concernés depuis la base
    c.execute("""
        SELECT m.doi, m.source
        FROM metadatas m
        JOIN paper_data p ON m.doi = p.doi
        WHERE m.doi IS NOT NULL AND p.local_key = 0
    """)
    doi_source_pairs = c.fetchall()
    articles_for_enrich = [{"doi": row[0], "source": row[1]} for row in doi_source_pairs]

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(enrich_article, article) for article in articles_for_enrich]
        for future in as_completed(futures):
            results.append(future.result())

    # Mise à jour SQLite dans le thread principal
    for doi, crossref_data, unpaywall_data, article_source in results:
        pdf_url = unpaywall_data.get("pdf_link") if unpaywall_data and article_source != "ArXiv" else None
        c.execute("""
        UPDATE metadatas
        SET journal=?, volume=?, issue=?, pages=?
        WHERE doi=?
    """, (
        crossref_data.get("journal") if crossref_data else None,
        crossref_data.get("volume") if crossref_data else None,
        crossref_data.get("issue") if crossref_data else None,
        crossref_data.get("pages") if crossref_data else None,
        doi
    ))

    c.execute("""
        UPDATE paper_data
        SET pdf_link=?, local_key=1
        WHERE doi=?
    """, (
        pdf_url,
        doi
    ))
            
    conn.commit()
    conn.close()

# Nouvelle section __main__ propre, sans duplication ni appels multiples
if __name__ == "__main__":
    sources = {
        "PubMed": lambda: list(fetch_pubmed(field, year_from, year_to, limit=divided_max_results)),
        "EuropePMC": lambda: list(fetch_EuropePMC(field, year_from, year_to, limit=divided_max_results)),
        "ArXiv": lambda: list(fetch_arxiv(field, year_from, year_to, limit=divided_max_results))
    }

    results = {}
    print("Démarrage des fetchs en parallèle...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source = {executor.submit(func): name for name, func in sources.items()}
        for future in as_completed(future_to_source):
            name = future_to_source[future]
            try:
                results[name] = future.result()
                print(f"[{name}] Récupération terminée. {len(results[name])} articles récupérés.")
            except Exception as e:
                print(f"[{name}] Erreur durant la récupération : {e}")

    # Insertion SQLite séquentielle
    for source_name, articles in results.items():
        print(f"Insertion articles {source_name} dans la base SQLite...")
        insert_articles_into_sqlite(articles, email=email)

    print("Tous les articles ont été insérés dans la base SQLite 'bibliography.db'.")