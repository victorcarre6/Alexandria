import os
import requests
import sqlite3
from requests.exceptions import HTTPError

PDF_DIR = "datas/pdfs"

def make_filename(title: str, journal: str, year: str) -> str:
    # Use "unknown_journal" if journal is missing or empty
    use_journal = journal if journal and journal.strip() != "" else ""
    safe_title = "".join(c for c in title[:50] if c.isalnum() or c in (" ", "_", "-")).rstrip()
    filename = f"{safe_title}({use_journal}{year}).pdf"
    return filename

def download(record: dict) -> dict:
    os.makedirs(PDF_DIR, exist_ok=True)
    filename = make_filename(record.get("title", ""), record.get("journal", ""), record.get("year", ""))
    local_path = os.path.join(PDF_DIR, filename)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/116.0.0.0 Safari/537.36"
    }

    pdf_link = record.get("pdf_link")
    if pdf_link:
        try:
            resp = requests.get(pdf_link, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            print(f"PDF downloaded from official link for '{record.get('title')}'")
            record["local_path"] = local_path
            record["file_name"] = os.path.splitext(filename)[0]
            return record
        except Exception as e:
            print(f"Warning: Failed to download PDF for '{record.get('title')}' from {pdf_link}: {e}")
    return record

def sync_pdf_cache_scihub(doi: str) -> str:
    """
    Attempt to download a PDF from Sci-Hub given a DOI.
    Returns the local path to the downloaded PDF if successful, else None.
    """
    if not doi:
        return None
    os.makedirs(PDF_DIR, exist_ok=True)
    # Use DOI as part of filename, sanitize it
    safe_doi = "".join(c for c in doi if c.isalnum() or c in ("_", "-", ".")).rstrip()
    filename = f"scihub_{safe_doi}.pdf"
    local_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    # Sci-Hub URL (using a common mirror)
    scihub_url = f"https://www.sci-hub.ru/{doi}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(scihub_url, headers=headers, timeout=30)
        resp.raise_for_status()
        # Sci-Hub returns an HTML page with an iframe containing the PDF link
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.content, "html.parser")
        iframe = soup.find("iframe")
        if iframe and iframe.has_attr("src"):
            pdf_url = iframe["src"]
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif pdf_url.startswith("/"):
                pdf_url = "https://sci-hub.se" + pdf_url
            pdf_resp = requests.get(pdf_url, headers=headers, timeout=30)
            pdf_resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(pdf_resp.content)
            return local_path
    except Exception as e:
        print(f"Warning: Failed to download PDF from Sci-Hub for DOI '{doi}': {e}")
    return None

def find_oa_pdf(record: dict) -> str:
    """
    Attempt to locate a PDF from open access sources using DOI or title.
    Currently supports arXiv and ChemRxiv.
    Returns a direct PDF URL if found, otherwise None.
    """
    doi = (record.get("doi") or "").strip()
    # arXiv
    if doi.startswith("10.48550/arXiv."):
        arxiv_id = doi[len("10.48550/arXiv.") :]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # ChemRxiv
    if doi.startswith("10.26434/chemrxiv."):
        doi_suffix = doi[len("10.26434/chemrxiv.") :]
        return f"https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/{doi_suffix}/bin"
    # Add more OA sources here as needed
    return None

def get_pdf_pipeline(record: dict) -> dict:
    if record.get("local_key", 0) != 1:
        return record
    os.makedirs(PDF_DIR, exist_ok=True)
    filename = make_filename(record.get("title", ""), record.get("journal", ""), record.get("year", ""))
    local_path = os.path.join(PDF_DIR, filename)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/116.0.0.0 Safari/537.36"
    }
    # 1. Try official pdf_link
    pdf_link = record.get("pdf_link")
    if pdf_link:
        try:
            resp = requests.get(pdf_link, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            print(f"PDF downloaded from official link for '{record.get('title')}'")
            record["local_path"] = local_path
            record["local_key"] = 2
            record["file_name"] = os.path.splitext(filename)[0]
            return record
        except Exception as e:
            print(f"Warning: Failed to download PDF for '{record.get('title')}' from {pdf_link}: {e}")
    # 2. Try open access sources
    oa_pdf_url = find_oa_pdf(record)
    if oa_pdf_url:
        try:
            resp = requests.get(oa_pdf_url, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            print(f"PDF downloaded from Open Access source for '{record.get('title')}'")
            record["local_path"] = local_path
            record["local_key"] = 2
            record["file_name"] = os.path.splitext(filename)[0]
            return record
        except Exception as e:
            print(f"Warning: Failed to download OA PDF for '{record.get('title')}' from {oa_pdf_url}: {e}")
    # 3. Try Sci-Hub fallback
    local_path_scihub = sync_pdf_cache_scihub(record.get("doi"))
    if local_path_scihub:
        print(f"PDF downloaded from Sci-Hub for '{record.get('title')}'")
        record["local_path"] = local_path_scihub
        record["local_key"] = 2
        record["file_name"] = os.path.splitext(os.path.basename(local_path_scihub))[0]
        return record
    return record

# --- Traitement de la base pour téléchargement des PDFs ---
def process_pdfs(db_path="datas/bibliography.db"):
    """
    Parcourt la base de données pour télécharger les PDFs pour toutes les entrées ayant un pdf_link ou un doi.
    Met à jour local_key et pdf_name si le PDF est téléchargé avec succès.
    """
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
    # Sélectionner tous les articles avec un pdf_link non NULL ou un doi non NULL, inclure local_key
    c.execute("SELECT id, title, journal, year, pdf_link, doi, local_key FROM metadatas WHERE pdf_link IS NOT NULL OR doi IS NOT NULL")
    rows = c.fetchall()
    pdfs_cached = 0
    for row in rows:
        row_id, title, journal, year, pdf_link, doi, local_key = row
        record = {
            "title": title,
            "journal": journal,
            "year": year,
            "pdf_link": pdf_link,
            "doi": doi,
            "local_key": local_key
        }
        result = get_pdf_pipeline(record)
        local_path = result.get("local_path")
        local_key = result.get("local_key")
        file_name = result.get("file_name")
        if local_path:
            c.execute("UPDATE metadatas SET local_key=?, file_name=? WHERE id=?", (local_key, file_name, row_id))
            pdfs_cached += 1
    conn.commit()
    conn.close()
    return pdfs_cached