import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1].parent)) 

from metadatas import fetch_subject
from metadatas.fetch_subject import insert_articles_into_sqlite
from metadatas import downloader
from metadatas import docling

# -----------------------
# Variables principales
# -----------------------
max_results = 20
subject = "4CzIPN"
year_from = 2015
year_to = 2026

email = "victorcarre@icloud.com"
divided_max_results = round(max_results/3)

# -----------------------
# Étape 1 : Fetch sources
# -----------------------
print(f"=== Étape 1 : Récupération de {max_results} publications sur le sujet '{subject}' ===")
articles = []
for source_func in [fetch_subject.fetch_pubmed, fetch_subject.fetch_EuropePMC, fetch_subject.fetch_arxiv]:
    fetched = list(source_func(subject, year_from, year_to, limit=divided_max_results))
    print(f" - {len(fetched)} articles récupérés depuis {source_func.__name__}")
    articles.extend(fetched)

print("\n=== Insertion en base ===")
insert_articles_into_sqlite(articles, email=email, db_path="datas/bibliography.db")
print(f"{len(articles)} articles insérés/enrichis dans la base.")

# -----------------------
# Étape 2 : Téléchargement PDFs
# -----------------------
print(f"\n=== Étape 2 : Téléchargement des PDFs ===")
# Utilisation directe de process_pdfs sur la base de données
downloaded_count = downloader.process_pdfs(db_path="datas/bibliography.db")
print(f"[Downloader] {downloaded_count} PDFs téléchargés.")

# -----------------------
# Étape 3 : Parsing PDFs avec Docling
# -----------------------
print(f"\n=== Étape 3 : Parsing PDFs avec Docling ===")
pdf_dir = Path("datas/pdfs")
export_dir = Path("datas/parses")
db_path = "datas/bibliography.db"

pdf_files = [str(f) for f in pdf_dir.glob("*.pdf")]
parsed_count = docling.parse_batch_to_json(pdf_files, export_dir=export_dir, db_path=db_path)

print(f"[Docling] {parsed_count}/{len(pdf_files)} PDFs parsés et exportés en JSON.")