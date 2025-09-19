from docling.document_converter import DocumentConverter
from pathlib import Path
import json
import sqlite3

def parse_to_json(file_path: str, export_dir: str = "datas/parses", db_path: str = "datas/bibliography.db") -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    file_name = path.stem

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # S'assurer que la table metadatas existe avec les bons champs
    cursor.execute("""
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
        )
    """)
    conn.commit()

    # Vérifier si une entrée existe déjà pour ce file_name
    cursor.execute("SELECT local_key FROM metadatas WHERE file_name = ?", (file_name,))
    row = cursor.fetchone()
    if row is not None:
        local_key = row[0]
        if local_key >= 3:
            print(f"Skipping {file_name}.pdf: already parsed (local_key={local_key})")
            conn.close()
            return None
        # sinon, on va parser et incrémenter local_key après
    else:
        # Insérer une nouvelle entrée avec file_name et local_key=2 (avant parsing)
        cursor.execute(
            "INSERT INTO metadatas (file_name, local_key) VALUES (?, ?)",
            (file_name, 2)
        )
        conn.commit()
        local_key = 2

    # Conversion avec Docling
    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document
    doc_dict = doc.export_to_dict()

    # Préparer un champ pour les métadonnées (à remplir plus tard)
    doc_dict["metadata"] = {}  # Placeholder for future metadata

    # Sauvegarde JSON
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    out_file = export_path / f"{path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)

    # Après parsing, incrémenter local_key de 1 dans metadatas
    cursor.execute("UPDATE metadatas SET local_key = ? WHERE file_name = ?", (local_key + 1, file_name))
    conn.commit()
    conn.close()

    return doc_dict

def parse_batch_to_json(pdf_paths, export_dir, db_path="datas/bibliography.db"):
    """
    Parse plusieurs PDFs en une seule passe de pipeline et exporte en JSON.
    Met aussi à jour la base (metadatas).
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Connexion DB
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS metadatas (
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
    );""")
    conn.commit()

    # Initialiser une fois le pipeline
    converter = DocumentConverter()
    parsed_count = 0
    for pdf_path in pdf_paths:
        file_stem = Path(pdf_path).stem
        json_path = export_dir / f"{file_stem}.json"

        # Vérifier l'état local_key en base AVANT parsing
        c.execute("SELECT local_key FROM metadatas WHERE file_name=?", (file_stem,))
        row = c.fetchone()
        if row is not None:
            local_key = row[0]
            if local_key is not None and local_key >= 3:
                print(f"Skipping {file_stem}.pdf: already parsed (local_key={local_key})")
                continue
            # sinon, on va parser et incrémenter local_key après
        else:
            # Insérer une nouvelle entrée avec file_name et local_key=2 (avant parsing)
            c.execute("INSERT INTO metadatas (file_name, local_key) VALUES (?, ?)", (file_stem, 2))
            conn.commit()
            local_key = 2

        # Conversion PDF un par un
        result = converter.convert(Path(pdf_path))
        doc = result.document if hasattr(result, "document") else result
        doc_dict = doc.export_to_dict() if hasattr(doc, "export_to_dict") else doc.to_dict()

        # Sauvegarde JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

        # Après parsing, incrémenter local_key de 1 dans metadatas
        c.execute("UPDATE metadatas SET local_key=? WHERE file_name=?", (local_key + 1, file_stem))
        parsed_count += 1

    conn.commit()
    conn.close()
    return parsed_count

# Exemple d'utilisation
if __name__ == "__main__":
    pdf_dir = Path("datas/pdfs")
    export_dir = "datas/parses"
    db_path = "datas/bibliography.db"
    for pdf_file in pdf_dir.glob("*.pdf"):
        parse_to_json(str(pdf_file), export_dir, db_path)