import json
import re
import csv

json_path = "lab/fumehood/docling/export/example.json"
csv_path = "references_separated.csv"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Trouver le header References
refs_start_idx = None
for i, item in enumerate(data["texts"]):
    if item.get("label") == "section_header" and "reference" in item.get("text", "").lower():
        refs_start_idx = i
        break

if refs_start_idx is None:
    raise ValueError("Header 'References' introuvable.")

# 2. Concaténer tous les textes après le header
all_refs_text = ""
for item in data["texts"][refs_start_idx+1:]:
    if item.get("label") == "section_header":
        break
    text = item.get("text", "").strip()
    if text:
        all_refs_text += " " + text

# 3. Séparer sur le pattern [n]
split_pattern = re.compile(r'(\[\d+\])')
parts = split_pattern.split(all_refs_text)
# Cette regex garde le numéro [n] comme élément séparé

refs = []
for i in range(1, len(parts), 2):
    ref_id = parts[i][1:-1]  # enlève les crochets
    ref_text = parts[i] + parts[i+1]  # numéro + texte
    refs.append({"id": ref_id, "reference": ref_text.strip()})

# 4. Export CSV
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["id", "reference"])
    writer.writeheader()
    for r in refs:
        writer.writerow(r)

print(f"{len(refs)} références exportées dans {csv_path}")