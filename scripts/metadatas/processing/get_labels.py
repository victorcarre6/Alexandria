import json
import re

json_path = "lab/fumehood/docling/export/example.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# IDs à vérifier : tous les texts
for idx, text_item in enumerate(data["texts"]):
    text_snippet = text_item['text'][:60]
    # détection plus large
    has_citation = bool(re.search(r'^\[\d+\]|arxiv|DOI|conference|preprint', text_item['text'], re.IGNORECASE))
    if has_citation:
        print(f"ID {idx}: label={text_item['label']}, has_citation={has_citation}, text={text_snippet}...")