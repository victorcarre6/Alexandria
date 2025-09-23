# Context – Neo4j / SQLite Interaction for the Bibliographic Project

## Objective

Create a **citation graph** in Neo4j while keeping SQLite as the primary documentary source, allowing for both **relational analysis** and **enriched consultation**.

---

## Key Principles

1. **Neo4j**: deep article graph
   - Nodes `Paper` with DOI as the unique key.
   - `CITES` relationships connecting citing articles to cited articles.
   - Optimized for exploring multiple citation layers (`depth > 1`) and calculating centrality.
   - Stores only lightweight metadata (title, year, optionally authors for quick display).

2. **SQLite**: rich documentary database
   - Contains abstracts, sections, PDFs, parsed JSON, embeddings.
   - Serves as the detailed info source for a given DOI.
   - Maintains full metadata and extracted references.

---

## Interaction Flow

1. **Insertion into Neo4j**
   - Insert `Paper` nodes with DOI + minimal metadata.
   - Insert `CITES` relationships using DOIs extracted (via CrossRef or SQLite).

2. **Hover / Detailed Consultation**
   - When a user hovers over a node in Neo4j:
     - Neo4j returns the DOI.
     - SQLite is queried to retrieve detailed information (abstract, sections, PDF path).
   - Keeps Neo4j lightweight while providing rich information on demand.

---

## Advantages

- **Neo4j**: efficient for multi-layer exploration, centrality calculations, and visualization of citation hubs.
- **SQLite**: content-rich, stores documents and parsing results, serves as the source of truth.
- **DOI Synchronization**: acts as the pivot key linking the two databases, enabling seamless navigation between the graph and detailed info.

---

## Practical Notes

- SQLite ideally contains: `papers(doi, title, authors, year, abstract, sections, pdf_path)` and `citations(citing_doi, cited_doi)`.
- Neo4j contains minimal nodes and relationships but can enrich hover information by fetching from SQLite.
- The deep graph is built by iterating over DOIs, retrieving references via CrossRef up to a depth `n`.
- This architecture allows for a performant relational graph while providing interactive access to detailed information.