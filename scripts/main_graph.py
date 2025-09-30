import networkx as nx
from neo4j import GraphDatabase
import networkx as nx

import sys
from pathlib import Path
import json

from scripts.connections.sim_graph import generate_graph

# === Connexions ===

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

keys_path = ROOT_DIR / "keys.json"
with open(keys_path) as f:
    keys = json.load(f)

NEO4J_URI = keys["NEO4J_URI"]
NEO4J_USERNAME = keys["NEO4J_USERNAME"]
NEO4J_PASSWORD = keys["NEO4J_PASSWORD"]
email = keys["EMAIL"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
G: nx.Graph = nx.Graph()

DB_PATH = "datas/bibliography.db"

# === Paramètres ===

PATIENT_ZERO = "10.1002/cssc.201900519"

DEPTH = 4
MAX_DOIS = 500

PAUSE = 0.5
BATCH_SIZE = 100
MAX_WORKER = 10

EDGE_MODE = "SIMILAR"   # "CITES", "SIMILAR"
WEIGHT_THRESHOLD = 10
TOP_N = 1
TOP_N_CITES = 30
TOP_N_SCORE = 10
TOP_N_COCIT = 10
TOP_N_COUPLING = 10

total_analyzed = 0

    
generate_graph(driver, DB_PATH, PATIENT_ZERO, DEPTH, MAX_DOIS, 
                   PAUSE, BATCH_SIZE, MAX_WORKER, 
                   EDGE_MODE, WEIGHT_THRESHOLD, 
                   TOP_N, TOP_N_CITES, TOP_N_SCORE, TOP_N_COCIT, TOP_N_COUPLING)