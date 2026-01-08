"""
✅ NEO4J CHUNK1 - Focused GraphRAG & Transformation Tool (Fixed)

Features:
1. Salesforce to Neo4j Deep Transformation (Recursive & Deduplicated)
2. Schema-based Ingestion
3. Vector Search (Semantic)
4. GraphRAG (Hybrid Semantic + Neo4j Graph)
5. Multi-hop Reasoning

Usage:
python tools/neo4j_chunk1.py
"""

import os
import json
import uuid
from pathlib import Path
from rich import print
from rich.table import Table
from rich.prompt import Prompt
import numpy as np
import google.generativeai as genai
from neo4j import GraphDatabase
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file (looking in script directory)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)


# Config
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

NEO4J_URI = (os.getenv("NEO4J_URI") or os.getenv("NEO_URL") or "neo4j://localhost:7687").strip()
NEO4J_USER = (os.getenv("NEO4J_USERNAME") or os.getenv("NEO_USER") or "neo4j").strip()
NEO4J_PASS = (os.getenv("NEO4J_PASSWORD") or os.getenv("NEO_PASS") or "password").strip()

if not NEO4J_URI:
    NEO4J_URI = "neo4j://localhost:7687"

neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
DATA_DIR = Path("./neo4j_data")
DATA_DIR.mkdir(exist_ok=True)

# State
file_registry = {}     # file_id → metadata
embedding_index = {}   # file_id → numpy vector
active_schema = None   # Loaded transformation schema

# ===================================================================
# HELPERS
# ===================================================================
def get_embedding(text):
    out = genai.embed_content(model=EMBED_MODEL, content=text)
    return np.array(out["embedding"], dtype=np.float32)

def find_sf_type(record_id):
    """Simple prefix match for core Salesforce objects."""
    if not record_id: return "Object"
    if record_id.startswith("001"): return "Account"
    if record_id.startswith("003"): return "Contact"
    if record_id.startswith("006"): return "Opportunity"
    if record_id.startswith("00Q"): return "Lead"
    if record_id.startswith("500"): return "Case"
    return "Object"

def get_label(rec, default="Knowledge"):
    """Unified label discovery from attributes or ID."""
    # 1. From Attributes
    attr_type = rec.get("attributes", {}).get("type")
    if attr_type: return attr_type
    
    # 2. From ID Prefix
    rid = rec.get("Id", "")
    return find_sf_type(rid) or default

# ===================================================================
# TRANSFORMATION & INGESTION
# ===================================================================
def ingest_record_recursively(record):
    """
    Recursively extracts nodes and relationships from a nested Salesforce JSON.
    Returns deduplicated results.
    """
    results = {
        "nodes": {},           # sfid -> (Label, Props)
        "relationships": set()  # set of (FL, FID, Type, TL, TID)
    }

    def process(rec):
        rec_id = rec.get("Id")
        if not rec_id: return
        
        label = get_label(rec)
        
        # Extract Properties (scalar only)
        props = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
        props["sfid"] = rec_id
        
        # Store node (overwrite if duplicate to ensure we have record data)
        results["nodes"][rec_id] = (label, props)

        # Look for nested objects / relationships
        for key, val in rec.items():
            # Handle Subqueries (e.g. { "Contacts": { "records": [...] } })
            if isinstance(val, dict) and "records" in val:
                predicate = key.upper()
                child_records = val.get("records", [])
                for child in child_records:
                    child_id = child.get("Id")
                    if child_id:
                        child_label = get_label(child)
                        results["relationships"].add((label, rec_id, predicate, child_label, child_id))
                        process(child)
            
            # Handle Lookup IDs (e.g. AccountId: "...")
            elif key.endswith("Id") and val and isinstance(val, str) and key != "Id":
                target_label = find_sf_type(val)
                predicate = key.replace("Id", "").upper()
                results["relationships"].add((label, rec_id, predicate, target_label, val))

    process(record)
    return results

def push_to_neo4j(data_package):
    """Bulk load nodes and relationships into Neo4j."""
    nodes_count = 0
    rels_count = 0
    with neo4j_driver.session() as session:
        # Load Nodes
        for sfid, (label, props) in data_package["nodes"].items():
            session.run(f"MERGE (n:`{label}` {{sfid: $id}}) SET n += $props", id=sfid, props=props)
            nodes_count += 1
        
        # Load Relationships
        for f_label, f_id, r_type, t_label, t_id in data_package["relationships"]:
            session.run(f"""
                MERGE (a:`{f_label}` {{sfid: $fid}})
                MERGE (b:`{t_label}` {{sfid: $tid}})
                MERGE (a)-[:`{r_type}`]->(b)
            """, fid=f_id, tid=t_id)
            rels_count += 1
    return nodes_count, rels_count

def local_index_for_rag(node_dict):
    """Create local text signals and embeddings for the vector side of GraphRAG."""
    for sfid, (label, props) in node_dict.items():
        # Create a semantic summary
        summary = [f"Type: {label}", f"ID: {sfid}"]
        for k, v in props.items():
            if k not in ["sfid", "attributes"]:
                summary.append(f"{k}: {v}")
        
        text = "\n".join(summary)
        fname = f"neo_{label}_{sfid}.txt"
        fpath = DATA_DIR / fname
        fpath.write_text(text)
        
        emb = get_embedding(text)
        file_registry[fname] = {
            "record_id": sfid,
            "object_type": label,
            "text": text,
            "path": str(fpath)
        }
        embedding_index[fname] = emb

# ===================================================================
# GRAPH RAG & SEARCH
# ===================================================================
def semantic_search(query, k=5):
    if not embedding_index:
        return []
    qemb = get_embedding(query)
    results = []
    for fid, emb in embedding_index.items():
        sim = float(np.dot(qemb, emb) / (np.linalg.norm(qemb) * np.linalg.norm(emb)))
        results.append((fid, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

def run_cypher(cypher):
    try:
        # Clean markdown
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        with neo4j_driver.session() as session:
            return [record.data() for record in session.run(cypher)]
    except Exception as e:
        return [{"error": str(e)}]

def graph_rag_answer(query):
    # 1. Semantic retrieval (Vector)
    sem_hits = semantic_search(query, k=3)
    sem_context = "\n\n".join([file_registry[fid]["text"] for fid, _ in sem_hits])
    
    # 2. Graph retrieval (Cypher)
    prompt = f"Translate to Neo4j Cypher. Return ONLY the code.\nQuestion: {query}"
    model = genai.GenerativeModel(LLM_MODEL)
    cy_resp = model.generate_content(prompt).text.strip()
    graph_results = run_cypher(cy_resp)
    
    # 3. Final synthesis
    context = f"""
    ### VECTOR RESULTS
    {sem_context}
    
    ### GRAPH RESULTS (Cypher: {cy_resp})
    {json.dumps(graph_results, indent=2)}
    """
    
    final_prompt = f"Answer the user question using the context provided. If no direct relationship is found, use the graph results to explain indirect links.\nContext: {context}\nQuestion: {query}"
    answer = model.generate_content(final_prompt).text
    return answer, cy_resp, graph_results

def multi_hop_reasoning(query):
    """
    Advanced GraphRAG specifically for discovering long-distance relationships.
    """
    prompt = f"""
    You are a Graph Expert. Translate the question into a Cypher query that finds PATHS (multi-hop).
    The graph contains: Account, Contact, Opportunity, Lead, Case.
    
    RULES:
    1. Use variable length paths if needed: (n)-[*1..3]-(m)
    2. Focus on connecting the entities mentioned in the question.
    3. Return the nodes and the relationships between them.
    4. Return ONLY the Cypher code.
    
    Question: {query}
    """
    model = genai.GenerativeModel(LLM_MODEL)
    cypher = model.generate_content(prompt).text.strip()
    cypher = cypher.replace("```cypher", "").replace("```", "").strip()
    
    print(f"[magenta]Generated Multi-hop Cypher:[/magenta]\n{cypher}")
    
    results = run_cypher(cypher)
    
    # Synthesis
    context = f"Question: {query}\n\nGraph Findings (Multi-hop):\n{json.dumps(results, indent=2)}"
    answer_prompt = f"Explain the multi-hop relationship found in this graph data clearly.\n{context}"
    answer = model.generate_content(answer_prompt).text
    return answer, cypher, results

# ===================================================================
# SCHEMA MANAGEMENT
# ===================================================================
def upload_schema():
    global active_schema
    path = Prompt.ask("Enter path to JSON schema", default="tools/neo4j_schema_example.json")
    if not os.path.exists(path):
        print(f"[red]File not found: {path}[/red]")
        return
    try:
        with open(path, 'r') as f:
            active_schema = json.load(f)
        print("[green]Schema loaded successfully![/green]")
    except Exception as e:
        print(f"[red]Failed to load schema: {e}[/red]")

# ===================================================================
# MAIN LOOP
# ===================================================================
def main():
    while True:
        print("\n[bold magenta]== NEO4J CHUNK1: GRAPH RAG CONSOLE ==[/bold magenta]")
        print("1) Load Salesforce JSON (Recursive Transform)")
        print("2) Upload/Set Neo4j Transformation Schema")
        print("3) Run semantic vector search (Local)")
        print("4) Ask a Hybrid GraphRAG question")
        print("5) List locally indexed nodes")
        print("6) Perform Advanced Multi-hop Search")
        print("7) Exit")
        
        ch = Prompt.ask("Choice", choices=["1", "2", "3", "4", "5", "6", "7"])
        
        if ch == "1":
            path = Prompt.ask("Path to Salesforce JSON", default="tools/testdata.json")
            if not os.path.exists(path):
                print("[red]File not found[/red]")
                continue
            with open(path, 'r') as f:
                data = json.load(f)
            
            records = data if isinstance(data, list) else [data]
            all_pkg = {"nodes": {}, "relationships": set()}
            
            print(f"[cyan]Processing {len(records)} top-level records...[/cyan]")
            for r in records:
                pkg = ingest_record_recursively(r)
                all_pkg["nodes"].update(pkg["nodes"])
                all_pkg["relationships"].update(pkg["relationships"])
            
            print(f"[cyan]Pushing {len(all_pkg['nodes'])} unique nodes and {len(all_pkg['relationships'])} relationships to Neo4j...[/cyan]")
            n_in, r_in = push_to_neo4j(all_pkg)
            
            print(f"[cyan]Building local vector index...[/cyan]")
            local_index_for_rag(all_pkg["nodes"])
            
            print(f"[green]Ingestion Complete! {n_in} nodes and {r_in} relationships added.[/green]")

        elif ch == "2":
            upload_schema()
            
        elif ch == "3":
            q = Prompt.ask("Enter query for vector search")
            hits = semantic_search(q)
            table = Table(title="Top Semantic Matches")
            table.add_column("File")
            table.add_column("Similarity")
            for fid, sim in hits:
                table.add_row(fid, f"{sim:.4f}")
            print(table)

        elif ch == "4":
            q = Prompt.ask("Enter your GraphRAG question")
            ans, cy, gres = graph_rag_answer(q)
            print(f"\n[bold green]ANSWER:[/bold green]\n{ans}")
            print(f"\n[cyan]Cypher Path:[/cyan] {cy}")
            print(f"[cyan]Graph Data:[/cyan] {len(gres)} records found.")

        elif ch == "5":
            table = Table(title="Local Document Registry")
            table.add_column("Filename")
            table.add_column("Type")
            for fid, meta in file_registry.items():
                table.add_row(fid, meta["object_type"])
            print(table)

        elif ch == "6":
            q = Prompt.ask("Enter complex multi-hop question (e.g. 'How is Contact X related to Opportunity Y?')")
            ans, cy, gres = multi_hop_reasoning(q)
            print(f"\n[bold magenta]MULTI-HOP DISCOVERY:[/bold magenta]\n{ans}")
            print(f"\n[cyan]Reasoning Path:[/cyan] {cy}")

        elif ch == "7":
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
