# Neo4j Chunk1: Advanced GraphRAG & Transformation Guide

`neo4j_chunk1.py` is a specialized tool that bridge the gap between structured Salesforce data, Neo4j Graph databases, and LLM-powered reasoning. It provides a robust pipeline for converting nested JSON responses into a high-fidelity Knowledge Graph while simultaneously enabling semantic search.

---

## 🚀 Core Features

*   **Recursive Deep Ingestion**: Automatically flattens complex nested JSON (like `testdata.json`) and extracts nodes and relationships in a single pass.
*   **Unified Identity Management**: Uses Salesforce `sfid` as a global unique key to prevent node duplication and ensure relationship integrity.
*   **Vector + Graph Hybrid RAG**: Uses Gemini embeddings for semantic matching and Neo4j for factual relationship traversal.
*   **Multi-hop Reasoning Engine**: Capable of discovering indirect connections (e.g., "Finding contacts of accounts with specific opportunity statuses") that standard vector search would miss.
*   **Schema-Driven Mapping**: Support for custom JSON schemas to control how fields are mapped and relationships are labeled.

---

## 🛠️ Quick Start Example

### 1. Ingesting Data
Run the tool and select **Option 1**. Enter the path to `tools/testdata.json`.
```powershell
python tools/neo4j_chunk1.py
# Select Choice: 1
# Enter Path: tools/testdata.json
```
**Result**: The tool will identify the 5 Accounts in `testdata.json`, extract the nested `Contacts` and `Opportunities`, and link them in Neo4j.

### 2. Verify Graph Ingestion
In your Neo4j Browser, run:
```cypher
MATCH (a:Account)-[r]->(b) RETURN a, r, b LIMIT 25
```
You will see Accounts linked to their related records via labels like `CONTACTS` and `OPPORTUNITIES`.

---

## 🔍 Search & Reasoning Examples

### Option 4: Hybrid GraphRAG (Standard)
Best for pinpointing facts within a specific record or direct relationship.
*   **Question**: *"What is the revenue of Edge Communications and who is their primary contact?"*
*   **How it works**: It retrieves the `Account` node for Edge Communications, looks up the related `Contact` nodes, and synthesizes the answer.

### Option 6: Advanced Multi-hop Search
Best for complex business logic requiring traversal across multiple nodes.
*   **Example 1**: *"Find the names of all contacts associated with accounts that have at least one Closed Won opportunity worth more than $50,000."*
    *   **Reasoning Path**: `(Opportunity {StageName: 'Closed Won', Amount > 50000}) <-[:OPPORTUNITIES]- (Account) -[:CONTACTS]-> (Contact)`
*   **Example 2**: *"Are there any construction companies we are currently prospecting, and who should we call there?"*
    *   **Reasoning Path**: `(Account {Industry: 'Construction'}) -[:OPPORTUNITIES]-> (Opportunity {StageName: 'Prospecting'})` then `(Account) -[:CONTACTS]-> (Contact)`

---

## 📋 Schema Mapping Guide

You can define how Salesforce data is transformed using a JSON schema (**Option 2**).

**Example Schema (`tools/neo4j_schema_example.json`):**
```json
{
  "mappings": {
    "Account": {
      "label": "Account",
      "properties": ["Name", "Industry", "AnnualRevenue"],
      "relationships": [
        {"field": "Contacts", "type": "HAS_CONTACT", "target": "Contact"},
        {"field": "Opportunities", "type": "HAS_OPPORTUNITY", "target": "Opportunity"}
      ]
    }
  }
}
```
*   **field**: The key in the Salesforce JSON.
*   **type**: The resulting Neo4j relationship label.
*   **target**: The label of the node being linked to.

---

## 📂 Internal Architecture

### 1. The Transformation Engine (`get_label`)
The tool uses a unified label discovery logic:
1.  **Priority 1**: Checks Salesforce `attributes.type`.
2.  **Priority 2**: Inspects the **ID Prefix** (e.g., `001` = Account, `006` = Opportunity).
3.  **Fallback**: Defaults to "Knowledge" or "Object".

### 2. The Semantic Index (`neo4j_data/`)
For every node created in Neo4j, a corresponding `.txt` file is generated in the `neo4j_data/` folder. This file contains a summarized view of the record's properties, which is then embedded to enable the "Vector Search" side of the RAG pipeline.

### 3. Identity Invariant `MERGE`
The tool safely uses the following Cypher logic during ingestion:
```cypher
MERGE (n:Account {sfid: '001...' }) SET n += { properties }
```
This ensures that running the same ingestion twice **updates** your data rather than creating duplicates.

---

## 💡 Pro Tips
*   **Complex Subqueries**: The tool is designed to work with SOQL results that use `(SELECT ... FROM ...)`. Ensure your SOQL includes the `Id` field for every object to enable relationship building.
*   **Performance**: For very large datasets, the tool deduplicates nodes in-memory before pushing to Neo4j to minimize API overhead.
