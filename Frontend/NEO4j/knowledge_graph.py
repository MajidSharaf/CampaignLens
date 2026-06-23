import os
import torch
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Neo4j connection
# Reads from .env — no hardcoded credentials.
# ---------------------------------------------------------------------------

def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password")
        )
    )

# ---------------------------------------------------------------------------
# REBEL model
# Loaded once at module level — expensive to reload per call.
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[knowledge_graph] Loading REBEL model...")
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
model.to(device)
print(f"[knowledge_graph] REBEL loaded on {device}")

# ---------------------------------------------------------------------------
# REBEL parser
# Converts raw token string output into subject/relation/object triplets.
# Taken directly from the notebook — logic unchanged.
# ---------------------------------------------------------------------------

def parse_rebel_output(text):
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current_token = 'x'

    processed_text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")

    for token in processed_text.split():
        if token == "<triplet>":
            current_token = 't'
            if relation != '':
                triplets.append({
                    "subject": subject.strip(),
                    "relation": relation.strip(),
                    "object": object_.strip()
                })
                relation, object_ = '', ''
            subject = ''
        elif token == "<subj>":
            current_token = 's'
            if relation != '':
                triplets.append({
                    "subject": subject.strip(),
                    "relation": relation.strip(),
                    "object": object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current_token = 'o'
            relation = ''
        else:
            if current_token == 't':
                subject += ' ' + token
            elif current_token == 's':
                object_ += ' ' + token
            elif current_token == 'o':
                relation += ' ' + token

    if subject != '' and relation != '' and object_ != '':
        triplets.append({
            "subject": subject.strip(),
            "relation": relation.strip(),
            "object": object_.strip()
        })

    return triplets


# ---------------------------------------------------------------------------
# Triplet extraction
# Runs REBEL on a single comment and returns list of triplets.
# ---------------------------------------------------------------------------

def extract_triplets(text):
    model_inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    gen_outputs = model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_length=256,
        num_beams=3,
        length_penalty=1.0,
        early_stopping=True
    )

    raw_token_string = tokenizer.batch_decode(gen_outputs, skip_special_tokens=False)[0]
    return parse_rebel_output(raw_token_string)


# ---------------------------------------------------------------------------
# Neo4j write
# Stores triplet with comment_id so every relationship is traceable
# back to the source comment in analysis_results.csv.
# Fixed from notebook: comment_id added to RELATION properties.
# ---------------------------------------------------------------------------

def addTriplet(tx, subject, relation, object_, comment_id):
    tx.run("""
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object_})
        MERGE (s)-[:RELATION {type: $relation, comment_id: $comment_id}]->(o)
    """, subject=subject, relation=relation, object_=object_, comment_id=comment_id)


# ---------------------------------------------------------------------------
# Load pipeline
# Reads analysis_results.csv, runs REBEL on each comment,
# pushes all triplets to Neo4j.
# This is what your teammate calls in stage 3.
# ---------------------------------------------------------------------------

def load_graph(csv_path, limit=None):
    df = pd.read_csv(csv_path)

    if limit:
        df = df[:limit]

    driver = get_driver()
    driver.verify_connectivity()
    print(f"[knowledge_graph] Connected to Neo4j. Processing {len(df)} comments...")

    all_triplets = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        comment = row.get("cleaned_text", "")
        comment_id = row.get("comment_id", "unknown")

        if not isinstance(comment, str) or len(comment.strip()) < 5:
            continue

        triplets = extract_triplets(comment)

        for t in triplets:
            t["comment_id"] = comment_id

        all_triplets.extend(triplets)

    print(f"[knowledge_graph] Total triplets extracted: {len(all_triplets)}")

    with driver.session() as session:
        for triplet in tqdm(all_triplets, desc="Pushing to Neo4j"):
            session.execute_write(
                addTriplet,
                triplet["subject"],
                triplet["relation"],
                triplet["object"],
                triplet["comment_id"]
            )

    print("[knowledge_graph] All triplets pushed to Neo4j")
    driver.close()
    return all_triplets


# ---------------------------------------------------------------------------
# Neo4j query functions
# These are what neo4j_query.py (your layer) imports and calls.
# ---------------------------------------------------------------------------

def queryEntity(tx, entity):
    """
    Returns all relationships where the entity is the subject.
    e.g. queryEntity(tx, "trump") -> all triplets where trump is the subject.
    """
    result = tx.run("""
        MATCH (s:Entity {name: $entity})-[r]->(o)
        RETURN s.name AS subject, r.type AS relation, o.name AS object, r.comment_id AS comment_id
    """, entity=entity)
    return result.data()


def queryRelationship(tx, subject, object_):
    """
    Returns all relationships between two specific entities.
    e.g. queryRelationship(tx, "trump", "economy")
    """
    result = tx.run("""
        MATCH (s:Entity {name: $subject})-[r]->(o:Entity {name: $object_})
        RETURN s.name AS subject, r.type AS relation, o.name AS object, r.comment_id AS comment_id
    """, subject=subject, object_=object_)
    return result.data()


def queryNeighbours(tx, entity, depth=1):
    """
    Returns all entities connected to the given entity within a certain depth.
    Useful for graph exploration queries.
    """
    result = tx.run("""
        MATCH path = (s:Entity {name: $entity})-[*1..$depth]-(connected)
        RETURN DISTINCT connected.name AS neighbour
    """, entity=entity, depth=depth)
    return [r["neighbour"] for r in result.data()]


def get_context_for_entity(entity, limit=10):
    """
    Main function your router calls.
    Takes an entity name, queries Neo4j, returns a formatted context string
    ready to pass to the LLM — same shape as what retrieval.py returns.
    """
    driver = get_driver()
    try:
        with driver.session() as session:
            results = session.execute_read(queryEntity, entity.lower())

        if not results:
            return f"No relationships found for entity: {entity}", []

        # format as readable context
        lines = []
        sources = []
        for r in results[:limit]:
            lines.append(f"- {r['subject']} -> {r['relation']} -> {r['object']}")
            if r.get("comment_id"):
                sources.append(r["comment_id"])

        context = f"Knowledge graph relationships for '{entity}':\n" + "\n".join(lines)
        return context, list(set(sources))

    except Exception as e:
        print(f"[knowledge_graph] query error: {e}")
        return "", []
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Visualization — saves an HTML file viewable in browser
# Kept from notebook, useful for demo/report.
# ---------------------------------------------------------------------------

def visualize(all_triplets, output_path="knowledge_graph.html", limit=100):
    from pyvis.network import Network

    net = Network(height="600px", width="100%", directed=True, cdn_resources="in_line")

    for triplet in all_triplets[:limit]:
        net.add_node(triplet["subject"], label=triplet["subject"])
        net.add_node(triplet["object"], label=triplet["object"])
        net.add_edge(triplet["subject"], triplet["object"], label=triplet["relation"])

    net.save_graph(output_path)
    print(f"[knowledge_graph] Graph saved to {output_path}")


# ---------------------------------------------------------------------------
# Run directly to load and test
# python knowledge_graph.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/analysis_results.csv"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    # load graph
    all_triplets = load_graph(csv_path, limit=limit)

    # visualize
    visualize(all_triplets)

    # test query
    print("\n--- Test query: 'trump' ---")
    context, sources = get_context_for_entity("trump")
    print(context)
    print(f"Sources: {sources[:5]}")