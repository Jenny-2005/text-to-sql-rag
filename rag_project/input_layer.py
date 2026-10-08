"""
input_layer.py

Layer 1 — Input validation and sanitization.

Three checks in order:
  1. Length check     — reject inputs that are too long
  2. Write operation  — reject obvious destructive keywords
  3. Relevance check  — uses ChromaDB similarity score (Layer 2)
                        instead of a hardcoded keyword list,
                        making Layer 1 automatically aware of
                        any new databases added to the schema index.

Depends on schema_index.retrieve_schema() for relevance check.
The embedding model is passed in from sql_generation.py so it
is loaded only once across the whole application.
"""

import re
# import chromadb

# ── Constants ────────────────────────────────────────────
BLOCKED_KEYWORDS = [
    "drop", "insert", "alter",
    "truncate", "create", "replace", "grant", "revoke"
]

# Minimum similarity score to consider a question relevant
# to any table in the schema index. Below this → off-topic.
RELEVANCE_THRESHOLD = 0.15

MAX_CHARS = 300


# ── Helper functions ─────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_write_operation(text: str) -> bool:
    """Check for clearly destructive SQL keywords using word boundaries."""
    normalized = clean_text(text)
    for word in BLOCKED_KEYWORDS:
        if re.search(rf"\b{word}\b", normalized):
            return True
    return False


def is_too_long(text: str) -> bool:
    return len(text.strip()) > MAX_CHARS


def is_relevant(question: str, model) -> tuple:
    """
    Check relevance using ChromaDB similarity search.
    Returns (is_relevant: bool, top_score: float).

    This replaces the hardcoded KNOWN_TOPICS keyword list —
    any question that scores above RELEVANCE_THRESHOLD against
    the schema index is considered on-topic, regardless of
    what specific words it uses.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    from schema_index import MODEL_NAME, CHROMA_PATH, COLLECTION_NAME
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        question_embedding = model.encode(question).tolist()
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=1,
            include=["distances"]
        )
        distance = results["distances"][0][0]
        top_score = round(1 - distance, 3)
        return top_score >= RELEVANCE_THRESHOLD, top_score
    except Exception as e:
        # If ChromaDB is unavailable, fail open — let the query through
        # so the rest of the pipeline can handle it
        print(f"[Layer 1] Relevance check unavailable: {e}. Allowing query.")
        return True, 0.0


# ── Main Layer 1 function ────────────────────────────────
def process_input(raw_query: str, model=None) -> dict:
    """
    Validate and sanitize incoming customer query.
    Returns a dict with status, reason, and cleaned query.
    """
    
    # Check 1 — empty
    if not raw_query or not raw_query.strip():
        return {"status": "rejected", "reason": "empty input", "query": None}

    # Check 2 — too long
    if is_too_long(raw_query):
        return {"status": "rejected", "reason": "input too long (max 300 chars)", "query": None}

    # Check 3 — obvious destructive keywords
    if is_write_operation(raw_query):
        return {"status": "rejected", "reason": "write operations not allowed", "query": None}

    # Check 4 — relevance via ChromaDB similarity
    # If no model passed, load one (fallback for standalone testing)
    if model is None:
        from sentence_transformers import SentenceTransformer
        from schema_index import MODEL_NAME
        model = SentenceTransformer(MODEL_NAME)

    relevant, score = is_relevant(raw_query, model)
    if not relevant:
        return {
            "status": "rejected",
            "reason": f"off-topic query (similarity score: {score}, threshold: {RELEVANCE_THRESHOLD})",
            "query": None
        }

    return {
        "status":  "accepted",
        "reason":  None,
        "query":   clean_text(raw_query),
        "score":   score
    }


# ── Standalone test ──────────────────────────────────────
if __name__ == "__main__":
    print("Loading model for standalone test...")
    from sentence_transformers import SentenceTransformer
    from schema_index import MODEL_NAME
    model = SentenceTransformer(MODEL_NAME)

    test_inputs = [
        "show me all employees in department Engineering",
        "what is the weather today",
        "drop the customers table",
        "which suppliers have rating above 4?",
        "",
        "a" * 400,
        "show me top 5 products by unit price",
        "bye",
        "total revenue by sales channel",
    ]

    print("\n--- Layer 1: Input Validation (ChromaDB-based) ---")
    for query in test_inputs:
        result = process_input(query, model=model)
        display = query[:60] + "..." if len(query) > 60 else query
        print(f"\nInput  : '{display}'")
        print(f"Status : {result['status']}")
        if result["status"] == "rejected":
            print(f"Reason : {result['reason']}")
        else:
            print(f"Score  : {result['score']}")