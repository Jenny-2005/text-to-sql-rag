"""
schema_index.py

Step 7 of the build plan: Layer 2 (Schema Retrieval).

TWO PARTS:
  1. build_index()  — offline, run once. Embeds all table/column
                      descriptions and stores them in ChromaDB.
  2. retrieve_schema() — online, called per query. Embeds the
                         incoming question, searches ChromaDB for
                         the most relevant schema chunks, returns
                         the matching database name + schema text.

This replaces the hardcoded ORACLE_SCHEMA string in sql_generation.py
with real vector-based retrieval across all three databases.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Embedding model (runs locally, no API key needed) ────
# all-MiniLM-L6-v2 is small (90MB), fast, and good enough
# for schema similarity matching.
MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"   # where ChromaDB stores its files
COLLECTION_NAME = "schema_index"

# ── Full schema descriptions for all three databases ─────
# Each entry is one "chunk" that gets embedded independently.
# The more descriptive these are, the better the retrieval.
# Format: (chunk_id, database, dialect, table, text)

SCHEMA_CHUNKS = [

    # ── POSTGRES · SALES ─────────────────────────────────
    (
        "pg_customers",
        "postgres",
        "postgres",
        "customers",
        """Database: Postgres Sales Department.
Table: customers.
Columns: customer_id (primary key), full_name (customer full name),
email (unique email address), city (Indian city), state (Indian state),
phone (phone number), segment (customer segment: Retail, Wholesale, Enterprise, SMB, Government),
signup_date (date customer signed up), is_active (TRUE if active customer).
Use this table for: customer profiles, customer count, city-wise customers,
segment analysis, active vs inactive customers, customer lookup by name or email."""
    ),
    (
        "pg_orders",
        "postgres",
        "postgres",
        "orders",
        """Database: Postgres Sales Department.
Table: orders.
Columns: order_id (primary key), customer_id (links to customers),
rep_id (links to sales_reps), order_date (date of order),
status (Pending/Confirmed/Shipped/Delivered/Cancelled/Returned),
channel (Online/In-Store/Phone/Partner/Direct),
discount_pct (discount percentage applied), total_amount (order value in INR),
tax_amount (tax charged in INR).
Use this table for: total sales revenue, order count, order status breakdown,
sales by channel, discount analysis, monthly or daily revenue trends."""
    ),
    (
        "pg_sales_reps",
        "postgres",
        "postgres",
        "sales_reps",
        """Database: Postgres Sales Department.
Table: sales_reps.
Columns: rep_id (primary key), full_name (rep name), email,
region (North/South/East/West/Central), hire_date,
target_amount (sales target in INR), achieved_amount (actual sales in INR),
commission_pct (commission percentage), is_active (TRUE if currently employed).
Use this table for: sales rep performance, target vs achievement,
commission analysis, regional breakdown of reps, top performing reps."""
    ),

    # ── MYSQL · HR ────────────────────────────────────────
    (
        "mysql_employees",
        "mysql",
        "mysql",
        "employees",
        """Database: MySQL HR Department.
Table: employees.
Columns: employee_id (primary key), full_name, email,
department (Engineering/Sales/HR/Finance/Marketing/Operations/Legal/Product),
designation (Analyst/Senior Analyst/Manager/Senior Manager/Director/VP/Associate/Lead),
salary (monthly salary in INR), hire_date, manager_id (links to another employee),
employment_status (Active/On Leave/Probation/Resigned/Terminated).
Use this table for: employee count, salary analysis, department headcount,
designation breakdown, active employees, hiring trends, manager hierarchy."""
    ),
    (
        "mysql_attendance",
        "mysql",
        "mysql",
        "attendance",
        """Database: MySQL HR Department.
Table: attendance.
Columns: attendance_id (primary key), employee_id (links to employees),
work_date (date of attendance), check_in (time), check_out (time),
hours_worked (decimal hours), status (Present/Absent/Half Day/Work From Home/Leave),
late_minutes (minutes late), notes (attendance notes).
Use this table for: attendance records, late arrivals, absenteeism analysis,
work from home frequency, hours worked per employee, attendance by date."""
    ),
    (
        "mysql_payroll",
        "mysql",
        "mysql",
        "payroll",
        """Database: MySQL HR Department.
Table: payroll.
Columns: payroll_id (primary key), employee_id (links to employees),
pay_month (month in YYYY-MM format), basic_salary, allowances,
deductions, net_salary (basic + allowances - deductions),
payment_date, payment_status (Paid/Pending/Processing).
Use this table for: salary disbursement, payroll by month, pending payments,
total payroll cost, net salary distribution, deduction analysis."""
    ),

    # ── ORACLE · INVENTORY ───────────────────────────────
    (
        "oracle_products",
        "oracle",
        "oracle",
        "products",
        """Database: Oracle Inventory Department.
Table: products.
Columns: product_id (primary key), product_name,
category (Electronics/Furniture/Stationery/Clothing/Food and Bev),
subcategory (Mobile/Laptop/Chair/Desk etc), unit_price (selling price INR),
cost_price (purchase price INR), weight_kg, supplier_id (links to suppliers),
is_active (Y if active, N if discontinued).
Use this table for: product catalog, price analysis, category breakdown,
profit margin (unit_price minus cost_price), active vs discontinued products,
products by supplier.
Oracle syntax: use FETCH FIRST n ROWS ONLY instead of LIMIT."""
    ),
    (
        "oracle_suppliers",
        "oracle",
        "oracle",
        "suppliers",
        """Database: Oracle Inventory Department.
Table: suppliers.
Columns: supplier_id (primary key), supplier_name, contact_person,
email, phone, city (Indian city), country (India/China/USA/Germany/Japan/South Korea),
rating (1.0 to 5.0 quality rating), is_active (Y/N).
Use this table for: supplier list, supplier ratings, country-wise suppliers,
top rated suppliers, active suppliers, supplier contact details.
Oracle syntax: use FETCH FIRST n ROWS ONLY instead of LIMIT."""
    ),
    (
        "oracle_stock_movements",
        "oracle",
        "oracle",
        "stock_movements",
        """Database: Oracle Inventory Department.
Table: stock_movements.
Columns: movement_id (primary key), product_id (links to products),
movement_type (Purchase/Sale/Return/Transfer/Adjustment/Damage),
quantity (units moved), warehouse (Mumbai Central/Delhi Hub/Bangalore South/
Chennai East/Kolkata North/Pune West), movement_date (date),
reference_no (invoice/reference number), unit_cost (INR), total_cost (INR).
Use this table for: stock movement history, warehouse activity,
purchase vs sale volumes, damaged or returned stock, movement trends by date,
total cost of movements by type or warehouse.
Oracle syntax: use FETCH FIRST n ROWS ONLY instead of LIMIT."""
    ),
]


def build_index():
    """
    OFFLINE STEP — run once.
    Embeds all schema chunks and stores them in ChromaDB.
    Safe to re-run — deletes and rebuilds the collection each time.
    """
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if it exists (clean rebuild)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Deleted existing schema index.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )

    print(f"Embedding {len(SCHEMA_CHUNKS)} schema chunks...")
    ids, documents, metadatas, embeddings = [], [], [], []

    for chunk_id, database, dialect, table, text in SCHEMA_CHUNKS:
        embedding = model.encode(text).tolist()
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append({
            "database": database,
            "dialect": dialect,
            "table": table
        })
        embeddings.append(embedding)
        print(f"  Embedded: {chunk_id}")

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"\nSchema index built successfully.")
    print(f"  {len(SCHEMA_CHUNKS)} chunks stored in ChromaDB at '{CHROMA_PATH}'")
    return collection


def retrieve_schema(question: str, top_k: int = 3, model=None) -> dict:
    """
    ONLINE STEP — called per query.
    Embeds the question, searches ChromaDB for the most relevant
    schema chunks, returns database routing info + schema text.

    Returns a dict:
    {
        "database": "postgres",     # which DB to query
        "dialect":  "postgres",     # SQL dialect to use
        "allowed_tables": {...},    # tables relevant to this question
        "schema_text": "..."        # schema description for the LLM prompt
    }
    """
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    # Embed the question using the same model used during indexing
    question_embedding = model.encode(question).tolist()

    # Search for top-K most similar schema chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Extract results
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nTop {top_k} schema matches for: '{question}'")
    for i, (meta, dist) in enumerate(zip(metas, distances)):
        score = round(1 - dist, 3)  # convert distance to similarity score
        print(f"  {i+1}. {meta['database']}.{meta['table']} (similarity: {score})")

    # The top match determines which database to route to
    top_meta = metas[0]
    top_database = top_meta["database"]
    top_dialect  = top_meta["dialect"]

    # Collect all tables from the top database that appeared in results
    # (in case the question spans multiple tables in the same DB)
    allowed_tables = set()
    schema_parts   = []

    for doc, meta in zip(docs, metas):
        if meta["database"] == top_database:
            allowed_tables.add(meta["table"])
            schema_parts.append(doc)

    schema_text = "\n\n".join(schema_parts)

    top_score = round(1 - distances[0], 3)

    return {
        "database":       top_database,
        "dialect":        top_dialect,
        "allowed_tables": allowed_tables,
        "schema_text":    schema_text,
        "top_score":      top_score
    }


if __name__ == "__main__":
    print("=" * 55)
    print("Building schema index for all 3 databases...")
    print("=" * 55)
    build_index()

    print("\n" + "=" * 55)
    print("Testing retrieval with sample questions...")
    print("=" * 55)

    test_questions = [
        "Show me total revenue by sales channel",
        "Which employees have the highest salary?",
        "What are the top rated suppliers from India?",
        "How many orders were delivered last month?",
        "Show me attendance records for absent employees",
        "What products have the highest profit margin?",
    ]

    for q in test_questions:
        result = retrieve_schema(q)
        print(f"\nQuestion : {q}")
        print(f"Routed to: {result['database']} | Tables: {result['allowed_tables']}")