"""
sql_generation.py — updated for Step 7

Now uses schema_index.retrieve_schema() instead of hardcoded schema.
Full pipeline: question → Layer 2 (schema retrieval) → Layer 3 (LLM)
               → Layer 4 (validation) → Layer 5 (execution)
"""

import os
import json
from groq import Groq
from sentence_transformers import SentenceTransformer
from pipeline import run_safe_query, ValidationError
from schema_index import retrieve_schema, MODEL_NAME, CHROMA_PATH
from input_layer import process_input
# ── Load embedding model ONCE at startup (not per query) ─
print("Loading embedding model...")
_embedding_model = SentenceTransformer(MODEL_NAME)
print("Model loaded.\n")

# ── Groq client ──────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Keywords for schema/description questions ────────────
SCHEMA_KEYWORDS = [
    "describe", "description", "what tables", "which tables",
    "schema", "what columns", "tell me about", "brief", "overview",
    "what is in", "what data", "structure", "columns in",
    "what does", "explain the", "list tables", "list columns"
]

# ── Tool definition for Groq function calling ────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Execute a SQL SELECT query against the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT query to execute."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this query answers the question."
                    }
                },
                "required": ["sql", "reasoning"]
            }
        }
    }
]

# ── Dialect syntax rules, including explicit relative-date guidance ──
# Each dialect gets an explicit relative-date pattern + worked example,
# and TO_DATE()/equivalents are scoped to fixed dates only.

DIALECT_SYNTAX = {
    "postgres": (
        "Use standard SQL with LIMIT for row limiting.\n"
        "For relative dates, use NOW() - INTERVAL 'n months' / 'n days' / 'n years'.\n"
        "Example — 'last 12 months': WHERE order_date > NOW() - INTERVAL '12 months'\n"
        "Only use TO_DATE()/date literals for a fixed, explicitly stated date "
        "(e.g. 'since Jan 2024'), never for relative phrases like 'last N months'."
    ),
    "mysql": (
        "Use standard SQL with LIMIT for row limiting.\n"
        "For relative dates, use DATE_SUB(NOW(), INTERVAL n MONTH/DAY/YEAR).\n"
        "Example — 'last 12 months': WHERE order_date > DATE_SUB(NOW(), INTERVAL 12 MONTH)\n"
        "Only use STR_TO_DATE()/date literals for a fixed, explicitly stated date "
        "(e.g. 'since Jan 2024'), never for relative phrases like 'last N months'."
    ),
    "oracle": (
        "Use FETCH FIRST n ROWS ONLY instead of LIMIT. Use 'Y'/'N' for boolean fields.\n"
        "For relative dates, use SYSDATE - INTERVAL 'n' MONTH/DAY/YEAR, or "
        "ADD_MONTHS(SYSDATE, -n) for months.\n"
        "Example — 'last 12 months': WHERE movement_date > SYSDATE - INTERVAL '12' MONTH\n"
        "Only use TO_DATE('literal', 'format') for a fixed, explicitly stated date "
        "(e.g. TO_DATE('2024-01-01','YYYY-MM-DD')). NEVER pass a phrase like "
        "'12 months ago' as either the value or the format string to TO_DATE() — "
        "TO_DATE() cannot parse relative time expressions."
    ),
}


def is_schema_question(question: str) -> bool:
    return any(kw in question.lower() for kw in SCHEMA_KEYWORDS)


def build_prompt(question: str, schema_text: str, dialect: str) -> str:
    syntax_note = DIALECT_SYNTAX.get(dialect, "")
    return f"""You are an expert SQL developer helping query a business database.

RELEVANT SCHEMA:
{schema_text}

DIALECT RULES ({dialect.upper()}):
{syntax_note}

Write a single SQL SELECT query to answer this question:
"{question}"

STRICT RULES:
- Only SELECT queries — never INSERT/UPDATE/DELETE/DROP/ALTER
- Only use tables shown in the schema above
- Never query system tables
- Always include a row limit (LIMIT 1000 or FETCH FIRST 1000 ROWS ONLY)
- Use the execute_sql_query tool to return your answer
- For trend/comparison questions, always GROUP BY both year AND month
  to get multiple data points per line
- For relative date phrases ("last N months/days/years", "this year",
  "past week"), use the relative-date syntax shown in DIALECT RULES above —
  never pass the natural-language phrase itself into a date-parsing function
"""


def generate_and_run(question: str) -> tuple:
    """
    Full pipeline — Layer 2 → Layer 3 → Layer 4 → Layer 5:
    1. Retrieve relevant schema via vector search (Layer 2)
    2. Call Groq LLM with retrieved schema (Layer 3)
    3. Extract SQL from tool call response
    4. Validate SQL (Layer 4)
    5. Execute against correct database (Layer 5)
    """
    print(f"\nQuestion : {question}")
    print("-" * 50)

    # Layer 2 — retrieve schema using pre-loaded model
    schema_result = retrieve_schema(
        question,
        top_k=3,
        model=_embedding_model  # pass pre-loaded model
    )

    database       = schema_result["database"]
    dialect        = schema_result["dialect"]
    allowed_tables = schema_result["allowed_tables"]
    schema_text    = schema_result["schema_text"]

    print(f"Routed to : {database} | Tables: {allowed_tables}")

    # Layer 3 — call LLM with retrieved schema
    prompt = build_prompt(question, schema_text, dialect)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        tools=TOOLS,
        tool_choice="required",
        temperature=0.3,
        max_tokens=1024
    )

    message = response.choices[0].message
    if not message.tool_calls:
        raise RuntimeError("LLM did not return a tool call — try rephrasing")

    tool_input = json.loads(message.tool_calls[0].function.arguments)
    sql        = tool_input["sql"]
    reasoning  = tool_input.get("reasoning", "")

    print(f"Reasoning : {reasoning}")
    print(f"Generated SQL:\n{sql}")
    print("-" * 50)

    # Layer 4 + 5 — validate then execute
    rows, columns = run_safe_query(sql, database, allowed_tables)
    return rows, columns, database


if __name__ == "__main__":
    print("=" * 55)
    print("Text-to-Query RAG — All 3 Databases")
    print("Postgres (Sales) · MySQL (HR) · Oracle (Inventory)")
    print("Ask anything. Type 'exit' to quit.")
    print("=" * 55)

    while True:
        print()
        question = input("Your question: ").strip()

        # 1. Exit check — includes bye
        if question.lower() in ("exit", "quit", "q", "bye"):
            print("Goodbye!")
            break

        # 2. Empty check
        if not question:
            print("Please enter a question.")
            continue

        # 3. Layer 1 FIRST — before schema check or LLM
        layer1_result = process_input(question,model=_embedding_model)
        if layer1_result["status"] == "rejected":
            print(f"\nQuery rejected: {layer1_result['reason']}")
            continue

        print(f"\nLayer 1 passed (similarity score: {layer1_result['score']})")


        # 5. Schema question check
        if is_schema_question(question):
            print("\nSchema question detected.")
            result = retrieve_schema(question, top_k=3, model=_embedding_model)
            print(f"Relevant schema from {result['database']}:\n")
            print(result["schema_text"])
            continue

        # 6. Full pipeline — Layer 2 → 3 → 4 → 5
        try:
            rows, columns, database = generate_and_run(question)
            print(f"\nResult from {database} ({len(rows)} rows returned)")
            print(f"Columns : {list(columns)}")
            for row in rows[:10]:
                print(f"  {row}")
            if len(rows) > 10:
                print(f"  ... and {len(rows) - 10} more rows")
        except ValidationError as e:
            print(f"\nQuery rejected by validation: {e}")
        except RuntimeError as e:
            print(f"\nRuntime error: {e}")
        except Exception as e:
            print(f"\nError: {e}")