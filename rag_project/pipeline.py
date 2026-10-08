"""
pipeline.py

Step 5 of the build plan: chain validation → execution.

This is the first time connectors.py and validation.py talk to each other.
A raw SQL string comes in, gets validated (Layer 4), and only if it passes
does it get handed to the right database connector (Layer 5).

Nothing about LLMs yet — just proving the safety pipeline works end to end
with hand-written SQL before the LLM enters the picture in Step 6.
"""

from validation import validate_sql, ValidationError
from connectors import CONNECTORS


def run_safe_query(sql: str, db_name: str, allowed_tables: set):
    """
    Validate a SQL string then execute it against the named database.
    Raises ValidationError if the SQL fails any safety check.
    Raises RuntimeError if the database execution itself fails.
    Returns (rows, columns) on success.
    """
    if db_name not in CONNECTORS:
        raise ValueError(f"Unknown database '{db_name}'. Available: {list(CONNECTORS.keys())}")

    connector = CONNECTORS[db_name]

    # Layer 4 — validate before anything touches the real database
    validated_sql = validate_sql(sql, connector.dialect, allowed_tables)

    # Layer 5 — only now execute against the live database
    rows, columns = connector.execute(validated_sql)
    return rows, columns


if __name__ == "__main__":
    ALLOWED = {
        "customers", "orders", "sales_reps",      # postgres
        "employees", "attendance", "payroll",      # mysql
        "products", "suppliers", "stock_movements" # oracle
    }

    print("=" * 55)
    print("Test 1: Valid SELECT against Postgres (should pass)")
    print("=" * 55)
    try:
        rows, columns = run_safe_query("SELECT * FROM customers", "postgres", ALLOWED)
        print(f"Columns : {list(columns)}")
        print(f"Rows    : {len(rows)} returned")
        for row in rows[:3]:
            print(f"  {row}")
    except (ValidationError, RuntimeError) as e:
        print(f"FAILED: {e}")

    print()
    print("=" * 55)
    print("Test 2: Valid SELECT against MySQL (should pass)")
    print("=" * 55)
    try:
        rows, columns = run_safe_query("SELECT * FROM employees", "mysql", ALLOWED)
        print(f"Columns : {list(columns)}")
        print(f"Rows    : {len(rows)} returned")
        for row in rows[:3]:
            print(f"  {row}")
    except (ValidationError, RuntimeError) as e:
        print(f"FAILED: {e}")

    print()
    print("=" * 55)
    print("Test 3: Valid SELECT against Oracle (should pass)")
    print("=" * 55)
    try:
        rows, columns = run_safe_query("SELECT * FROM products", "oracle", ALLOWED)
        print(f"Columns : {list(columns)}")
        print(f"Rows    : {len(rows)} returned")
        for row in rows[:3]:
            print(f"  {row}")
    except (ValidationError, RuntimeError) as e:
        print(f"FAILED: {e}")

    print()
    print("=" * 55)
    print("Test 4: DROP TABLE (should be rejected by validation)")
    print("=" * 55)
    try:
        rows, columns = run_safe_query("DROP TABLE customers", "postgres", ALLOWED)
        print("ERROR: This should have been rejected but wasn't!")
    except ValidationError as e:
        print(f"Correctly rejected before reaching DB: {e}")
    except RuntimeError as e:
        print(f"ERROR: Reached the DB before being caught: {e}")

    print()
    print("=" * 55)
    print("Test 5: Table not in whitelist (should be rejected)")
    print("=" * 55)
    try:
        rows, columns = run_safe_query("SELECT * FROM admin_secrets", "postgres", ALLOWED)
        print("ERROR: This should have been rejected but wasn't!")
    except ValidationError as e:
        print(f"Correctly rejected before reaching DB: {e}")

    print()
    print("=" * 55)
    print("Test 6: DELETE statement (should be rejected)")
    print("=" * 55)
    try:
        rows, columns = run_safe_query(
            "DELETE FROM orders WHERE order_id = 1", "mysql", ALLOWED
        )
        print("ERROR: This should have been rejected but wasn't!")
    except ValidationError as e:
        print(f"Correctly rejected before reaching DB: {e}")

    print()
    print("All tests done.")
    print("Tests 1-3 should show real rows. Tests 4-6 should show 'Correctly rejected'.")