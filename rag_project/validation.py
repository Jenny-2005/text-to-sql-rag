"""
validation.py

Step 4 of the build plan: Layer 4 (Validation) from your architecture.

Takes a raw SQL string an LLM might generate, and proves it's safe before
it's allowed anywhere near Step 3's connectors.py. Four checks, in order:

  1. SELECT only?      -- reject DROP/DELETE/UPDATE/INSERT immediately
  2. AST parse          -- catch destructive statements hidden in subqueries
  3. Table whitelist     -- only allow tables we explicitly retrieved/permit
  4. Inject LIMIT        -- force a row cap if the query doesn't have one

Tested standalone here with hand-written SQL -- both valid and deliberately
broken -- before any LLM enters the picture (that's Step 5/6, later).
"""

import sqlglot
from sqlglot import exp


class ValidationError(Exception):
    """Raised when a SQL query fails any validation check."""
    pass


# Dialect names sqlglot expects, mapped from the dialect names used in
# connectors.py -- keeps the two files speaking the same language.
DIALECT_MAP = {
    "postgres": "postgres",
    "mysql": "mysql",
    "oracle": "oracle",
}


def check_select_only(sql: str) -> None:
    """Fast, cheap string-level check before we even bother parsing.
    Rejects anything that isn't a SELECT (including WITH...SELECT CTEs),
    and catches obvious destructive keywords even if they're disguised
    with extra whitespace/casing."""
    normalized = sql.strip().upper()

    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        raise ValidationError(
            f"Only SELECT statements are allowed. Got: {normalized[:30]}..."
        )

    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
    for word in forbidden:
        import re
        if re.search(rf"\b{word}\b", normalized):
            raise ValidationError(f"Forbidden keyword detected: {word}")


def parse_and_check_ast(sql: str, dialect: str) -> exp.Expression:
    """Parse into an AST and walk the whole tree -- not just the first
    statement -- to catch destructive operations hidden in subqueries,
    CTEs, or anywhere else a string-only check would miss."""
    sqlglot_dialect = DIALECT_MAP.get(dialect, dialect)

    try:
        parsed = sqlglot.parse_one(sql, dialect=sqlglot_dialect)
    except sqlglot.errors.ParseError as e:
        raise ValidationError(f"SQL failed to parse: {e}")

    if not isinstance(parsed, exp.Select):
        raise ValidationError(
            f"Root statement is not a SELECT (found: {type(parsed).__name__})"
        )

    destructive_types = (exp.Drop, exp.Delete, exp.Update, exp.Insert, exp.Alter)
    for node in parsed.walk():
        if isinstance(node[0], destructive_types):
            raise ValidationError(
                f"Destructive operation found inside query: {type(node[0]).__name__}"
            )

    return parsed


def check_table_whitelist(parsed_sql: exp.Expression, allowed_tables: set) -> list:
    """Extract every table referenced in the query and confirm each one
    is in the allowed list. Table names are compared case-insensitively
    since dialects differ on default casing (Oracle uppercases, etc).

    CTE names (e.g. the 'regional_sales' in WITH regional_sales AS (...))
    are excluded -- they're not real tables, just aliases the query defines
    for itself. The whitelist still applies to whatever real tables are
    referenced *inside* the CTE body."""
    cte_names = {cte.alias_or_name.lower() for cte in parsed_sql.find_all(exp.CTE)}

    tables_used = [
        t.name.lower() for t in parsed_sql.find_all(exp.Table)
        if t.name.lower() not in cte_names
    ]
    allowed_lower = {t.lower() for t in allowed_tables}

    for table in tables_used:
        if table not in allowed_lower:
            raise ValidationError(
                f"Table '{table}' is not in the allowed list: {allowed_tables}"
            )

    return tables_used


def ensure_limit(parsed_sql: exp.Expression, dialect: str, max_rows: int = 1000) -> exp.Expression:
    """If no LIMIT clause exists, inject one. Oracle uses different syntax
    under the hood (FETCH FIRST / ROWNUM) but sqlglot's .limit() handles
    the dialect-correct SQL generation for us when we call .sql(dialect=...)."""
    if not parsed_sql.find(exp.Limit):
        parsed_sql = parsed_sql.limit(max_rows)
    return parsed_sql


def validate_sql(sql: str, dialect: str, allowed_tables: set, max_rows: int = 1000) -> str:
    """Run all four checks in order and return a safe, validated SQL string
    ready to hand to connectors.py for execution."""
    check_select_only(sql)
    parsed = parse_and_check_ast(sql, dialect)
    check_table_whitelist(parsed, allowed_tables)
    parsed = ensure_limit(parsed, dialect, max_rows)

    sqlglot_dialect = DIALECT_MAP.get(dialect, dialect)
    return parsed.sql(dialect=sqlglot_dialect)


if __name__ == "__main__":
    ALLOWED_TABLES = {
        "customers", "orders", "sales_reps",      # postgres
        "employees", "attendance", "payroll",      # mysql
        "products", "suppliers", "stock_movements" # oracle
    }

    test_cases = [
        # (label, sql, dialect, should_pass)
        ("Valid SELECT, no LIMIT (should get one injected)",
         "SELECT * FROM customers", "postgres", True),

        ("Valid SELECT, already has LIMIT",
         "SELECT * FROM orders LIMIT 50", "mysql", True),

        ("Valid SELECT against Oracle dialect",
         "SELECT * FROM customers", "oracle", True),

        ("Destructive: DROP TABLE",
         "DROP TABLE customers", "postgres", False),

        ("Destructive: DELETE",
         "DELETE FROM orders WHERE order_id = 1", "postgres", False),

        ("Destructive hidden in subquery",
         "SELECT * FROM (DELETE FROM customers RETURNING *) AS x", "postgres", False),

        ("Table not in whitelist",
         "SELECT * FROM admin_secrets", "postgres", False),

        ("Column name containing forbidden word (should NOT false-positive)",
         "SELECT updated_at FROM orders", "postgres", True),

        ("UPDATE statement",
         "UPDATE customers SET city = 'Hacked'", "postgres", False),
    ]

    print(f"Running {len(test_cases)} validation test cases...\n")
    passed = 0
    failed = 0

    for label, sql, dialect, should_pass in test_cases:
        try:
            result_sql = validate_sql(sql, dialect, ALLOWED_TABLES)
            actually_passed = True
        except ValidationError as e:
            actually_passed = False
            error_msg = str(e)

        if actually_passed == should_pass:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL (unexpected result)"
            failed += 1

        print(f"[{status}] {label}")
        if actually_passed:
            print(f"         -> Validated SQL: {result_sql}")
        else:
            print(f"         -> Rejected: {error_msg}")
        print()

    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")