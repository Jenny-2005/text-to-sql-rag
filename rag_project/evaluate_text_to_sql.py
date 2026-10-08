"""
Evaluation Harness for Text-to-SQL RAG Pipeline
=================================================

WHAT THIS MEASURES:
1. SQL Validity Rate      -> does the generated SQL parse correctly (via sqlglot)?
2. Execution Success Rate -> does the SQL actually run against the DB without error?
3. Result Match Rate      -> does the query RESULT match the expected/ground-truth result?
4. Exact SQL Match (opt.) -> does generated SQL exactly match a reference SQL (strict, optional)
5. Average Latency        -> how long does generation take per query?

HOW TO USE:
1. Fill in `test_cases.json` (see generate_sample_test_cases() below for the format).
2. Wire up the three functions marked "PLUG IN YOUR PIPELINE HERE" to call your
   actual RAG pipeline (schema retrieval + Groq generation) and your DB executor.
3. Run: python evaluate_text_to_sql.py
4. Read the printed report + look at results.json for per-query breakdown.

This is intentionally dependency-light (just sqlglot) so it runs standalone.
"""

import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import sqlglot
from sqlglot import exp


# ---------------------------------------------------------------------------
# 1. PLUG IN YOUR PIPELINE HERE
# ---------------------------------------------------------------------------
# Replace the bodies of these three functions with calls into your actual
# text-to-SQL RAG system. Keeping them as separate functions means you can
# evaluate any stage in isolation later (e.g. just retrieval quality).

def generate_and_run(natural_language_query: str, db_dialect: str = "postgres") -> str:
    """
    Call your RAG pipeline: retrieval (ChromaDB + sentence-transformers) ->
    Groq LLaMA-3.3-70B generation -> return raw SQL string.

    Example of what to put here:
        from your_pipeline import run_text_to_sql
        return run_text_to_sql(natural_language_query, dialect=db_dialect)
    """
    raise NotImplementedError("Wire this up to your actual pipeline's generate step")


def execute_sql(sql: str, db_dialect: str = "postgres") -> list[dict]:
    """
    Execute SQL against your target DB via SQLAlchemy and return rows as list of dicts.

    Example:
        from your_db import get_engine
        engine = get_engine(db_dialect)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            return [dict(row._mapping) for row in result]
    """
    raise NotImplementedError("Wire this up to your actual DB executor")


def validate_sql_ast(sql: str, db_dialect: str = "postgres") -> tuple[bool, Optional[str]]:
    """
    Your AST-level validation layer (sqlglot). If you already have this function
    in your pipeline, just import and call it instead of reimplementing here.
    """
    try:
        parsed = sqlglot.parse_one(sql, read=db_dialect)
        if parsed is None:
            return False, "Empty parse result"
        return True, None
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# 2. TEST CASE FORMAT
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    query: str                          # natural language question
    expected_sql: Optional[str] = None  # reference SQL (optional, for exact-match)
    expected_result: Optional[list] = None  # ground-truth rows, e.g. [{"count": 5}]
    dialect: str = "postgres"
    difficulty: str = "medium"          # easy / medium / hard -- lets you break down accuracy by difficulty


@dataclass
class EvalResult:
    id: str
    query: str
    generated_sql: str = ""
    is_valid_sql: bool = False
    validation_error: Optional[str] = None
    executed_successfully: bool = False
    execution_error: Optional[str] = None
    result_matches_expected: Optional[bool] = None  # None if no expected_result given
    exact_sql_match: Optional[bool] = None
    latency_seconds: float = 0.0
    difficulty: str = "medium"


def generate_sample_test_cases(path: str = "test_cases.json") -> None:
    """Creates a sample test_cases.json so you can see the expected format.
    Replace the contents with real queries from your IOC schema."""
    samples = [
        {
            "id": "q1",
            "query": "How many customers signed up in the last 30 days?",
            "expected_sql": "SELECT COUNT(*) FROM customers WHERE signup_date >= CURRENT_DATE - INTERVAL '30 days'",
            "expected_result": [{"count": 42}],
            "dialect": "postgres",
            "difficulty": "easy",
        },
        {
            "id": "q2",
            "query": "List the top 5 products by total sales revenue",
            "expected_sql": None,
            "expected_result": None,
            "dialect": "postgres",
            "difficulty": "medium",
        },
        {
            "id": "q3",
            "query": "Show customers who redeemed more reward points than the average across all branches",
            "expected_sql": None,
            "expected_result": None,
            "dialect": "postgres",
            "difficulty": "hard",
        },
    ]
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Sample test cases written to {path} -- replace with real queries from your schema.")


def load_test_cases(path: str = "test_cases.json") -> list[TestCase]:
    with open(path) as f:
        raw = json.load(f)
    return [TestCase(**item) for item in raw]


# ---------------------------------------------------------------------------
# 3. RESULT COMPARISON LOGIC
# ---------------------------------------------------------------------------

def results_match(actual: list[dict], expected: list[dict]) -> bool:
    """
    Order-independent comparison of result sets. Converts rows to frozensets
    of (key, value) pairs so row order and column order don't cause false negatives.
    For numeric columns you may want to add tolerance -- adjust as needed.
    """
    if actual is None or expected is None:
        return False
    try:
        actual_set = {frozenset(row.items()) for row in actual}
        expected_set = {frozenset(row.items()) for row in expected}
        return actual_set == expected_set
    except TypeError:
        # Unhashable values (e.g. nested dicts) -- fall back to sorted list compare
        return sorted(actual, key=str) == sorted(expected, key=str)


def normalize_sql(sql: str, dialect: str = "postgres") -> Optional[str]:
    """Normalize SQL via sqlglot for exact-match comparison (ignores whitespace/casing)."""
    try:
        return sqlglot.parse_one(sql, read=dialect).sql(dialect=dialect, normalize=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. EVALUATION LOOP
# ---------------------------------------------------------------------------

def evaluate(test_cases: list[TestCase]) -> list[EvalResult]:
    results = []

    for tc in test_cases:
        result = EvalResult(id=tc.id, query=tc.query, difficulty=tc.difficulty)

        # --- Generation ---
        start = time.time()
        try:
            sql = generate_and_run(tc.query, db_dialect=tc.dialect)
            result.generated_sql = sql
        except NotImplementedError:
            raise  # stop immediately -- you haven't wired up the pipeline yet
        except Exception as e:
            result.execution_error = f"Generation failed: {e}"
            result.latency_seconds = time.time() - start
            results.append(result)
            continue
        result.latency_seconds = time.time() - start

        # --- AST Validation ---
        is_valid, err = validate_sql_ast(sql, db_dialect=tc.dialect)
        result.is_valid_sql = is_valid
        result.validation_error = err

        # --- Exact SQL match (only if reference SQL provided) ---
        if tc.expected_sql:
            norm_gen = normalize_sql(sql, tc.dialect)
            norm_exp = normalize_sql(tc.expected_sql, tc.dialect)
            result.exact_sql_match = (norm_gen is not None and norm_gen == norm_exp)

        # --- Execution (only attempt if AST-valid) ---
        if is_valid:
            try:
                rows = execute_sql(sql, db_dialect=tc.dialect)
                result.executed_successfully = True

                if tc.expected_result is not None:
                    result.result_matches_expected = results_match(rows, tc.expected_result)
            except NotImplementedError:
                raise
            except Exception as e:
                result.executed_successfully = False
                result.execution_error = str(e)

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 5. REPORTING
# ---------------------------------------------------------------------------

def print_report(results: list[EvalResult]) -> None:
    n = len(results)
    valid = sum(r.is_valid_sql for r in results)
    executed = sum(r.executed_successfully for r in results)

    result_checked = [r for r in results if r.result_matches_expected is not None]
    result_correct = sum(r.result_matches_expected for r in result_checked)

    exact_checked = [r for r in results if r.exact_sql_match is not None]
    exact_correct = sum(r.exact_sql_match for r in exact_checked)

    avg_latency = sum(r.latency_seconds for r in results) / n if n else 0

    print("=" * 60)
    print(f"TEXT-TO-SQL EVALUATION REPORT  ({n} queries)")
    print("=" * 60)
    print(f"SQL Validity Rate      : {valid}/{n}  ({100*valid/n:.1f}%)")
    print(f"Execution Success Rate : {executed}/{n}  ({100*executed/n:.1f}%)")
    if result_checked:
        print(f"Result Match Rate      : {result_correct}/{len(result_checked)}  "
              f"({100*result_correct/len(result_checked):.1f}%)  "
              f"[on {len(result_checked)} queries with ground-truth results]")
    if exact_checked:
        print(f"Exact SQL Match Rate   : {exact_correct}/{len(exact_checked)}  "
              f"({100*exact_correct/len(exact_checked):.1f}%)  "
              f"[strict, only on {len(exact_checked)} queries with reference SQL]")
    print(f"Avg. Latency           : {avg_latency:.2f}s per query")
    print("=" * 60)

    # Breakdown by difficulty
    difficulties = sorted(set(r.difficulty for r in results))
    if len(difficulties) > 1:
        print("\nBreakdown by difficulty:")
        for d in difficulties:
            subset = [r for r in results if r.difficulty == d]
            v = sum(r.is_valid_sql for r in subset)
            print(f"  {d:8s}: {v}/{len(subset)} valid  "
                  f"({100*v/len(subset):.1f}%)")

    # Failures worth looking at
    failures = [r for r in results if not r.is_valid_sql or r.execution_error]
    if failures:
        print(f"\n{len(failures)} queries had issues -- see results.json for details.")

    with open("results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print("\nFull per-query results saved to results.json")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    if not os.path.exists("test_cases.json"):
        generate_sample_test_cases()
        print("\nEdit test_cases.json with real queries from your schema, then re-run.")
    else:
        test_cases = load_test_cases()
        print(f"Loaded {len(test_cases)} test cases. Running evaluation...\n")
        try:
            results = evaluate(test_cases)
            print_report(results)
        except NotImplementedError as e:
            print(f"\n[SETUP NEEDED] {e}")
            print("Open evaluate_text_to_sql.py and fill in generate_sql() and execute_sql()")
            print("with calls into your actual pipeline before running this.")
