"""
connectors.py

One connector interface for Postgres, MySQL, and Oracle.
Industry pattern: a single class wrapping SQLAlchemy, instantiated once
per database, all exposed through the same .execute() method.
"""

import oracledb
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError


class DatabaseConnector:
    """Wraps one database connection behind a consistent interface."""

    def __init__(self, name: str, connection_url: str, dialect: str):
        self.name = name
        self.dialect = dialect
        self.engine = create_engine(connection_url, pool_pre_ping=True)

    def execute(self, sql: str, timeout_seconds: int = 10):
        """Run a SQL string and return (rows, column_names)."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                columns = list(result.keys())
                return rows, columns
        except (OperationalError, ProgrammingError) as e:
            raise RuntimeError(f"[{self.name}] Query failed: {e}")

    def test_connection(self):
        """Quick sanity check -- does the connection even work?"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1 FROM dual" if self.dialect == "oracle" else "SELECT 1"))
            return True
        except Exception as e:
            print(f"[{self.name}] Connection failed: {e}")
            return False


# Oracle DSN built explicitly via oracledb, not a raw URL string.
# This is the fix: oracledb.makedsn() builds a proper Easy Connect string
# that correctly targets the freepdb1 PDB instead of silently landing in
# CDB$ROOT, which is what was happening with the plain URL approach.
oracle_dsn = oracledb.makedsn("localhost", 1521, service_name="freepdb1")
oracle_connection_url = f"oracle+oracledb://system:root@{oracle_dsn}"

CONNECTORS = {
    "postgres": DatabaseConnector(
        name="postgres",
        connection_url="postgresql+psycopg2://postgres:postgres@localhost:5432/postgres",
        dialect="postgres",
    ),
    "mysql": DatabaseConnector(
        name="mysql",
        connection_url="mysql+pymysql://root:root@localhost:3307/hr_db",
        dialect="mysql",
    ),
    "oracle": DatabaseConnector(
        name="oracle",
        connection_url=oracle_connection_url,
        dialect="oracle",
    ),
}


if __name__ == "__main__":
    print("Testing connections...\n")
    for db_name, connector in CONNECTORS.items():
        ok = connector.test_connection()
        status = "OK" if ok else "FAILED"
        print(f"  {db_name:10s} -> {status}")

    print("\nTesting Postgres (Sales)...")
    rows, columns = CONNECTORS["postgres"].execute("SELECT * FROM customers LIMIT 3")
    print(f"Columns: {list(columns)}")
    for row in rows:
        print(row)

    print("\nTesting MySQL (HR)...")
    rows, columns = CONNECTORS["mysql"].execute("SELECT * FROM employees LIMIT 3")
    print(rows)

    print("\nTesting Oracle (Inventory)...")
    rows, columns = CONNECTORS["oracle"].execute("SELECT * FROM products WHERE ROWNUM <= 3")
    print(rows)