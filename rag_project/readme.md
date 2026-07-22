# Text-to-Query RAG System

A schema-based agentic RAG system that allows customers to query 
SQL databases using natural language.

## Architecture
- Layer 1: Input sanitization
- Layer 2: Schema retrieval (ChromaDB + sentence-transformers)
- Layer 3: SQL generation (Groq LLM with tool calling)
- Layer 4: SQL validation (sqlglot AST)
- Layer 5: Execution (SQLAlchemy — Postgres, MySQL, Oracle)
- Layer 6: Response (text summary + charts)

## Setup

### Prerequisites
- Python 3.10+
- Docker Desktop

### Install dependencies
pip install -r requirements.txt

### Start databases
docker run --name pg_db -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres
docker run --name mysql_db -e MYSQL_ROOT_PASSWORD=root -p 3307:3306 -d mysql
docker run --name oracle_db -e ORACLE_PWD=root -p 1521:1521 -d gvenzl/oracle-xe

### Load sample data
cd sql_setup
docker exec -i pg_db psql -U postgres -d postgres < postgres_sales.sql
docker exec -i mysql_db mysql -u root -proot < mysql_setup.sql
docker exec -i oracle_db sqlplus system/root@localhost:1521/XEPDB1 @oracle_setup.sql

### Set environment variables
Copy .env.example to .env and fill in your keys

### Build schema index
python schema_index.py

### Run the app
streamlit run app.py