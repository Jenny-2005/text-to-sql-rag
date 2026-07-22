-- ============================================
-- ORACLE · METADATA REGISTRY
-- Stores project, database, and business
-- metadata for the Text-to-Query RAG system.
-- 
-- Run with:
-- docker cp oracle_metadata.sql oracle_db:/tmp/oracle_metadata.sql
-- docker exec -it oracle_db sqlplus system/root@localhost:1521/XEPDB1 @/tmp/oracle_metadata.sql
-- ============================================

-- Drop existing tables if they exist
BEGIN EXECUTE IMMEDIATE 'DROP TABLE business_metadata CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE database_metadata CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE project_metadata CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/

-- ── Level 1: Project ──────────────────────────────────────
-- Describes the overall RAG project this system belongs to.
CREATE TABLE project_metadata (
    project_id          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    project_name        VARCHAR2(200)  NOT NULL,
    project_type        VARCHAR2(100)  NOT NULL,
    project_description VARCHAR2(1000) NOT NULL,
    created_at          DATE DEFAULT SYSDATE
);
/

-- ── Level 2: Database ────────────────────────────────────
-- Describes each physical database connected to the system.
-- Links back to the project it belongs to.
CREATE TABLE database_metadata (
    db_id               NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    db_name             VARCHAR2(200)  NOT NULL,
    db_type             VARCHAR2(100)  NOT NULL,
    db_description      VARCHAR2(1000) NOT NULL,
    project_id          NUMBER,
    created_at          DATE DEFAULT SYSDATE,
    CONSTRAINT fk_db_project FOREIGN KEY (project_id) REFERENCES project_metadata(project_id)
);
/

-- ── Level 3: Business ────────────────────────────────────
-- Describes the business domain and department each
-- database serves. Links back to its database.
CREATE TABLE business_metadata (
    business_id          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    business_name        VARCHAR2(200)  NOT NULL,
    business_type        VARCHAR2(100)  NOT NULL,
    business_description VARCHAR2(1000) NOT NULL,
    db_id                NUMBER,
    created_at           DATE DEFAULT SYSDATE,
    CONSTRAINT fk_biz_db FOREIGN KEY (db_id) REFERENCES database_metadata(db_id)
);
/

-- ── Insert project metadata ───────────────────────────────
INSERT INTO project_metadata (project_name, project_type, project_description)
VALUES (
    'Text-to-Query RAG System',
    'Agentic Schema-Based RAG',
    'An LLM-powered system that allows customers to query data from multiple databases using natural language. The system retrieves relevant database schemas using vector similarity search, generates SQL using LLM tool calling, validates and executes the query safely, and returns either a natural language summary or a chart config to the frontend.'
);
/

-- ── Insert database metadata ──────────────────────────────
-- Postgres — Sales
INSERT INTO database_metadata (db_name, db_type, db_description, project_id)
VALUES (
    'postgres',
    'PostgreSQL 15 — Relational Database',
    'Sales department database running on PostgreSQL. Contains customer profiles, sales orders, and sales representative records. Primary source for revenue analysis, customer segmentation, order tracking, and sales performance queries. Hosted locally on port 5432.',
    1
);
/

-- MySQL — HR
INSERT INTO database_metadata (db_name, db_type, db_description, project_id)
VALUES (
    'mysql',
    'MySQL 8 — Relational Database',
    'Human Resources department database running on MySQL. Contains employee records, daily attendance logs, and monthly payroll data. Primary source for headcount analysis, salary queries, attendance tracking, and workforce management. Hosted locally on port 3307.',
    1
);
/

-- Oracle — Inventory
INSERT INTO database_metadata (db_name, db_type, db_description, project_id)
VALUES (
    'oracle',
    'Oracle XE 21c — Relational Database',
    'Inventory department database running on Oracle XE. Contains product catalog, supplier information, and stock movement records. Primary source for inventory analysis, supplier performance, stock level queries, and supply chain management. Hosted locally on port 1521.',
    1
);
/

-- ── Insert business metadata ──────────────────────────────
-- Sales business domain
INSERT INTO business_metadata (business_name, business_type, business_description, db_id)
VALUES (
    'Sales Department',
    'Revenue and Customer Management',
    'Responsible for managing customer relationships, processing sales orders, and tracking sales representative performance across five regions (North, South, East, West, Central). Key business questions include total revenue by region, top customers by order value, monthly sales trends, discount analysis, and rep performance against targets. Data spans from 2020 to present with 1000 customers, 1000 orders, and 1000 sales reps.',
    1
);
/

-- HR business domain
INSERT INTO business_metadata (business_name, business_type, business_description, db_id)
VALUES (
    'Human Resources Department',
    'Workforce and Payroll Management',
    'Responsible for managing employee records across 8 departments (Engineering, Sales, HR, Finance, Marketing, Operations, Legal, Product), tracking daily attendance, and processing monthly payroll. Key business questions include headcount by department, average salary by designation, attendance patterns, late arrivals, payroll disbursement status, and employee retention. Data covers 1000 employees with attendance and payroll records from 2025 onwards.',
    2
);
/

-- Inventory business domain
INSERT INTO business_metadata (business_name, business_type, business_description, db_id)
VALUES (
    'Inventory Department',
    'Supply Chain and Stock Management',
    'Responsible for managing the product catalog across 5 categories (Electronics, Furniture, Stationery, Clothing, Food and Beverages), maintaining supplier relationships across 6 countries, and tracking all stock movements across 6 warehouses in India. Key business questions include current stock levels, supplier ratings, product profitability, movement trends by warehouse, and low stock alerts. Data includes 1000 products, 1000 suppliers, and 1000 stock movement records.',
    3
);
/

COMMIT;
/

-- ── Verification queries ──────────────────────────────────
SELECT 'PROJECT METADATA' AS section, COUNT(*) AS row_count FROM project_metadata
UNION ALL
SELECT 'DATABASE METADATA', COUNT(*) FROM database_metadata
UNION ALL
SELECT 'BUSINESS METADATA', COUNT(*) FROM business_metadata;
/

-- Show the full metadata with joins
SELECT
    p.project_name,
    p.project_type,
    d.db_name,
    d.db_type,
    b.business_name,
    b.business_type
FROM project_metadata p
JOIN database_metadata d ON d.project_id = p.project_id
JOIN business_metadata b ON b.db_id = d.db_id
ORDER BY d.db_id;
/