import sqlglot
from sqlglot import exp

sql = ("WITH regional_sales AS (SELECT region, SUM(amount) AS total FROM orders GROUP BY region) "
       "SELECT * FROM regional_sales WHERE total > 1000")
parsed = sqlglot.parse_one(sql, dialect="postgres")

for cte in parsed.find_all(exp.CTE):
    print("CTE alias_or_name:", repr(cte.alias_or_name))

for t in parsed.find_all(exp.Table):
    print("Table node:", repr(t.name))