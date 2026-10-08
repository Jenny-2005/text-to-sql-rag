# test_save.py — save this next to app.py
from app import save_session_to_db, get_oracle_connection

conn = get_oracle_connection()
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM chat_sessions")
print("Row count:", cur.fetchone())
conn.close()

save_session_to_db([
    {"role": "user", "content": "test question"},
    {"role": "assistant", "content": "test answer"}
])

print("Done — check Oracle for a new row now")