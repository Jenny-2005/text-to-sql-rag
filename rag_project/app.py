import os
import re
from decimal import Decimal

import pandas as pd
import streamlit as st
import plotly.express as px
from groq import Groq
from sentence_transformers import SentenceTransformer

from input_layer import process_input
from schema_index import retrieve_schema, MODEL_NAME
from sql_generation import generate_and_run

# CHART_KEYWORDS = [
#     "chart", "graph", "plot", "visualize", "visualization",
#     "bar chart", "line chart", "pie chart", "show me a graph",
#     "bar graph", "trend", "compare visually",
# ]
import json
import oracledb
ID_LIKE_PATTERN = re.compile(r"(^id$|_id$|^year$|_year$)", re.IGNORECASE)
def get_oracle_connection():
    oracle_dsn = oracledb.makedsn("localhost", 1521, service_name="XEPDB1")
    return oracledb.connect(user="system", password="root", dsn=oracle_dsn)

def save_session_to_db(messages):
    """Save current session to Oracle chat_sessions table."""
    if not messages:
        return
    try:
        # Extract metadata from messages
        user_messages = [m for m in messages if m["role"] == "user"]
        title = user_messages[0]["content"][:500] if user_messages else "Untitled"
        question_count = len(user_messages)
        databases_used = list(set(
            m.get("database", "") for m in messages 
            if m.get("database")
        ))
        
        # Serialize messages to JSON
        messages_json = json.dumps(messages, default=str)
        
        conn = get_oracle_connection()
        cursor = conn.cursor()
        
        # Create CLOB variable for messages
        clob_var = cursor.var(oracledb.DB_TYPE_CLOB)
        clob_var.setvalue(0, messages_json)
        
        cursor.execute("""
            INSERT INTO chat_sessions 
                (session_title, ended_at, question_count, 
                 databases_used, messages)
            VALUES 
                (:1, SYSDATE, :2, :3, :4)
        """, [
            title,
            question_count,
            ", ".join(databases_used),
            clob_var
        ])
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        st.warning(f"Could not save session: {e}")

def load_sessions_from_db():
    """Load all past sessions from Oracle."""
    conn = None
    try:
        conn = get_oracle_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, session_title, started_at, 
                   question_count, databases_used, messages
            FROM chat_sessions
            ORDER BY started_at DESC
            FETCH FIRST 20 ROWS ONLY
        """)
        rows = cursor.fetchall()

        sessions = []
        for row in rows:
            session_id, title, started_at, q_count, dbs, messages_clob = row
            messages = json.loads(messages_clob.read() if messages_clob else "[]")
            sessions.append({
                "session_id": session_id,
                "title": title,
                "started_at": started_at,
                "question_count": q_count,
                "databases_used": dbs,
                "messages": messages
            })

        cursor.close()   # moved here, after all CLOBs are read
        conn.close()
        print(f"DEBUG: built {len(sessions)} session dicts")
        return sessions

    except Exception as e:
        st.warning(f"Could not load sessions: {e}")
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


model = load_model()
st.title("Text-to-Query RAG System")


def _word_in(keywords, text):
    """Word-boundary-safe keyword match, tolerant of simple plurals
    (e.g. 'chart' matches both 'chart' and 'charts')."""
    return any(re.search(rf"\b{re.escape(kw)}s?\b", text) for kw in keywords)


# def wants_chart(question: str) -> bool:
#     return _word_in(CHART_KEYWORDS, question.lower())


def summarize_results(question, df, database):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # If too many rows, pre-aggregate to avoid huge prompt
    if len(df) > 50:
        data_text = df.head(50).to_string(index=False)
        data_text += f"\n... and {len(df) - 50} more rows"
    else:
        data_text = df.to_string(index=False)

    prompt = f"""You are a helpful business analyst.
A user asked: "{question}"

The following data was retrieved from the {database} database:
{data_text}

Write a clear, concise 3-4 sentence summary answering the user's question
based on this data. Be specific — mention actual numbers, names, or values
from the data. Don't say "the data shows" — just answer directly."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


def decide_chart_type(df, question, numeric_cols, text_cols):
    q = question.lower()

    # Explicit user request wins
    if _word_in(["pie", "pie chart"], q):
        return "pie"
    if _word_in(["line", "trend", "over time", "monthly", "daily", "yearly"], q):
        return "line"
    if _word_in(["scatter", "correlation", "vs", "against"], q) and len(numeric_cols) >= 2:
        return "scatter"
    if _word_in(["bar", "bar chart", "compare"], q):
        return "bar"

    # Data shape decides if user didn't specify
    if text_cols and numeric_cols:
        unique_categories = df[text_cols[0]].nunique()
        return "pie" if unique_categories <= 6 else "bar"

    if len(numeric_cols) >= 2 and not text_cols:
        return "scatter"

    return "bar"


def _pick_y_col(numeric_cols):
    """Prefer a real metric column over ID/year-like numeric columns."""
    candidates = [c for c in numeric_cols if not ID_LIKE_PATTERN.search(c)]
    pool = candidates if candidates else numeric_cols
    return pool[-1]


def _aggregate_for_categorical_chart(df, x_col, y_col):
    """Sum y_col per x_col so duplicate categories don't distort pie/bar charts."""
    return df.groupby(x_col, as_index=False)[y_col].sum()


def build_chart(df, question):
    if df.empty:
        st.info("No data returned for this query.")
        return

    # Convert Decimal columns to float (SQLAlchemy returns Decimal for NUMERIC types)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, Decimal)).any():
            df[col] = df[col].astype(float)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()

    if not numeric_cols:
        if text_cols:
            # No numeric column to plot — fall back to a count-based pie chart
            cat_col = text_cols[0]
            counts_df = df[cat_col].value_counts().reset_index()
            counts_df.columns = [cat_col, "count"]

            title = question if len(question) <= 80 else question[:77] + "..."
            try:
                if counts_df[cat_col].nunique() <= 8:
                    fig = px.pie(counts_df, names=cat_col, values="count", title=title)
                else:
                    # Too many categories for a readable pie chart — bar is clearer
                    fig = px.bar(counts_df, x=cat_col, y="count", title=title)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Couldn't render chart ({e}); showing raw data instead.")
                st.dataframe(df)
        else:
            st.warning("No columns found to chart.")
            st.dataframe(df)
        return

    x_col = text_cols[0] if text_cols else df.columns[0]
    y_col = _pick_y_col(numeric_cols)

    # If the data has separate year/month columns (common for monthly trend
    # queries), combine them into one sortable chronological x-axis instead of
    # defaulting to just "year", which collapses every month in a year onto
    # the same x value and connects rows in arbitrary order.
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    month_col = next((c for c in df.columns if c.lower() == "month"), None)

    if year_col and month_col and x_col == year_col:
        # Cast to int first -- these may have come through as floats
        # (2025.0, 7.0) from the Decimal-to-float conversion above.
        year_int = df[year_col].astype(float).astype(int).astype(str)
        month_int = df[month_col].astype(float).astype(int).astype(str).str.zfill(2)
        df["period"] = pd.to_datetime(year_int + "-" + month_int + "-01")
        df = df.sort_values("period")
        x_col = "period"
    else:
        df[x_col] = df[x_col].astype(str)

    chart_type = decide_chart_type(df, question, numeric_cols, text_cols)
    title = question if len(question) <= 80 else question[:77] + "..."

    try:
        if chart_type == "line":
            group_candidates = [
                c for c in text_cols + [c for c in df.columns if any(
                    kw in c.lower() for kw in ("year", "category", "type")
                )]
                if c != x_col  # avoid grouping by the same column used for x-axis
            ]

            if group_candidates:
                group_col = group_candidates[0]
                df[group_col] = df[group_col].astype(str)
                fig = px.line(
                    df, x=x_col, y=y_col, color=group_col, title=title,
                    labels={x_col: x_col.replace("_", " ").title(),
                            y_col: y_col.replace("_", " ").title()},
                )
            else:
                fig = px.line(df, x=x_col, y=y_col, title=title)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "pie":
            plot_df = _aggregate_for_categorical_chart(df, x_col, y_col)
            if (plot_df[y_col] < 0).any():
                st.warning("Data contains negative values; showing a bar chart instead of a pie chart.")
                fig = px.bar(plot_df, x=x_col, y=y_col, title=title)
            else:
                fig = px.pie(plot_df, names=x_col, values=y_col, title=title)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "scatter":
            if len(numeric_cols) < 2:
                st.warning("Not enough numeric columns for a scatter plot; showing a bar chart instead.")
                plot_df = _aggregate_for_categorical_chart(df, x_col, y_col)
                fig = px.bar(plot_df, x=x_col, y=y_col, title=title)
            else:
                x_num, y_num = numeric_cols[0], numeric_cols[1]
                fig = px.scatter(df, x=x_num, y=y_num, title=title)
            st.plotly_chart(fig, use_container_width=True)

        else:  # bar
            plot_df = _aggregate_for_categorical_chart(df, x_col, y_col)
            fig = px.bar(plot_df, x=x_col, y=y_col, title=title)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Couldn't render chart ({e}); showing raw data instead.")
        st.dataframe(df)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "past_sessions" not in st.session_state:
    st.session_state.past_sessions = load_sessions_from_db()

if "session_started" not in st.session_state:
    st.session_state.session_started = pd.Timestamp.now()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("data"):
            hist_df = pd.DataFrame(message["data"], columns=message.get("columns"))
            build_chart(hist_df, message.get("question", ""))
            with st.expander(f"View raw data ({len(hist_df)} rows)"):
                st.dataframe(hist_df)

question = st.chat_input("Ask a question about your data:")

if question:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)
    
    # Add to history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    
    # Run your pipeline and show response
    with st.chat_message("assistant"):
        # Layer 1 check
        layer1_result = process_input(question)
        if layer1_result["status"] == "rejected":
            response = f"Sorry, I can't answer that: {layer1_result['reason']}"
            st.error(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
        else:
            with st.spinner("Querying your databases..."):
                try:
                    rows, columns, database = generate_and_run(question)
                    df = pd.DataFrame(rows, columns=list(columns))
                    # Text summary
                    try:
                        summary = summarize_results(question, df, database)
                        st.write(summary)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": summary,
                            "database": database,
                            "question": question,
                            "data": df.to_dict(orient="records"),   # <-- store the rows
                            "columns": list(df.columns)  
                        })
                    except Exception as e:
                        st.warning(f"Summary failed: {e}")
                        
                    build_chart(df, question)

                    with st.expander(f"View raw data ({len(rows)} rows)"):
                        st.dataframe(df)
                            
                except Exception as e:
                    st.error(f"Error: {e}")

with st.sidebar:
    st.header("Connected Databases")
    st.write("Postgres — Sales")
    st.write("MySQL — HR")
    st.write("Oracle — Inventory")
    
    st.divider()
    
    if st.button("+ New Chat"):
        if st.session_state.messages:
            save_session_to_db(st.session_state.messages)
            st.session_state.past_sessions = load_sessions_from_db()
        st.session_state.messages = []
        st.session_state.session_start = pd.Timestamp.now()
        st.rerun()
    
    st.subheader("Previous Chats")
    
    if st.session_state.past_sessions:
        for session in st.session_state.past_sessions:
            label = f"💬 {session['title'][:35]}"
            caption = f"{session['started_at']} · {session['question_count']} questions"
            
            with st.expander(label):
                st.caption(caption)
                st.caption(f"Databases: {session['databases_used']}")
                if st.button("Load", key=f"load_{session['session_id']}"):
                    st.session_state.messages = session["messages"].copy()
                    st.rerun()
    else:
        st.caption("No previous sessions.")
    
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()