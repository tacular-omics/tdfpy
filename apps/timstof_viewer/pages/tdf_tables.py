"""TDF tables page — browse any table in analysis.tdf and run ad-hoc SQL.

The `.tdf` file is a SQLite database; this page surfaces every table plus a
read-only `SELECT` box for arbitrary queries.

All inputs live in the sidebar; the main area shows the resulting data only.
"""

from __future__ import annotations

import streamlit as st

from _shared import load_table, require_analysis_dir, run_sql, table_names

st.title("TDF tables (SQL)")

analysis_dir = require_analysis_dir()
names = table_names(analysis_dir)
if not names:
    st.warning("No tables found in `analysis.tdf`.")
    st.stop()


# -- Sidebar: mode + inputs -------------------------------------------------

with st.sidebar:
    st.header("Source")
    mode = st.radio("Mode", ["Browse table", "SQL query"])
    if mode == "Browse table":
        name = st.selectbox("Table", options=sorted(names))
    else:
        st.caption("Read-only `SELECT` / `WITH` against `analysis.tdf`.")
        query = st.text_area(
            "SQL query", value="SELECT * FROM Frames LIMIT 100", height=160)
        run = st.button("Run query", type="primary")
        with st.expander("Available tables"):
            st.write(", ".join(f"`{n}`" for n in sorted(names)))


# ===========================================================================
# Main: data only
# ===========================================================================

if mode == "Browse table":
    try:
        df = load_table(analysis_dir, name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load table `{name}`: {exc}")
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    st.dataframe(df, use_container_width=True, height=560)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}.csv",
        mime="text/csv",
    )

else:  # SQL query
    if not run:
        st.info("Enter a query in the sidebar and press **Run query**.")
        st.stop()
    stripped = query.strip().rstrip(";").strip()
    if not stripped.lower().startswith(("select", "with")):
        st.error("Only read-only `SELECT` / `WITH` queries are allowed.")
        st.stop()
    try:
        result = run_sql(analysis_dir, stripped)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Query failed: {exc}")
        st.stop()
    st.success(f"{len(result):,} rows")
    st.dataframe(result, use_container_width=True, height=520)
    st.download_button(
        "Download CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="query_result.csv",
        mime="text/csv",
    )
