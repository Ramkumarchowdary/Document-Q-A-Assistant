"""Excel Q&A Assistant - ask plain-English questions about an uploaded spreadsheet."""

from __future__ import annotations

import hashlib
import os
from io import BytesIO, StringIO

import pandas as pd
import pdfplumber
import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_MODEL = "gemini-3.6-flash"

st.set_page_config(page_title="Excel Q&A Assistant", layout="wide")


def _get_api_key() -> str:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")


# ---------------------------------------------------------------------------
# Sidebar: credentials, model choice, file upload
# ---------------------------------------------------------------------------

_PDF_TABLE_STRATEGIES = [
    {},  # pdfplumber defaults: detect tables via visible grid lines
    {"vertical_strategy": "text", "horizontal_strategy": "text"},  # no visible lines
]


def _extract_pdf_tables(file_bytes: bytes) -> list[tuple[str, pd.DataFrame]]:
    tables = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_tables = []
            for settings in _PDF_TABLE_STRATEGIES:
                raw_tables = page.extract_tables(table_settings=settings) if settings else page.extract_tables()
                if raw_tables:
                    break
            for table_num, raw_table in enumerate(raw_tables, start=1):
                if not raw_table or len(raw_table) < 2:
                    continue
                header, *rows = raw_table
                header = [col if col not in (None, "") else f"col_{i}" for i, col in enumerate(header)]
                rows = [row for row in rows if any(cell not in (None, "") for cell in row)]
                if not rows:
                    continue
                table_df = pd.DataFrame(rows, columns=header)
                tables.append((f"Page {page_num} - Table {table_num}", table_df))
    return tables


def render_sidebar():
    st.sidebar.header("Upload your data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel, CSV, or PDF", type=["xlsx", "xls", "csv", "pdf"]
    )

    df = None
    sheet_name = ""
    file_hash = ""
    file_type = None
    file_bytes = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        name = uploaded_file.name.lower()
        try:
            if name.endswith((".xlsx", ".xls")):
                file_type = "excel"
                excel_file = pd.ExcelFile(BytesIO(file_bytes))
                sheet_names = excel_file.sheet_names
                if len(sheet_names) > 1:
                    sheet_name = st.sidebar.selectbox("Select sheet", sheet_names)
                else:
                    sheet_name = sheet_names[0]
                df = excel_file.parse(sheet_name)
            elif name.endswith(".pdf"):
                file_type = "pdf"
                tables = _extract_pdf_tables(file_bytes)
                if not tables:
                    st.sidebar.warning(
                        "No tables were found in this PDF. Try a PDF with a "
                        "clearly gridded table, or upload Excel/CSV instead."
                    )
                else:
                    labels = [label for label, _ in tables]
                    sheet_name = (
                        st.sidebar.selectbox("Select table", labels)
                        if len(labels) > 1
                        else labels[0]
                    )
                    df = dict(tables)[sheet_name]
            else:
                file_type = "csv"
                df = pd.read_csv(BytesIO(file_bytes))
        except Exception as exc:
            st.sidebar.error(f"Could not read the uploaded file: {exc}")
            df = None

    return df, sheet_name, file_hash, file_type, file_bytes


# ---------------------------------------------------------------------------
# Agent construction (cached per file/key/model)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Setting up the assistant...")
def get_agent(api_key: str, model: str, file_hash: str, sheet_name: str, _df: pd.DataFrame):
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
    return create_pandas_dataframe_agent(
        llm,
        _df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        return_intermediate_steps=True,
    )


def _stringify_output(output) -> str:
    """Some chat models (e.g. Gemini) return structured content blocks
    instead of a plain string; flatten those down to text."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(output)


# ---------------------------------------------------------------------------
# Pulling a supporting table out of the agent's work
# ---------------------------------------------------------------------------

def _find_python_tool(agent):
    tools = getattr(agent, "tools", None) or []
    for tool in tools:
        if getattr(tool, "name", "") in ("python_repl_ast", "python_repl"):
            return tool
    return None


def _try_parse_table_text(text):
    text = text.strip()
    if not text or "\n" not in text:
        return None
    try:
        parsed = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")
    except Exception:
        return None
    if parsed.shape[0] >= 1 and parsed.shape[1] >= 1:
        return parsed
    return None


def extract_table(agent, intermediate_steps):
    tool = _find_python_tool(agent)
    if tool is not None:
        local_vars = getattr(tool, "locals", None) or {}
        candidates = [
            value
            for key, value in local_vars.items()
            if key != "df" and isinstance(value, (pd.DataFrame, pd.Series))
        ]
        if candidates:
            last = candidates[-1]
            return last.to_frame() if isinstance(last, pd.Series) else last

    for _, observation in reversed(intermediate_steps or []):
        table = _try_parse_table_text(str(observation))
        if table is not None:
            return table
    return None


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("Excel Q&A Assistant")
    st.caption("Upload a spreadsheet and ask questions about it in plain English.")

    api_key = _get_api_key()
    model = DEFAULT_MODEL
    df, sheet_name, file_hash, file_type, file_bytes = render_sidebar()

    if file_type == "pdf" and file_bytes is not None:
        st.subheader("Data preview")
        st.pdf(file_bytes, height=500)
    elif df is not None:
        st.subheader("Data preview")
        st.caption(f"Showing up to 50 of {len(df)} rows.")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("Upload an Excel, CSV, or PDF file from the sidebar to get started.")

    st.subheader("Ask a question")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Question about your data")
        ask_clicked = st.form_submit_button("Ask")

    if "history" not in st.session_state:
        st.session_state.history = []

    if ask_clicked:
        if not api_key:
            st.error(
                "This app isn't configured with a Google API key. "
                "The site owner needs to set the GOOGLE_API_KEY "
                "environment variable or Streamlit secret."
            )
        elif df is None:
            st.warning("Upload a file first.")
        elif not question.strip():
            st.warning("Type a question before asking.")
        else:
            try:
                agent = get_agent(api_key, model, file_hash, sheet_name, df)
            except Exception as exc:
                st.error(f"Could not set up the assistant: {exc}")
                agent = None

            if agent is not None:
                with st.spinner("Thinking..."):
                    try:
                        result = agent.invoke({"input": question})
                        answer_text = _stringify_output(result.get("output", ""))
                        steps = result.get("intermediate_steps", [])
                        try:
                            table = extract_table(agent, steps)
                        except Exception:
                            table = None
                        st.session_state.history.insert(
                            0,
                            {"question": question, "answer": answer_text, "table": table},
                        )
                    except Exception as exc:
                        st.error(f"The assistant failed to answer: {exc}")

    if st.session_state.history:
        st.subheader("Chat history")
        for entry in st.session_state.history:
            with st.container(border=True):
                st.markdown(f"**Q: {entry['question']}**")
                st.write(entry["answer"])

                table = entry.get("table")
                if table is not None and not table.empty:
                    st.dataframe(table, use_container_width=True)


if __name__ == "__main__":
    main()
