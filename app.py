


"""Excel Q&A Assistant - ask plain-English questions about an uploaded spreadsheet."""

from __future__ import annotations

import hashlib
from io import BytesIO, StringIO

import pandas as pd
import pdfplumber
import streamlit as st
from fpdf import FPDF
from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents import create_pandas_dataframe_agent

MODEL_OPTIONS = ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"]

st.set_page_config(page_title="Excel Q&A Assistant", layout="wide")


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
                table_df = pd.DataFrame(rows, columns=header)
                tables.append((f"Page {page_num} - Table {table_num}", table_df))
    return tables


def render_sidebar():
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Anthropic API Key", type="password")
    model = st.sidebar.selectbox("Model", MODEL_OPTIONS)
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel, CSV, or PDF", type=["xlsx", "xls", "csv", "pdf"]
    )

    df = None
    sheet_name = ""
    file_hash = ""

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        name = uploaded_file.name.lower()
        try:
            if name.endswith((".xlsx", ".xls")):
                excel_file = pd.ExcelFile(BytesIO(file_bytes))
                sheet_names = excel_file.sheet_names
                if len(sheet_names) > 1:
                    sheet_name = st.sidebar.selectbox("Select sheet", sheet_names)
                else:
                    sheet_name = sheet_names[0]
                df = excel_file.parse(sheet_name)
            elif name.endswith(".pdf"):
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
                df = pd.read_csv(BytesIO(file_bytes))
        except Exception as exc:
            st.sidebar.error(f"Could not read the uploaded file: {exc}")
            df = None

    return api_key, model, df, sheet_name, file_hash


# ---------------------------------------------------------------------------
# Agent construction (cached per file/key/model)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Setting up the assistant...")
def get_agent(api_key: str, model: str, file_hash: str, sheet_name: str, _df: pd.DataFrame):
    llm = ChatAnthropic(model=model, anthropic_api_key=api_key, temperature=0)
    return create_pandas_dataframe_agent(
        llm,
        _df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        return_intermediate_steps=True,
    )


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
# Export helpers
# ---------------------------------------------------------------------------

def _safe_text(text: str) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


def build_pdf(question: str, answer: str, table: pd.DataFrame | None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 10, "Excel Q&A Assistant", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 7, "Question", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, _safe_text(question), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 7, "Answer", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, _safe_text(answer), new_x="LMARGIN", new_y="NEXT")

    if table is not None and not table.empty:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Supporting data (up to 30 rows)", new_x="LMARGIN", new_y="NEXT")

        subset = table.head(30)
        columns = list(subset.columns)
        col_width = 190 / max(len(columns), 1)

        pdf.set_font("Helvetica", "B", 8)
        for col in columns:
            pdf.cell(col_width, 6, _safe_text(col)[:20], border=1)
        pdf.ln()

        pdf.set_font("Helvetica", size=8)
        for _, row in subset.iterrows():
            for value in row:
                pdf.cell(col_width, 6, _safe_text(value)[:20], border=1)
            pdf.ln()

    return bytes(pdf.output())


def build_excel(question: str, answer: str, table: pd.DataFrame | None) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        answer_df = pd.DataFrame({"Question": [question], "Answer": [answer]})
        answer_df.to_excel(writer, sheet_name="Answer", index=False)
        if table is not None and not table.empty:
            table.to_excel(writer, sheet_name="Data", index=False)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("Excel Q&A Assistant")
    st.caption("Upload a spreadsheet and ask questions about it in plain English.")

    api_key, model, df, sheet_name, file_hash = render_sidebar()

    if df is not None:
        st.subheader("Data preview")
        st.caption(f"Showing up to 50 of {len(df)} rows.")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("Upload an Excel or CSV file from the sidebar to get started.")

    st.subheader("Ask a question")
    question = st.text_input("Question about your data")
    ask_clicked = st.button("Ask")

    if "history" not in st.session_state:
        st.session_state.history = []

    if ask_clicked:
        if not api_key:
            st.warning("Enter your Anthropic API key in the sidebar.")
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
                        answer_text = result.get("output", "")
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
        for idx, entry in enumerate(st.session_state.history):
            with st.container(border=True):
                st.markdown(f"**Q: {entry['question']}**")
                st.write(entry["answer"])

                table = entry.get("table")
                if table is not None and not table.empty:
                    st.dataframe(table, use_container_width=True)

                pdf_bytes = build_pdf(entry["question"], entry["answer"], table)
                excel_bytes = build_excel(entry["question"], entry["answer"], table)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download as PDF",
                        data=pdf_bytes,
                        file_name=f"answer_{idx}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{idx}",
                    )
                with col2:
                    st.download_button(
                        "Download as Excel",
                        data=excel_bytes,
                        file_name=f"answer_{idx}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"xlsx_{idx}",
                    )


if __name__ == "__main__":
    main()
