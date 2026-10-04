from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import settings
from src.guardrails.refusal_templates import FACTS_ONLY_DISCLAIMER
from src.llm.groq_client import GroqRequestError
from src.retrieval.query_pipeline import QueryResponse, answer_question
from src.vectorstore.index_manager import ensure_index


EXAMPLE_QUESTIONS = (
    "What is the exit load for HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "How can I download a consolidated account statement?",
)


@st.cache_resource(show_spinner="Preparing the local knowledge index...")
def _prepare_index(persist_directory: str) -> dict[str, object]:
    return ensure_index(persist_directory=persist_directory)


def _select_example(question: str) -> None:
    st.session_state["question_input"] = question


def _display_response(response: QueryResponse) -> None:
    with st.container(border=True):
        st.markdown('<div class="answer-label">ANSWER</div>', unsafe_allow_html=True)
        if response.answer:
            st.write(response.answer)
        if response.source_url and response.source_date:
            st.link_button(
                "Open official source",
                response.source_url,
                key="source_link",
                icon=":material/open_in_new:",
            )
            st.caption(f"Last updated from sources: {response.source_date}")
        elif response.decision.allowed:
            st.info("The available approved sources could not verify this answer.")


def main() -> None:
    st.set_page_config(page_title="Mutual Fund Facts", layout="centered")
    st.markdown(
        """
        <style>
        :root {
            --groww-green: #00b386;
            --groww-green-dark: #008c69;
            --ink: #18332c;
            --muted: #61726c;
            --line: #e2ebe7;
            --soft-green: #effaf6;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] { background: #ffffff !important; }
        .block-container { max-width: 880px; padding: 1.4rem 2rem 4rem; }
        h1, h2, h3, p, label { color: var(--ink); }
        .hero-heading { color: var(--ink) !important; }
        .hero-heading [data-heading-text] > span { color: var(--groww-green-dark) !important; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"] p { color: var(--ink) !important; }
        .brand-lockup { display: flex; align-items: center; min-height: 42px; }
        .brand-wordmark { color: var(--groww-green); font-size: 1.65rem; font-weight: 800; line-height: 1; }
        .brand-divider { width: 1px; height: 22px; margin: 0 0.7rem; background: var(--line); }
        .brand-context { color: var(--muted); font-size: 0.9rem; font-weight: 600; }
        .hero-heading {
            margin: 3.3rem 0 1.8rem;
            color: var(--ink);
            font-size: 2.75rem;
            font-weight: 750;
            line-height: 1.15;
            text-align: center;
        }
        .hero-heading span { color: var(--groww-green-dark); }
        .facts-banner {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            max-width: 690px;
            margin: 0 auto 2.4rem;
            padding: 1rem 1.2rem;
            border: 1px solid #d8f0e6;
            border-radius: 14px;
            background: var(--soft-green);
            color: var(--ink);
            font-size: 0.98rem;
        }
        .facts-icon {
            display: inline-flex;
            flex: 0 0 24px;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #d9f5e9;
            color: var(--groww-green-dark);
            font-weight: 800;
        }
        .section-heading { margin: 0 0 0.75rem; color: var(--ink); font-size: 1rem; font-weight: 700; }
        .stApp [data-testid="stAppViewContainer"] .st-key-example_0 button,
        .stApp [data-testid="stAppViewContainer"] .st-key-example_1 button,
        .stApp [data-testid="stAppViewContainer"] .st-key-example_2 button {
            min-height: 72px;
            padding: 0.75rem 1rem;
            border: 1px solid var(--line) !important;
            border-radius: 16px;
            background-color: #ffffff !important;
            color: var(--ink) !important;
            box-shadow: 0 2px 8px rgba(24, 51, 44, 0.035);
            transition: border-color 120ms ease, background-color 120ms ease;
        }
        .stApp [data-testid="stAppViewContainer"] .st-key-example_0 button:hover,
        .stApp [data-testid="stAppViewContainer"] .st-key-example_1 button:hover,
        .stApp [data-testid="stAppViewContainer"] .st-key-example_2 button:hover {
            border-color: #9cddc7 !important;
            background-color: #f7fcfa !important;
            color: var(--ink) !important;
        }
        .st-key-example_0 button p,
        .st-key-example_1 button p,
        .st-key-example_2 button p { color: var(--ink) !important; font-size: 0.9rem; line-height: 1.35; }
        [data-testid="stTextInputRootElement"] > div,
[data-testid="stTextInputRootElement"] div[data-baseweb="input"] {
    min-height: 58px;
    border: 1px solid #d7e3de !important;
    border-radius: 18px;
    background: #f1f4f3 !important;
    box-shadow: 0 2px 10px rgba(24, 51, 44, 0.04);
}

[data-testid="stTextInputRootElement"] input {
    min-height: 54px;
    padding: 0.2rem 1rem;
    color: #18332c !important;
    background: #f1f4f3 !important;
    caret-color: #18332c !important;
}

[data-testid="stTextInputRootElement"] input::placeholder {
    color: #71817c !important;
    opacity: 1 !important;
}

[data-testid="stTextInputRootElement"] > div:focus-within {
    border-color: var(--groww-green) !important;
    box-shadow: 0 0 0 1px var(--groww-green) !important;
}
        [data-testid="stTextInputRootElement"] > div:focus-within {
            border-color: var(--groww-green);
            box-shadow: 0 0 0 1px var(--groww-green);
        }
        .st-key-ask_button button {
            min-height: 58px;
            border: 1px solid var(--groww-green);
            border-radius: 18px;
            background: var(--groww-green);
            color: #ffffff;
            font-weight: 700;
        }
        .st-key-ask_button button:hover { border-color: var(--groww-green-dark); background: var(--groww-green-dark); color: #ffffff; }
        .st-key-ask_button button p { color: #ffffff !important; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            margin-top: 2rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 5px 20px rgba(24, 51, 44, 0.045);
        }
        .answer-label { margin-bottom: 0.5rem; color: var(--groww-green-dark); font-size: 0.76rem; font-weight: 800; }
        [data-testid="stLinkButton"] a,
        [data-testid="stLinkButton"] p { color: #ffffff !important; }
        [data-testid="stLinkButton"] a {
            min-height: 46px;
            border: 1px solid var(--groww-green);
            border-radius: 999px;
            background: var(--groww-green);
            color: #ffffff;
        }
        [data-testid="stLinkButton"] a:hover { border-color: var(--groww-green-dark); background: var(--groww-green-dark); }
        [data-testid="stCaptionContainer"] p { margin-top: 0.75rem; color: var(--muted) !important; font-size: 0.86rem; }
        @media (max-width: 640px) {
            .block-container { padding: 1rem 1rem 3rem; }
            .hero-heading { margin: 2.6rem 0 1.4rem; font-size: 2rem; }
            .facts-banner { margin-bottom: 1.8rem; padding: 0.9rem 1rem; }
            [data-testid="stHorizontalBlock"] { gap: 0.55rem; }
            .st-key-example_0 button,
            .st-key-example_1 button,
            .st-key-example_2 button { min-height: 76px; padding: 0.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        _prepare_index(str(settings.CHROMA_DB_PATH))
    except Exception as exc:
        st.error(f"Unable to prepare the local knowledge index: {exc}")
        return

    st.markdown(
        '<div class="brand-lockup"><span class="brand-wordmark">groww</span>'
        '<span class="brand-divider"></span><span class="brand-context">Mutual Fund Facts</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 class="hero-heading">Welcome to <span>Mutual Fund Facts</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="facts-banner"><span class="facts-icon">i</span>'
        f"<span>{FACTS_ONLY_DISCLAIMER}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Example questions</div>', unsafe_allow_html=True)
    columns = st.columns(3)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        columns[index].button(
            question,
            key=f"example_{index}",
            use_container_width=True,
            on_click=_select_example,
            args=(question,),
        )

    with st.form("question_form"):
        input_column, send_column = st.columns([6, 1])
        question = input_column.text_input(
            "Question",
            key="question_input",
            label_visibility="collapsed",
            placeholder="Ask a factual mutual fund question",
        )
        submitted = send_column.form_submit_button(
            "Ask",
            key="ask_button",
            type="primary",
            icon=":material/send:",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Checking approved sources..."):
                    response = answer_question(question.strip())
                _display_response(response)
            except GroqRequestError as exc:
                st.error(str(exc))
            except Exception:
                st.error("The question could not be answered. Check the local index and Groq configuration, then try again.")


if __name__ == "__main__":
    main()