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
    "Should I invest in HDFC Flexi Cap Fund?",
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
    st.set_page_config(
        page_title="Mutual Fund Facts",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #0d1114;
            --bg-card: #13181b;
            --bg-card-hover: #182025;
            --border-card: #20272d;
            --border-card-hover: #2e3b44;
            --emerald-primary: #00d09c;
            --emerald-dark: #008c69;
            --emerald-bg: rgba(0, 208, 156, 0.06);
            --emerald-border: rgba(0, 208, 156, 0.22);
            --groww-blue: #5367ff;
            --text-primary: #ffffff;
            --text-secondary: #dce3eb;
            --text-muted: #717e8c;
            --text-faint: #5a6672;
            --btn-dark: #242b30;
            --btn-dark-border: #333d45;
            --btn-dark-hover: #2b3339;
        }

        /* Global resets & dark theme background */
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
            background: var(--bg-main) !important;
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Completely remove the sidebar and any sidebar toggle buttons */
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        .stAppDeployButton,
        [data-testid="stStatusWidget"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }

        /* Center content layout */
        .block-container {
            max-width: 780px !important;
            padding-top: 4.5rem !important;
            padding-bottom: 4rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            margin: 0 auto !important;
        }

        /* Center Groww Brand Lockup */
        .center-brand-wrapper {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 1.25rem;
        }

        .center-brand-logo {
            width: 40px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            overflow: hidden;
        }

        .center-brand-name {
            color: #ffffff;
            font-size: 2.15rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1;
        }

        /* Hero Heading */
        .hero-heading {
            color: #ffffff !important;
            font-size: 2.65rem !important;
            font-weight: 700 !important;
            text-align: center !important;
            margin: 0 0 1.5rem 0 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.2 !important;
        }

        /* Facts Disclaimer Banner (100% width matching cards) */
        .facts-banner {
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100% !important;
            box-sizing: border-box !important;
            margin: 0 0 2rem 0 !important;
            padding: 0.9rem 1.25rem !important;
            border: 1px solid var(--emerald-border) !important;
            border-radius: 12px !important;
            background: var(--emerald-bg) !important;
            color: #9ab3a8 !important;
            font-size: 0.92rem !important;
        }

        .facts-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--emerald-primary);
            flex-shrink: 0;
        }

        /* Example questions section label */
        .section-label {
            color: var(--text-muted);
            font-size: 0.84rem;
            font-weight: 500;
            margin: 0 0 0.75rem 0;
            text-align: left;
            width: 100%;
        }

        /* Example Question Buttons (Cards) - No truncation, wrap lines naturally */
        .stApp [data-testid="stAppViewContainer"] .st-key-example_0 button,
        .stApp [data-testid="stAppViewContainer"] .st-key-example_1 button,
        .stApp [data-testid="stAppViewContainer"] .st-key-example_2 button {
            min-height: 94px !important;
            height: 100% !important;
            padding: 1.1rem 1.15rem !important;
            border: 1px solid var(--border-card) !important;
            border-radius: 14px !important;
            background-color: var(--bg-card) !important;
            color: var(--text-secondary) !important;
            box-shadow: none !important;
            text-align: left !important;
            display: flex !important;
            align-items: flex-start !important;
            justify-content: flex-start !important;
            transition: border-color 0.15s ease, background-color 0.15s ease !important;
        }

        .st-key-example_0 button:hover,
        .st-key-example_1 button:hover,
        .st-key-example_2 button:hover {
            border-color: var(--border-card-hover) !important;
            background-color: var(--bg-card-hover) !important;
        }

        /* Force wrapping across all child containers inside example buttons */
        .st-key-example_0 button *,
        .st-key-example_1 button *,
        .st-key-example_2 button * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: normal !important;
            overflow-wrap: break-word !important;
        }

        .st-key-example_0 button p,
        .st-key-example_1 button p,
        .st-key-example_2 button p {
            color: var(--text-secondary) !important;
            font-size: 0.92rem !important;
            line-height: 1.42 !important;
            text-align: left !important;
            font-weight: 400 !important;
            margin: 0 !important;
        }

        .st-key-example_0 button:hover p,
        .st-key-example_1 button:hover p,
        .st-key-example_2 button:hover p {
            color: #ffffff !important;
        }

        /* Input Form (100% width matching cards) */
        [data-testid="stForm"] {
            border: 1px solid var(--border-card) !important;
            border-radius: 16px !important;
            background: var(--bg-card) !important;
            padding: 6px 8px 6px 14px !important;
            box-shadow: none !important;
            margin-top: 1.5rem !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stTextInputRootElement"] div[data-baseweb="input"] {
            min-height: 48px !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        [data-testid="stTextInputRootElement"] input {
            min-height: 48px !important;
            padding: 0 !important;
            color: #f0f6fc !important;
            background: transparent !important;
            font-size: 0.95rem !important;
            caret-color: #00d09c !important;
        }

        [data-testid="stTextInputRootElement"] input::placeholder {
            color: var(--text-faint) !important;
            opacity: 1 !important;
        }

        /* Submit (Ask) Button */
        .st-key-ask_button button {
            min-height: 44px !important;
            border: 1px solid var(--btn-dark-border) !important;
            border-radius: 10px !important;
            background: var(--btn-dark) !important;
            color: #a9b5c0 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 6px !important;
            padding: 0 14px !important;
            transition: all 0.15s ease !important;
            box-shadow: none !important;
        }

        .st-key-ask_button button:hover {
            background: var(--btn-dark-hover) !important;
            border-color: #414f59 !important;
            color: #ffffff !important;
        }

        .st-key-ask_button button p {
            color: #a9b5c0 !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            margin: 0 !important;
        }

        .st-key-ask_button button:hover p {
            color: #ffffff !important;
        }

        .st-key-ask_button button [data-testid="stIconMaterial"] {
            color: #8a96a1 !important;
            font-size: 17px !important;
        }

        .st-key-ask_button button:hover [data-testid="stIconMaterial"] {
            color: #ffffff !important;
        }

        /* Response Container (100% width matching input) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            width: 100% !important;
            box-sizing: border-box !important;
            margin-top: 1.5rem !important;
            border: 1px solid var(--border-card) !important;
            border-radius: 16px !important;
            background: var(--bg-card) !important;
            padding: 1.35rem 1.5rem !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25) !important;
        }

        .answer-label {
            margin-bottom: 0.75rem;
            color: var(--emerald-primary);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }

        [data-testid="stLinkButton"] a {
            min-height: 40px !important;
            border: 1px solid #165b45 !important;
            border-radius: 8px !important;
            background: #0c2b21 !important;
            color: var(--emerald-primary) !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            transition: all 0.15s ease !important;
        }

        [data-testid="stLinkButton"] a:hover {
            border-color: var(--emerald-primary) !important;
            background: #11382b !important;
            color: #34d399 !important;
        }

        [data-testid="stLinkButton"] a p {
            color: var(--emerald-primary) !important;
        }

        [data-testid="stLinkButton"] a:hover p {
            color: #34d399 !important;
        }

        [data-testid="stCaptionContainer"] p {
            margin-top: 0.75rem !important;
            color: var(--text-muted) !important;
            font-size: 0.84rem !important;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 2rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            .hero-heading {
                font-size: 2rem !important;
            }
            .center-brand-name {
                font-size: 1.75rem !important;
            }
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

    # Centered Groww Logo Lockup above the title (cropped horizontally into a semicircle)
    st.markdown(
        """
        <div class="center-brand-wrapper">
            <div class="center-brand-logo">
                <svg width="40" height="20" viewBox="0 0 100 50" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <clipPath id="groww-semicircle-clip">
                        <path d="M 0 50 A 50 50 0 0 1 100 50 Z"/>
                    </clipPath>
                    <g clip-path="url(#groww-semicircle-clip)">
                        <rect width="100" height="50" fill="#5367FF"/>
                        <path d="M-5 55 L-5 32 L40 25 L60 30 L105 18 L105 55 Z" fill="#00D09C"/>
                    </g>
                </svg>
            </div>
            <span class="center-brand-name">Groww</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Title
    st.markdown(
        '<h1 class="hero-heading">Welcome to Mutual Fund Facts</h1>',
        unsafe_allow_html=True,
    )

    # Facts Disclaimer Banner
    st.markdown(
        '<div class="facts-banner">'
        '<div class="facts-icon">'
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"></circle>'
        '<line x1="12" y1="16" x2="12" y2="12"></line>'
        '<line x1="12" y1="8" x2="12.01" y2="8"></line>'
        '</svg>'
        '</div>'
        f"<span>{FACTS_ONLY_DISCLAIMER}</span></div>",
        unsafe_allow_html=True,
    )

    # Example Questions
    st.markdown('<div class="section-label">Example questions</div>', unsafe_allow_html=True)
    columns = st.columns(3)
    for index, question_text in enumerate(EXAMPLE_QUESTIONS):
        columns[index].button(
            question_text,
            key=f"example_{index}",
            use_container_width=True,
            on_click=_select_example,
            args=(question_text,),
        )

    # Input Form
    with st.form("question_form"):
        input_column, send_column = st.columns([6, 1])
        question = input_column.text_input(
            "Question",
            key="question_input",
            label_visibility="collapsed",
            placeholder="Ask a factual mutual fund question...",
        )
        submitted = send_column.form_submit_button(
            "Ask",
            key="ask_button",
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