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
from src.vectorstore.index_manager import compute_index_fingerprint, ensure_index


EXAMPLE_QUESTIONS = (
    "What is the exit load for HDFC Flexi Cap Fund?",
    "Where can I find the statutory SID and KIM for HDFC schemes?",
    "Compare Direct vs Regular TER for HDFC Top 100 Fund",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
)

RECENT_INQUIRIES = (
    ("Parag Parikh Flexi Cap Exit L...", "What is the exit load for HDFC Flexi Cap Fund?"),
    ("Mirae Asset ELSS 3Y Lock-in", "What is the lock-in period for HDFC ELSS Tax Saver Fund?"),
    ("HDFC Mid-Cap Direct vs Re...", "Compare Direct vs Regular TER for HDFC Mid-Cap Fund"),
    ("Quant Small Cap Portfolio ...", "What is the portfolio disclosure frequency in SEBI SID guidelines?"),
)


@st.cache_resource(show_spinner="Preparing the local knowledge index...")
def _prepare_index(persist_directory: str, fingerprint: str | None = None) -> dict[str, object]:
    return ensure_index(persist_directory=persist_directory)


def _select_example(question: str) -> None:
    st.session_state["question_input"] = question
    st.session_state["auto_submit"] = question


def _display_response(response: QueryResponse, question_text: str = "") -> None:
    with st.container():
        st.markdown(
            f"""
            <div class="response-card">
                <div class="response-header">
                    <div class="response-tag">
                        <span class="response-dot"></span> FACTS AI VERIFIED ANSWER
                    </div>
                </div>
                <div class="response-body">
            """,
            unsafe_allow_html=True,
        )
        if question_text:
            st.markdown(f'<div class="query-bubble">{question_text}</div>', unsafe_allow_html=True)

        if response.answer:
            st.markdown(f'<div class="answer-content">{response.answer}</div>', unsafe_allow_html=True)

        if response.source_url and response.source_date:
            st.markdown('<div class="source-action-row">', unsafe_allow_html=True)
            st.link_button(
                "Open official source",
                response.source_url,
                key="source_link",
                icon=":material/open_in_new:",
            )
            st.markdown(
                f'<span class="source-date">Last updated from sources: {response.source_date}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif response.decision.allowed:
            st.info("The available approved sources could not verify this answer.")

        st.markdown(
            f'<div class="response-disclaimer">{FACTS_ONLY_DISCLAIMER}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Groww • FACTS AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Initialize session state keys
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "active_response" not in st.session_state:
        st.session_state["active_response"] = None
    if "last_asked_question" not in st.session_state:
        st.session_state["last_asked_question"] = ""
    if "auto_submit" not in st.session_state:
        st.session_state["auto_submit"] = None

    # Custom styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-page: #F8FAFC;
            --bg-sidebar: #FFFFFF;
            --bg-card: #FFFFFF;
            --groww-emerald: #00D09C;
            --groww-emerald-dark: #00875A;
            --groww-emerald-subtle: #E6FBF5;
            --groww-slate: #1E2229;
            --text-primary: #1E2229;
            --text-secondary: #64748B;
            --text-muted: #94A3B8;
            --border-light: #EAECF0;
            --border-hover: #00D09C;
            --shadow-subtle: 0 4px 20px rgba(0, 0, 0, 0.04);
            --shadow-card: 0 2px 10px rgba(30, 34, 41, 0.04);
        }

        /* Global Canvas resets */
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
            background-color: var(--bg-page) !important;
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        [data-testid="stHeader"] {
            display: none !important;
        }

        /* Remove default streamlit paddings & decorations */
        .block-container {
            max-width: 1360px !important;
            padding-top: 1.25rem !important;
            padding-bottom: 3.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

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

        /* Top Navigation Bar */
        .top-navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0 24px 0;
            border-bottom: 1px solid var(--border-light);
            margin-bottom: 2rem;
            width: 100%;
        }

        .nav-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .groww-brand {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 22px;
            font-weight: 800;
            color: var(--groww-slate);
            text-decoration: none;
            letter-spacing: -0.03em;
        }

        .facts-tag {
            background: var(--groww-emerald-subtle);
            color: var(--groww-emerald-dark);
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            letter-spacing: 0.04em;
        }

        .facts-tag-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--groww-emerald);
        }

        .nav-center {
            display: flex;
            align-items: center;
            gap: 28px;
        }

        .nav-link {
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
            text-decoration: none;
            transition: color 0.15s ease;
        }

        .nav-link.active {
            color: var(--groww-emerald-dark);
            font-weight: 600;
        }

        .nav-right {
            display: flex;
            align-items: center;
            gap: 18px;
        }

        .avatar-circle {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: #0B132B;
            color: #FFFFFF;
            font-size: 12px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            letter-spacing: 0.02em;
        }

        /* New Check Button (Right at Top of Left Sidebar) */
        .st-key-btn_new_check button {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 11px 16px !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
            margin-top: 4px !important;
            margin-bottom: 0.5rem !important;
            width: 100% !important;
            transition: all 0.15s ease !important;
        }

        .st-key-btn_new_check button:hover {
            border-color: var(--groww-emerald) !important;
            background: #F8FAFC !important;
            color: var(--groww-emerald-dark) !important;
        }

        .sidebar-heading {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 24px 0 10px 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Recent Inquiry Items */
        .st-key-recent_0 button,
        .st-key-recent_1 button,
        .st-key-recent_2 button,
        .st-key-recent_3 button {
            background: transparent !important;
            border: none !important;
            padding: 9px 8px !important;
            color: #475569 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            text-align: left !important;
            display: block !important;
            width: 100% !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            transition: all 0.15s ease !important;
        }

        .st-key-recent_0 button:hover,
        .st-key-recent_1 button:hover,
        .st-key-recent_2 button:hover,
        .st-key-recent_3 button:hover {
            background: #F1F5F9 !important;
            color: var(--text-primary) !important;
        }

        /* Bottom Search Placement */
        .search-area-bottom {
            margin-top: clamp(4rem, 18vh, 14rem);
        }

        .search-area-normal {
            margin-top: 1.5rem;
        }

        .sidebar-footer {
            margin-top: clamp(6rem, 28vh, 22rem);
            padding-top: 14px;
            border-top: 1px solid var(--border-light);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .status-feed {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--groww-emerald);
        }

        /* Hero Area */
        .hero-section {
            text-align: center;
            margin: 1.5rem auto 2.5rem auto;
            max-width: 680px;
        }

        .hero-title {
            font-size: 46px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.035em;
            margin: 0 0 10px 0;
            line-height: 1.15;
        }

        .hero-highlight {
            color: var(--groww-emerald-dark);
        }

        .hero-subtitle {
            font-size: 16px;
            color: var(--text-secondary);
            font-weight: 400;
            line-height: 1.5;
            margin: 0;
        }

        /* 4 Prompt Cards Grid */
        .st-key-card_0 button,
        .st-key-card_1 button,
        .st-key-card_2 button,
        .st-key-card_3 button {
            background: #FFFFFF !important;
            border: 1.5px solid #EAECF0 !important;
            border-radius: 16px !important;
            min-height: 175px !important;
            height: 100% !important;
            padding: 16px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-start !important;
            justify-content: flex-start !important;
            text-align: left !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }

        .st-key-card_0 button:hover,
        .st-key-card_1 button:hover,
        .st-key-card_2 button:hover,
        .st-key-card_3 button:hover {
            border-color: var(--border-hover) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 24px rgba(0, 208, 156, 0.12) !important;
            background: #FFFFFF !important;
        }

        .st-key-card_0 button p,
        .st-key-card_1 button p,
        .st-key-card_2 button p,
        .st-key-card_3 button p {
            color: var(--text-primary) !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            margin: 0 0 10px 0 !important;
            line-height: 1.3 !important;
        }

        /* Card 0: Calculation Box */
        .st-key-card_0 button::after {
            content: "1. Exit Load Calculation:\\A load = (days < 365) ? 1.0%\\A : 0%\\A net_nav = nav * (1 - load)";
            white-space: pre-wrap;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10.5px;
            line-height: 1.45;
            color: var(--text-secondary);
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 8px 10px;
            width: calc(100% - 20px);
            display: block;
            text-align: left;
        }

        /* Card 1: Illustrated Document Preview */
        .st-key-card_1 button::after {
            content: "";
            width: calc(100% - 20px);
            height: 85px;
            background: #F8FAFC url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="160" height="85" viewBox="0 0 160 85"><rect x="18" y="8" width="124" height="68" rx="6" fill="%23FFFFFF" stroke="%23E2E8F0" stroke-width="1.5"/><rect x="26" y="16" width="46" height="15" rx="3" fill="%23FEE2E2"/><text x="31" y="27" font-family="sans-serif" font-size="8.5" font-weight="bold" fill="%23EF4444">PDF SID</text><circle cx="124" cy="24" r="7" fill="%2300D09C"/><text x="121" y="28" font-family="sans-serif" font-size="11" font-weight="bold" fill="%23FFFFFF">+</text><line x1="26" y1="42" x2="132" y2="42" stroke="%23E2E8F0" stroke-width="2.2" stroke-linecap="round"/><line x1="26" y1="52" x2="105" y2="52" stroke="%23E2E8F0" stroke-width="2.2" stroke-linecap="round"/><line x1="26" y1="62" x2="118" y2="62" stroke="%23E2E8F0" stroke-width="2.2" stroke-linecap="round"/></svg>') center/contain no-repeat;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            display: block;
        }

        /* Card 2: TER Breakdown */
        .st-key-card_2 button::after {
            content: "Statutory TER Difference:\\A [Parag Parikh Flexi Cap]\\A 1. Direct Plan: 0.78%\\A 2. Regular Plan: 1.63% (Δ 0.85%)";
            white-space: pre-wrap;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 10.5px;
            line-height: 1.45;
            color: var(--text-secondary);
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 8px 10px;
            width: calc(100% - 20px);
            display: block;
            text-align: left;
        }

        /* Card 3: Emerald Bar Chart */
        .st-key-card_3 button::after {
            content: "";
            width: calc(100% - 20px);
            height: 85px;
            background: #F8FAFC url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="160" height="85" viewBox="0 0 160 85"><rect x="25" y="48" width="9" height="24" rx="2" fill="%2300D09C"/><rect x="38" y="34" width="9" height="38" rx="2" fill="%2300D09C"/><rect x="51" y="16" width="9" height="56" rx="2" fill="%2300D09C"/><rect x="64" y="26" width="9" height="46" rx="2" fill="%2300D09C"/><rect x="77" y="52" width="9" height="20" rx="2" fill="%2300D09C"/><rect x="90" y="40" width="9" height="32" rx="2" fill="%2300D09C"/><rect x="103" y="20" width="9" height="52" rx="2" fill="%2300D09C"/><rect x="116" y="28" width="9" height="44" rx="2" fill="%2300D09C"/></svg>') center/contain no-repeat;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            display: block;
        }

        /* Floating Pill Search Bar Form */
        [data-testid="stForm"] {
            background: #FFFFFF !important;
            border: 1.5px solid #EAECF0 !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
            padding: 6px 8px 6px 14px !important;
            margin-top: 1.5rem !important;
            width: 100% !important;
            box-sizing: border-box !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }

        [data-testid="stForm"]:focus-within {
            border-color: var(--groww-emerald) !important;
            box-shadow: 0 6px 24px rgba(0, 208, 156, 0.14) !important;
        }

        /* Input Element Inside Searchbar - Pure White Fill */
        .stTextInput,
        .stTextInput > div,
        .stTextInput > div > div,
        [data-testid="stTextInput"],
        [data-testid="stTextInput"] > div,
        [data-testid="stTextInput"] > div > div,
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stTextInputRootElement"] div[data-baseweb="input"],
        [data-testid="stTextInputRootElement"] div[data-baseweb="base-input"],
        [data-testid="stTextInputRootElement"] input {
            border: none !important;
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            box-shadow: none !important;
            outline: none !important;
            color: var(--text-primary) !important;
        }

        /* Input row with left search/tune icon and right mic icon */
        [data-testid="stTextInputRootElement"] div[data-baseweb="input"] {
            padding: 0 !important;
            min-height: 48px !important;
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            display: flex !important;
            align-items: center !important;
        }

        /* Far-left search/tune icon */
        [data-testid="stTextInputRootElement"] div[data-baseweb="input"]::before {
            content: "";
            display: inline-block;
            width: 19px;
            height: 19px;
            background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="%2364748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="11" y2="6"/><line x1="3" y1="12" x2="9" y2="12"/><line x1="3" y1="18" x2="11" y2="18"/><circle cx="16" cy="12" r="3.5"/><line x1="18.5" y1="14.5" x2="21" y2="17"/></svg>') center/contain no-repeat;
            margin-right: 10px;
            margin-left: 2px;
            flex-shrink: 0;
        }

        /* Right microphone icon */
        [data-testid="stTextInputRootElement"] div[data-baseweb="input"]::after {
            content: "";
            display: inline-block;
            width: 18px;
            height: 18px;
            background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="%2364748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>') center/contain no-repeat;
            margin-right: 12px;
            margin-left: 6px;
            flex-shrink: 0;
            cursor: pointer;
        }

        [data-testid="stTextInputRootElement"] div[data-baseweb="base-input"] {
            position: relative !important;
            display: flex !important;
            align-items: center !important;
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            border-radius: 12px !important;
            flex: 1 !important;
        }

        /* ⌘K keyboard shortcut badge */
        [data-testid="stTextInputRootElement"] div[data-baseweb="base-input"]::after {
            content: "⌘K";
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            background: #F1F5F9;
            color: #64748B;
            font-size: 12px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 6px;
            letter-spacing: 0.02em;
            pointer-events: none;
            z-index: 3;
        }

        [data-testid="stTextInputRootElement"] input {
            color: var(--text-primary) !important;
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            font-size: 14.5px !important;
            font-weight: 500 !important;
            caret-color: var(--groww-emerald-dark) !important;
            padding-left: 0 !important;
            padding-right: 56px !important;
            min-height: 48px !important;
            position: relative;
            z-index: 2;
            border-radius: 12px !important;
        }

        /* Suggested Questions Smooth Animation in Searchbar */
        @keyframes smoothQuestionSlide {
            0% {
                content: "Ask a follow-up fact about HDFC ELSS Tax Saver Fund...";
                opacity: 0;
                transform: translateY(calc(-50% + 10px));
            }
            2%, 17% {
                content: "Ask a follow-up fact about HDFC ELSS Tax Saver Fund...";
                opacity: 1;
                transform: translateY(-50%);
            }
            19.5% {
                content: "Ask a follow-up fact about HDFC ELSS Tax Saver Fund...";
                opacity: 0;
                transform: translateY(calc(-50% - 10px));
            }
            19.6% {
                content: "What is the exit load for HDFC Flexi Cap Fund?";
                opacity: 0;
                transform: translateY(calc(-50% + 10px));
            }
            22%, 37% {
                content: "What is the exit load for HDFC Flexi Cap Fund?";
                opacity: 1;
                transform: translateY(-50%);
            }
            39.5% {
                content: "What is the exit load for HDFC Flexi Cap Fund?";
                opacity: 0;
                transform: translateY(calc(-50% - 10px));
            }
            39.6% {
                content: "Compare Direct vs Regular TER for HDFC Top 100...";
                opacity: 0;
                transform: translateY(calc(-50% + 10px));
            }
            42%, 57% {
                content: "Compare Direct vs Regular TER for HDFC Top 100...";
                opacity: 1;
                transform: translateY(-50%);
            }
            59.5% {
                content: "Compare Direct vs Regular TER for HDFC Top 100...";
                opacity: 0;
                transform: translateY(calc(-50% - 10px));
            }
            59.6% {
                content: "What is the lock-in period for HDFC ELSS Tax Saver?";
                opacity: 0;
                transform: translateY(calc(-50% + 10px));
            }
            62%, 77% {
                content: "What is the lock-in period for HDFC ELSS Tax Saver?";
                opacity: 1;
                transform: translateY(-50%);
            }
            79.5% {
                content: "What is the lock-in period for HDFC ELSS Tax Saver?";
                opacity: 0;
                transform: translateY(calc(-50% - 10px));
            }
            79.6% {
                content: "Inspect statutory SID & KIM for HDFC Mid-Cap Opportunities...";
                opacity: 0;
                transform: translateY(calc(-50% + 10px));
            }
            82%, 97% {
                content: "Inspect statutory SID & KIM for HDFC Mid-Cap Opportunities...";
                opacity: 1;
                transform: translateY(-50%);
            }
            99.5% {
                content: "Inspect statutory SID & KIM for HDFC Mid-Cap Opportunities...";
                opacity: 0;
                transform: translateY(calc(-50% - 10px));
            }
            100% {
                content: "Ask a follow-up fact about HDFC ELSS Tax Saver Fund...";
                opacity: 0;
                transform: translateY(calc(-50% + 10px));
            }
        }

        [data-testid="stTextInputRootElement"] div[data-baseweb="base-input"]::before {
            content: "Ask a follow-up fact about HDFC ELSS Tax Saver Fund...";
            position: absolute;
            left: 2px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 14.5px;
            font-weight: 400;
            pointer-events: none;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: calc(100% - 70px);
            z-index: 1;
            animation: smoothQuestionSlide 20s cubic-bezier(0.2, 0.8, 0.2, 1) infinite;
        }

        /* Hide animated placeholder on focus or when input has content */
        [data-testid="stTextInputRootElement"] div[data-baseweb="base-input"]:focus-within::before,
        [data-testid="stTextInputRootElement"] div[data-baseweb="base-input"]:has(input:not(:placeholder-shown))::before {
            display: none !important;
            opacity: 0 !important;
            animation: none !important;
        }

        /* Ask Facts AI Button (Green Rounded Rect) */
        .st-key-ask_button button {
            background: var(--groww-emerald-dark) !important;
            color: #FFFFFF !important;
            border-radius: 12px !important;
            border: none !important;
            min-height: 44px !important;
            height: 44px !important;
            padding: 0 18px !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 6px !important;
            box-shadow: 0 2px 8px rgba(0, 135, 90, 0.2) !important;
            transition: all 0.15s ease !important;
        }

        .st-key-ask_button button:hover {
            background: #00704A !important;
            transform: scale(1.02) !important;
            color: #FFFFFF !important;
        }

        .st-key-ask_button button p {
            color: #FFFFFF !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }

        /* Disclaimer under Search Bar */
        .footer-disclaimer {
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
            margin-top: 14px;
        }

        /* Response Card */
        .response-card {
            background: #FFFFFF;
            border: 1.5px solid var(--border-light);
            border-radius: 20px;
            padding: 24px;
            margin-top: 2rem;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease;
        }

        .response-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .response-tag {
            background: var(--groww-emerald-subtle);
            color: var(--groww-emerald-dark);
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            letter-spacing: 0.04em;
        }

        .response-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--groww-emerald);
        }

        .query-bubble {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-primary);
            padding: 10px 14px;
            background: #F8FAFC;
            border-radius: 10px;
            margin-bottom: 16px;
            border-left: 3px solid var(--groww-emerald);
        }

        .answer-content {
            font-size: 15px;
            line-height: 1.65;
            color: #334155;
            margin-bottom: 18px;
        }

        .source-action-row {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-top: 14px;
            padding-top: 14px;
            border-top: 1px solid var(--border-light);
        }

        [data-testid="stLinkButton"] a {
            background: var(--groww-emerald-subtle) !important;
            border: 1px solid rgba(0, 208, 156, 0.3) !important;
            border-radius: 8px !important;
            color: var(--groww-emerald-dark) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            padding: 6px 14px !important;
            transition: all 0.15s ease !important;
        }

        [data-testid="stLinkButton"] a:hover {
            background: rgba(0, 208, 156, 0.18) !important;
            border-color: var(--groww-emerald) !important;
            color: var(--groww-emerald-dark) !important;
        }

        .source-date {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .response-disclaimer {
            font-size: 11.5px;
            color: var(--text-muted);
            margin-top: 14px;
            padding-top: 10px;
            border-top: 1px dashed var(--border-light);
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 32px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        fingerprint = compute_index_fingerprint()
        _prepare_index(str(settings.CHROMA_DB_PATH), fingerprint=fingerprint)
    except Exception as exc:
        st.error(f"Unable to prepare the local knowledge index: {exc}")
        return

    # Two column layout: Left Sidebar (width 1) and Main Canvas (width 3.8)
    sidebar_col, main_col = st.columns([1, 3.8], gap="large")

    # ------------------ LEFT SIDEBAR ------------------
    with sidebar_col:
        # New Check Button placed directly at the top
        if st.button("+ New Check", key="btn_new_check", use_container_width=True):
            st.session_state["question_input"] = ""
            st.session_state["active_response"] = None
            st.session_state["last_asked_question"] = ""
            st.session_state["auto_submit"] = None
            st.rerun()

        # Recent Inquiries Header
        st.markdown(
            '<div class="sidebar-heading">'
            '<span>RECENT INQUIRIES</span>'
            '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Recent Inquiries list
        for idx, (label, query) in enumerate(RECENT_INQUIRIES):
            st.button(
                label,
                key=f"recent_{idx}",
                on_click=_select_example,
                args=(query,),
                use_container_width=True,
            )

        # Bottom status in sidebar
        st.markdown(
            '<div class="sidebar-footer">'
            '<span class="status-feed"><span class="status-dot"></span> SEBI SID Feeds</span>'
            '<span class="status-ver">v2.4</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ------------------ MAIN CANVAS ------------------
    with main_col:
        # Top Navigation Bar
        st.markdown(
            """
            <div class="top-navbar">
                <div class="nav-left">
                    <a href="#" class="groww-brand">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00D09C" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline>
                            <polyline points="16 7 22 7 22 13"></polyline>
                        </svg>
                        <span>Groww</span>
                    </a>
                    <div class="facts-tag">
                        <span class="facts-tag-dot"></span> FACTS AI
                    </div>
                </div>
                <div class="nav-center">
                    <a href="#" class="nav-link active">Explore</a>
                    <a href="#" class="nav-link">Mutual Funds</a>
                    <a href="#" class="nav-link">Tools</a>
                </div>
                <div class="nav-right">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                    </svg>
                    <div class="avatar-circle">IN</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hero Greeting & Subtitle
        st.markdown(
            """
            <div class="hero-section">
                <h1 class="hero-title">Hello, <span class="hero-highlight">Investor</span></h1>
                <p class="hero-subtitle">Ask any factual question about mutual funds, exit loads, expense ratios, or statutory rules.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Prompt Cards Grid
        card_cols = st.columns(4)
        card_titles = (
            "Explain Scheme Exit Loads",
            "Inspect Statutory SID & KIM",
            "Compare Direct vs Regular TER",
            "Analyze Rolling Returns & Risk",
        )
        for idx, (title, query) in enumerate(zip(card_titles, EXAMPLE_QUESTIONS)):
            with card_cols[idx]:
                st.button(
                    title,
                    key=f"card_{idx}",
                    on_click=_select_example,
                    args=(query,),
                    use_container_width=True,
                )

        # Display active response above the search bar if present
        if st.session_state.get("active_response"):
            _display_response(
                st.session_state["active_response"],
                question_text=st.session_state.get("last_asked_question", ""),
            )

        # Space pushing search bar to the bottom on initial screen
        search_spacer = "search-area-normal" if st.session_state.get("active_response") else "search-area-bottom"
        st.markdown(f'<div class="{search_spacer}"></div>', unsafe_allow_html=True)

        # Floating Pill Search Bar Form at the bottom
        with st.form("question_form", clear_on_submit=False):
            search_cols = st.columns([8.2, 1.8], gap="small")
            with search_cols[0]:
                question = st.text_input(
                    "Question",
                    key="question_input",
                    label_visibility="collapsed",
                    placeholder=" ",
                )
            with search_cols[1]:
                submitted = st.form_submit_button(
                    "Ask Facts AI ↑",
                    key="ask_button",
                    use_container_width=True,
                )

        # Footer disclaimer under bottom search bar
        st.markdown(
            '<div class="footer-disclaimer">'
            'Strictly factual data from SEBI SID & AMFI filings • Zero subjective advice'
            '</div>',
            unsafe_allow_html=True,
        )

        # Execute query if submitted or if auto_submit was triggered by a card/recent inquiry
        active_query = ""
        if submitted and question.strip():
            active_query = question.strip()
        elif st.session_state.get("auto_submit"):
            active_query = st.session_state["auto_submit"]
            st.session_state["auto_submit"] = None

        if active_query:
            try:
                with st.spinner("Checking official SEBI and AMC sources..."):
                    response = answer_question(active_query)
                st.session_state["active_response"] = response
                st.session_state["last_asked_question"] = active_query
                st.rerun()
            except GroqRequestError as exc:
                st.error(str(exc))
            except Exception:
                st.error("The question could not be answered. Check the local index and Groq configuration, then try again.")
        elif submitted and not question.strip():
            st.warning("Enter a question first.")


if __name__ == "__main__":
    main()
