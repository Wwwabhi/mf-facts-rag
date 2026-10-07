from __future__ import annotations

import html
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import settings
from src.guardrails.answer_policy import GuardrailDecision
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
    st.session_state["active_chat_id"] = None


def _generate_chat_title(question: str) -> str:
    clean = question.strip().rstrip("?").strip()
    for prefix in (
        "What is the exit load for ",
        "What is the lock-in period for ",
        "What is the expense ratio of ",
        "What is the NAV of ",
        "Should I invest in ",
        "How can I download a ",
        "Can I invest in ",
        "What is ",
        "How to ",
    ):
        if clean.lower().startswith(prefix.lower()):
            clean = clean[len(prefix):].strip()
            break
    clean = clean[:32].strip()
    if not clean:
        clean = "Mutual Fund Query"
    return clean[0].upper() + clean[1:]


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
        initial_sidebar_state="expanded",
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "active_chat_id" not in st.session_state:
        st.session_state["active_chat_id"] = None

    # Handle query param actions
    if "new_chat" in st.query_params:
        st.session_state["active_chat_id"] = None
        st.session_state["question_input"] = ""
        st.query_params.clear()
    elif "chat_id" in st.query_params:
        target_id = st.query_params.get("chat_id")
        matching = next((c for c in st.session_state["chat_history"] if c["id"] == target_id), None)
        if matching:
            st.session_state["active_chat_id"] = target_id
            st.session_state["question_input"] = matching["question"]
    elif "q" in st.query_params:
        requested_q = st.query_params.get("q")
        if requested_q and ("question_input" not in st.session_state or not st.session_state["question_input"]):
            st.session_state["question_input"] = requested_q
            st.session_state["active_chat_id"] = None

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

        /* Global resets & background */
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
            background: var(--bg-main) !important;
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Hide Streamlit Deploy button, decoration line, and toolbar */
        .stAppDeployButton,
        [data-testid="stStatusWidget"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }

        /* Center content layout with strict unified alignment */
        .block-container {
            max-width: 780px !important;
            padding-top: 3.5rem !important;
            padding-bottom: 4rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            margin: 0 auto !important;
        }

        /* Sidebar container */
        [data-testid="stSidebar"] {
            background-color: #0e1215 !important;
            border-right: 1px solid #1a2025 !important;
        }

        /* Sidebar Header and Brand alignment at top left */
        [data-testid="stSidebarHeader"] {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
            padding: 0.85rem 1rem 0 1rem !important;
            height: 48px !important;
            min-height: 48px !important;
            position: relative !important;
            background: transparent !important;
        }

        [data-testid="stSidebarUserContent"] {
            padding: 0 1rem 1.5rem 1rem !important;
            margin-top: -48px !important;
        }

        /* Collapse / Expand Hamburger button */
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button {
            background: #14191c !important;
            border: 1px solid #232c32 !important;
            border-radius: 8px !important;
            color: #9ba8b5 !important;
            width: 32px !important;
            height: 32px !important;
            min-height: 32px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
            z-index: 10 !important;
            transition: background 0.15s ease, border-color 0.15s ease !important;
        }

        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover {
            background: #1c2227 !important;
            border-color: #354049 !important;
            color: #ffffff !important;
        }

        /* Completely hide any inner default icons/spans/svgs */
        [data-testid="stSidebarCollapseButton"] button *,
        [data-testid="stSidebarCollapsedControl"] button * {
            display: none !important;
        }

        /* Single 3-bar hamburger icon */
        [data-testid="stSidebarCollapseButton"] button::after,
        [data-testid="stSidebarCollapsedControl"] button::after {
            content: "";
            display: block;
            width: 14px;
            height: 10px;
            background: linear-gradient(
                to bottom,
                #9ba8b5 0px, #9ba8b5 2px,
                transparent 2px, transparent 4px,
                #9ba8b5 4px, #9ba8b5 6px,
                transparent 6px, transparent 8px,
                #9ba8b5 8px, #9ba8b5 10px
            );
        }

        [data-testid="stSidebarCollapseButton"] button:hover::after,
        [data-testid="stSidebarCollapsedControl"] button:hover::after {
            background: linear-gradient(
                to bottom,
                #ffffff 0px, #ffffff 2px,
                transparent 2px, transparent 4px,
                #ffffff 4px, #ffffff 6px,
                transparent 6px, transparent 8px,
                #ffffff 8px, #ffffff 10px
            );
        }

        [data-testid="stSidebarCollapsedControl"] {
            top: 1.1rem !important;
            left: 1.1rem !important;
        }

        /* Top Left Corner Brand Lockup */
        .sidebar-brand-wrapper {
            display: flex !important;
            align-items: center !important;
            height: 48px !important;
            padding-right: 42px !important;
            margin-bottom: 1.25rem !important;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .sidebar-logo {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            overflow: hidden;
        }

        .sidebar-brand-text {
            display: flex;
            flex-direction: column;
            justify-content: center;
            line-height: 1.15;
        }

        .groww-brand-name {
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .brand-sub-title {
            color: #8a96a1;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.01em;
        }

        /* New Chat Button */
        .new-chat-button {
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
            height: 44px;
            padding: 0 12px;
            background: #181e23;
            border: 1px solid #232c33;
            border-radius: 12px;
            color: #f0f6fc !important;
            text-decoration: none !important;
            font-size: 0.92rem;
            font-weight: 500;
            box-sizing: border-box;
            transition: background 0.15s ease, border-color 0.15s ease;
            margin-bottom: 1.75rem;
        }

        .new-chat-button:hover {
            background: #1e262c;
            border-color: #2f3b45;
            color: #ffffff !important;
        }

        .new-chat-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            background: #00d09c;
            border-radius: 6px;
            color: #0b1512;
            font-weight: 700;
            flex-shrink: 0;
        }

        /* Previous Chats Section */
        .sidebar-section-header {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #657382;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            margin-bottom: 0.85rem;
        }

        .sidebar-section-header svg {
            color: #657382;
        }

        .sidebar-chat-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .sidebar-chat-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            border-radius: 8px;
            color: #9ba8b5 !important;
            text-decoration: none !important;
            font-size: 0.89rem;
            font-weight: 400;
            transition: background 0.12s ease, color 0.12s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .sidebar-chat-item svg {
            color: #6d7b88;
            flex-shrink: 0;
            transition: color 0.12s ease;
        }

        .sidebar-chat-item:hover {
            background: #161c21;
            color: #f0f6fc !important;
        }

        .sidebar-chat-item:hover svg {
            color: #9ba8b5;
        }

        .sidebar-chat-item.active {
            background: #1c2329;
            color: #ffffff !important;
            font-weight: 500;
        }

        .sidebar-chat-item.active svg {
            color: #00d09c;
        }

        .sidebar-empty-state {
            color: #55606d;
            font-size: 0.82rem;
            padding: 8px 10px;
            font-style: italic;
        }

        /* Center Section: Hero Title */
        .hero-heading {
            color: #ffffff !important;
            font-size: 2.75rem !important;
            font-weight: 700 !important;
            text-align: center !important;
            margin: 0 0 1.5rem 0 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.2 !important;
        }

        /* Center Section: Facts Disclaimer Banner (100% width matching cards) */
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

        /* Center Section: Example questions label */
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

        /* Center Section: Input Form (100% width matching cards) */
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

        /* User Active Question Card */
        .user-query-container {
            width: 100% !important;
            box-sizing: border-box !important;
            background: #13181b;
            border: 1px solid var(--border-card);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .user-query-badge {
            color: var(--emerald-primary);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }

        .user-query-text {
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 500;
            line-height: 1.4;
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
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        brand_html = """
        <div class="sidebar-brand-wrapper">
            <div class="sidebar-brand">
                <div class="sidebar-logo">
                    <svg width="28" height="28" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <clipPath id="groww-circle-clip">
                            <circle cx="50" cy="50" r="50"/>
                        </clipPath>
                        <g clip-path="url(#groww-circle-clip)">
                            <rect width="100" height="100" fill="#5367FF"/>
                            <path d="M-5 105 L-5 72 L42 56 L62 66 L105 44 L105 105 Z" fill="#00D09C"/>
                        </g>
                    </svg>
                </div>
                <div class="sidebar-brand-text">
                    <span class="groww-brand-name">Groww</span>
                    <span class="brand-sub-title">Mutual Fund Facts</span>
                </div>
            </div>
        </div>

        <a href="?new_chat=1" class="new-chat-button" target="_self">
            <div class="new-chat-icon">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
            </div>
            <span>New Chat</span>
        </a>

        <div class="sidebar-section-header">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            <span>PREVIOUS CHATS</span>
        </div>
        """

        history_items_html = []
        if not st.session_state["chat_history"]:
            history_items_html.append(
                '<div class="sidebar-empty-state">No previous chats yet</div>'
            )
        else:
            active_id = st.session_state["active_chat_id"]
            for chat in st.session_state["chat_history"]:
                is_active = (chat["id"] == active_id)
                active_class = "active" if is_active else ""
                escaped_title = html.escape(chat["title"])
                escaped_id = html.escape(chat["id"])
                history_items_html.append(
                    f'<a href="?chat_id={escaped_id}" class="sidebar-chat-item {active_class}" target="_self">'
                    '<svg class="chat-item-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                    '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>'
                    '</svg>'
                    f'<span>{escaped_title}</span>'
                    '</a>'
                )

        chat_list_html = f'<div class="sidebar-chat-list">{"".join(history_items_html)}</div>'
        st.markdown(brand_html + chat_list_html, unsafe_allow_html=True)

    try:
        _prepare_index(str(settings.CHROMA_DB_PATH))
    except Exception as exc:
        st.error(f"Unable to prepare the local knowledge index: {exc}")
        return

    active_chat = None
    if st.session_state["active_chat_id"]:
        active_chat = next((c for c in st.session_state["chat_history"] if c["id"] == st.session_state["active_chat_id"]), None)

    if active_chat:
        # Display selected previous conversation
        st.markdown(
            f'<div class="user-query-container">'
            f'<span class="user-query-badge">QUESTION</span>'
            f'<div class="user-query-text">{html.escape(active_chat["question"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        saved_response = QueryResponse(
            answer=active_chat["answer"],
            source_url=active_chat.get("source_url"),
            source_date=active_chat.get("source_date"),
            retrieved_chunks=[],
            decision=GuardrailDecision(
                allowed=active_chat.get("allowed", True),
                category="history",
                reason="history",
                message=active_chat.get("answer"),
            ),
        )
        _display_response(saved_response)
    else:
        # Display new chat welcome view
        st.markdown(
            '<h1 class="hero-heading">Welcome to Mutual Fund Facts</h1>',
            unsafe_allow_html=True,
        )
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
                
                chat_entry = {
                    "id": str(len(st.session_state["chat_history"]) + 1),
                    "question": question.strip(),
                    "title": _generate_chat_title(question.strip()),
                    "answer": response.answer,
                    "source_url": response.source_url,
                    "source_date": response.source_date,
                    "allowed": response.decision.allowed,
                }
                st.session_state["chat_history"].insert(0, chat_entry)
                st.session_state["active_chat_id"] = chat_entry["id"]
                _display_response(response)
            except GroqRequestError as exc:
                st.error(str(exc))
            except Exception:
                st.error("The question could not be answered. Check the local index and Groq configuration, then try again.")


if __name__ == "__main__":
    main()