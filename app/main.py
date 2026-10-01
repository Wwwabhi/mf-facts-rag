from __future__ import annotations

import streamlit as st

from src.retrieval.query_pipeline import QueryResponse, answer_question


EXAMPLE_QUESTIONS = (
    "What is the exit load for HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "How can I download a consolidated account statement?",
)


def _select_example(question: str) -> None:
    st.session_state["question_input"] = question


def _display_response(response: QueryResponse) -> None:
    st.subheader("Answer")
    if response.answer:
        st.write(response.answer)
    if response.source_url and response.source_date:
        st.link_button("Open official source", response.source_url)
        st.caption(f"Last updated from sources: {response.source_date}")
    elif response.decision.allowed:
        st.info("The available approved sources could not verify this answer.")


def main() -> None:
    st.set_page_config(page_title="Mutual Fund Facts", page_icon="📄", layout="centered")
    st.markdown(
        """
        <style>
        .block-container { max-width: 780px; padding-top: 2.5rem; }
        [data-testid="stAppViewContainer"] { background: #f7f8f5; }
        [data-testid="stHeader"] { background: transparent; }
        h1, h2, h3 { color: #17352c; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"] p { color: #17352c !important; }
        [data-testid="stAppViewContainer"] a,
        [data-testid="stAppViewContainer"] a p { color: #155b45 !important; }
        .facts-notice {
            border-left: 4px solid #28755b;
            background: #eaf2ed;
            color: #17352c;
            padding: 0.75rem 1rem;
            font-weight: 600;
            margin: 0.5rem 0 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Welcome to Mutual Fund Facts")
    st.markdown('<div class="facts-notice">Facts-only. No investment advice.</div>', unsafe_allow_html=True)

    st.markdown("**Example questions**")
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
        question = st.text_input(
            "Question",
            key="question_input",
            placeholder="Ask a factual question about a supported fund",
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Checking approved sources..."):
                    response = answer_question(question.strip())
                _display_response(response)
            except Exception:
                st.error("The question could not be answered. Check the local index and Groq configuration, then try again.")


if __name__ == "__main__":
    main()