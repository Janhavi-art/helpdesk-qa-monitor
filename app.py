"""Simple Streamlit interface for HelpDeskQA Monitor."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.rag import answer_question
from src.telemetry import read_jsonl, usage_summary
from src.config import API_USAGE_LOG, INTERACTIONS_LOG

st.set_page_config(page_title="HelpDeskQA Monitor", page_icon="💬", layout="centered")
st.title("HelpDeskQA Monitor")
st.caption("Ask an IT support question and get an answer grounded in the knowledge base.")

ask_tab, usage_tab = st.tabs(["Ask Helpdesk", "AI Usage"])

with ask_tab:
    with st.form("ask_form"):
        question = st.text_area(
            "Your question",
            placeholder="Example: My Wi-Fi keeps disconnecting. How can I fix it?",
            height=100,
        )
        submitted = st.form_submit_button("Generate answer", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching the knowledge base and generating an answer..."):
                    result = answer_question(question.strip())
                st.subheader("Answer")
                st.markdown(result.get("answer", "No answer was returned."))

                status = result.get("status", "")
                if status == "cache_hit":
                    st.caption("Cached answer — no new Gemini API call was made.")
                elif status == "local_abstention":
                    st.caption("No sufficiently relevant article was found; Gemini was not called.")
                elif status == "blocked":
                    st.caption("The request was blocked by an application usage limit.")

                sources = result.get("sources", [])
                if sources:
                    with st.expander("Sources used"):
                        for source in sources:
                            st.markdown(f"**{source.get('doc_name', 'Article')}** — {source.get('section', 'General')}")
                            st.caption(f"Relevance score: {source.get('score', 0):.3f}")
                            st.write(source.get("text", ""))

                usage = result.get("usage") or {}
                c1, c2 = st.columns(2)
                c1.metric("Tokens for this request", f"{result.get('tokens_this_request', 0):,}")
                c2.metric("Response time", f"{result.get('latency_seconds', 0):.2f} s")
                if usage:
                    st.caption(
                        f"Gemini reported: {usage.get('input_tokens', 0):,} input tokens · "
                        f"{usage.get('output_tokens', 0):,} output tokens · "
                        f"{usage.get('total_tokens', 0):,} total tokens"
                    )
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")

with usage_tab:
    st.subheader("AI usage")
    summary = usage_summary()
    a, b = st.columns(2)
    a.metric("API calls today", f"{summary['api_calls_today']:,}")
    b.metric("Tokens today", f"{summary['tokens_today']:,}")
    c, d = st.columns(2)
    c.metric("Input tokens today", f"{summary['input_tokens_today']:,}")
    d.metric("Output tokens today", f"{summary['output_tokens_today']:,}")
    e, f = st.columns(2)
    e.metric("Cached answers", f"{summary['cache_hits']:,}")
    f.metric("Questions answered/checked", f"{summary['interaction_count']:,}")
    st.caption("Token counts are shown when Gemini returns usage metadata. Cached answers and local retrieval do not use new Gemini tokens.")

    usage_rows = read_jsonl(API_USAGE_LOG)
    if usage_rows:
        st.markdown("#### Recent Gemini requests")
        df = pd.DataFrame(usage_rows)
        preferred = [c for c in ["timestamp", "model", "status", "input_tokens", "output_tokens", "total_tokens", "latency_seconds"] if c in df.columns]
        if preferred:
            st.dataframe(df[preferred].tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("No Gemini API usage has been logged yet. Ask a question in the Ask Helpdesk tab to test it.")

    interaction_rows = read_jsonl(INTERACTIONS_LOG)
    if interaction_rows:
        with st.expander("Recent questions and status"):
            df = pd.DataFrame(interaction_rows)
            preferred = [c for c in ["timestamp", "question", "status", "cache_hit", "total_tokens", "latency_seconds", "top_score"] if c in df.columns]
            if preferred:
                st.dataframe(df[preferred].tail(20), use_container_width=True, hide_index=True)
