"""Streamlit interface: ask a question, view the answer, and inspect source articles."""
import streamlit as st
from src.rag import answer_question, get_threshold

st.set_page_config(page_title="HelpDeskQA — Basic RAG", page_icon="💬", layout="centered")
st.title("HelpDeskQA")
st.caption("Answers are generated from a small IT knowledge base using ChromaDB retrieval.")

with st.sidebar:
    st.subheader("About this demo")
    st.write("Knowledge-base sections: Wi-Fi, Passwords, Software, Hardware, and Network access.")
    st.write("The app rejects questions when the best retrieval match is below the configured threshold.")
    st.metric("Active similarity threshold", f"{get_threshold():.3f}")
    st.caption("Run evaluate.py to measure retrieval and tune the threshold.")

question = st.chat_input("Ask an IT support question...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the knowledge base..."):
                result = answer_question(question)
            st.write(result["answer"])
            st.caption(
                f"Best cosine similarity: {result['best_similarity']} · "
                f"Rejection threshold: {result['threshold']:.3f}"
                if result["best_similarity"] is not None
                else f"Rejection threshold: {result['threshold']:.3f}"
            )
            if result["citations"]:
                st.subheader("Source articles")
                # Deduplicate citations when multiple chunks come from one article.
                seen = set()
                for item in result["citations"]:
                    if item["article_id"] in seen:
                        continue
                    seen.add(item["article_id"])
                    st.markdown(
                        f"- **{item['article_title']}** — {item['section']}  "
                        f"`{item['article_id']}` (similarity {item['similarity']:.3f})"
                    )
            elif result["answer"] != "I couldn’t find this.":
                st.info("The response did not return a valid supporting article ID.")
            with st.expander("Inspect retrieved chunks"):
                for item in result["retrieved"]:
                    st.markdown(
                        f"**{item['article_title']}** · {item['article_id']} · "
                        f"similarity {item['similarity']:.3f}"
                    )
                    st.write(item["text"])
        except Exception as exc:
            st.error(str(exc))
            st.info("Check setup instructions in README.md and confirm that the index has been built.")
