import os, textwrap, subprocess, sys
import streamlit as st
from rag_core import (
    load_config, get_embedder, Retriever,
    answer_with_openai, answer_with_local_llm
)

st.set_page_config(page_title="RAG System", page_icon="📄", layout="wide")
st.title("📄RAG System")

cfg = load_config("config.yaml")
data_dir = cfg["data_dir"]
artifacts_dir = cfg["artifacts_dir"]
index_name = cfg["index_name"]
meta_name = cfg["meta_name"]

with st.sidebar:
    st.header("Settings")
    st.caption("Put PDFs in `data/` then (re)build index.")
    if st.button("🔨 Rebuild index"):
        with st.spinner("Indexing PDFs..."):
            cmd = [sys.executable, "ingest.py"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            st.text(res.stdout if res.stdout else "")
            if res.returncode != 0:
                st.error(res.stderr)
            else:
                st.success("Index rebuilt. You can ask questions now.")

    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=cfg["retrieval"]["top_k"], step=1)
    score_threshold = st.number_input("Score threshold (0 to disable)", min_value=0.0, max_value=1.0, value=float(cfg["retrieval"]["score_threshold"]), step=0.01, format="%.2f")
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    model_info = "OpenAI (env var set)" if use_openai else f"Local: {cfg['model']['local_generator']}"
    st.write(f"Generator: **{model_info}**")
    st.divider()
    st.caption("Index status:")
    has_index = all(os.path.exists(os.path.join(artifacts_dir, p)) for p in [f"{index_name}.faiss", "texts.pkl", meta_name])
    st.write("✅ Ready" if has_index else "❌ Not built")

# Stop if no index
if not all(os.path.exists(os.path.join(artifacts_dir, p)) for p in [f"{index_name}.faiss", "texts.pkl", meta_name]):
    st.info("Add PDFs to `data/` and click **Rebuild index** in the sidebar.")
    st.stop()

# Load retriever + embedder
embedder = get_embedder(cfg["model"]["embedding"])
retriever = Retriever(artifacts_dir=artifacts_dir, index_name=index_name, meta_name=meta_name)

# Chat UI
if "history" not in st.session_state: st.session_state.history = []

q = st.text_input("Ask a question about your PDFs")
ask = st.button("Ask")

def render_hit(hit):
    meta = hit["meta"]
    st.markdown(f"**Score:** {hit['score']:.3f} • **File:** `{os.path.basename(meta['doc_path'])}` • **Page:** {meta['page']}")
    with st.expander("View chunk"):
        st.code(textwrap.shorten(hit["text"], width=2000, placeholder=" … "), language="markdown")

if ask and q.strip():
    with st.spinner("Retrieving..."):
        hits = retriever.search(q, embedder, top_k=top_k, score_threshold=score_threshold)
    if not hits:
        st.warning("No relevant chunks found. Try rewording.")
    else:
        context = "\n\n---\n\n".join([h["text"] for h in hits])
        with st.spinner("Generating answer..."):
            try:
                if bool(os.getenv("OPENAI_API_KEY")):
                    answer = answer_with_openai(context, q)
                else:
                    answer = answer_with_local_llm(context, q, model_name=cfg["model"]["local_generator"])
            except Exception as e:
                answer = f"(Generation failed) {e}"

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for h in hits:
            render_hit(h)

        st.session_state.history.append({"q": q, "a": answer})

if st.session_state.history:
    st.divider()
    st.subheader("History")
    for turn in st.session_state.history[::-1]:
        st.markdown(f"**Q:** {turn['q']}")
        st.markdown(f"**A:** {turn['a']}")
        st.markdown("---")
