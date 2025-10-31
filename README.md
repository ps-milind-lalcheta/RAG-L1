# Basic RAG over PDFs

A simple Retrieval-Augmented Generation (RAG) app.

## Steps

1. Put PDFs in `data/`
2. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Click **Rebuild index** in sidebar, then ask questions!

If you have `OPENAI_API_KEY`, GPT-4 will be used; otherwise a local FLAN-T5 model.
