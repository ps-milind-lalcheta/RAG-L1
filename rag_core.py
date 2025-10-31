import os, json, fitz, pickle, numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

@dataclass
class Chunk:
    text: str
    doc_path: str
    page: int
    chunk_id: str

def read_pdf(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text) for a PDF."""
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text and text.strip():
            pages.append((i + 1, text))
    doc.close()
    return pages

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n: break
        start = end - overlap
        if start < 0: start = 0
    return [c.strip() for c in chunks if c.strip()]

def collect_pdf_chunks(pdf_path: str, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, page_text in read_pdf(pdf_path):
        for idx, c in enumerate(chunk_text(page_text, chunk_size, overlap)):
            chunk_id = f"{os.path.basename(pdf_path)}::p{page_num}::c{idx}"
            chunks.append(Chunk(text=c, doc_path=pdf_path, page=page_num, chunk_id=chunk_id))
    return chunks

def load_config(cfg_path: str = "config.yaml") -> Dict:
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_embedder(model_name: str):
    return SentenceTransformer(model_name)

def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def build_or_rebuild_index(
    data_dir: str,
    artifacts_dir: str,
    embed_model_name: str,
    chunk_size: int,
    overlap: int,
    index_name: str = "faiss_index",
    meta_name: str = "chunks_meta.json",
):
    ensure_dirs(artifacts_dir)
    embedder = get_embedder(embed_model_name)
    dim = embedder.get_sentence_embedding_dimension()

    # Collect all PDFs
    pdfs = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.lower().endswith(".pdf")]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    all_chunks: List[Chunk] = []
    for pdf in tqdm(pdfs, desc="Reading PDFs"):
        all_chunks.extend(collect_pdf_chunks(pdf, chunk_size, overlap))

    texts = [c.text for c in all_chunks]
    embeddings = embedder.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    # Build FAISS index
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors + inner product
    index.add(embeddings.astype(np.float32))

    # Persist artifacts
    faiss_path = os.path.join(artifacts_dir, f"{index_name}.faiss")
    faiss.write_index(index, faiss_path)

    meta = [{
        "chunk_id": c.chunk_id,
        "doc_path": c.doc_path,
        "page": c.page,
    } for c in all_chunks]
    meta_path = os.path.join(artifacts_dir, meta_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta}, f, ensure_ascii=False, indent=2)

    # Save mapping texts (for quick retrieval)
    with open(os.path.join(artifacts_dir, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    return {
        "faiss_index_path": faiss_path,
        "meta_path": meta_path,
        "texts_path": os.path.join(artifacts_dir, "texts.pkl"),
        "num_chunks": len(all_chunks),
    }

class Retriever:
    def __init__(self, artifacts_dir: str, index_name: str, meta_name: str):
        self.artifacts_dir = artifacts_dir
        self.index = faiss.read_index(os.path.join(artifacts_dir, f"{index_name}.faiss"))
        with open(os.path.join(artifacts_dir, "texts.pkl"), "rb") as f:
            self.texts: List[str] = pickle.load(f)
        with open(os.path.join(artifacts_dir, meta_name), "r", encoding="utf-8") as f:
            self.meta = json.load(f)["meta"]

    def search(self, query: str, embedder, top_k: int = 5, score_threshold: float = 0.0):
        q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        D, I = self.index.search(q.astype(np.float32), top_k * 3)  # over-retrieve a bit
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: 
                continue
            if score_threshold and float(score) < score_threshold:
                continue
            results.append({
                "score": float(score),
                "text": self.texts[idx],
                "meta": self.meta[idx]
            })
            if len(results) >= top_k:
                break
        return results

def answer_with_openai(context: str, question: str) -> str:
    import os
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    system = "You are a helpful assistant. Answer only from the provided context. If the answer is not in the context, say you don't know."
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely:"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def answer_with_local_llm(context: str, question: str, model_name: str = "google/flan-t5-base") -> str:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok)
    prompt = (
        "You are a helpful assistant. Use only the context to answer. "
        "If the answer is not contained, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    out = pipe(prompt, max_new_tokens=256, temperature=0.2)
    return out[0]["generated_text"].strip()
