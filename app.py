"""Research RAG Chat — a focused interface for asking questions about PDFs."""

from __future__ import annotations

import io
import os
from html import escape
from dataclasses import dataclass

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


APP_NAME = "Research RAG Chat"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


@dataclass
class Chunk:
    text: str
    page: int


def make_chunks(pdf_bytes: bytes, chunk_size: int = 1_100, overlap: int = 180) -> list[Chunk]:
    """Extract pages from a PDF and split them into slightly overlapping passages."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ")
        text = " ".join(text.split())
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size]
            if len(piece.strip()) >= 80:
                chunks.append(Chunk(piece, page_number))
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Create embeddings in batches so larger papers stay within request limits."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 80):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=texts[start : start + 80]
        )
        vectors.extend(item.embedding for item in response.data)
    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


def build_index(client: OpenAI, pdf_bytes: bytes) -> tuple[faiss.Index, list[Chunk], int]:
    chunks = make_chunks(pdf_bytes)
    if not chunks:
        raise ValueError("No selectable text was found in this PDF.")
    vectors = embed_texts(client, [chunk.text for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks, len(PdfReader(io.BytesIO(pdf_bytes)).pages)


def search_paper(client: OpenAI, question: str, index: faiss.Index, chunks: list[Chunk]) -> list[Chunk]:
    vector = embed_texts(client, [question])
    _, matches = index.search(vector, min(5, len(chunks)))
    return [chunks[i] for i in matches[0] if i != -1]


def answer_question(client: OpenAI, question: str, history: list[dict], sources: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[Page {source.page}]\n{source.text}" for source in sources
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise research assistant. Answer only from the supplied paper "
                "context. If the context is insufficient, say so plainly. Cite statements with "
                "[Page N] markers. Be concise, but explain technical ideas clearly."
            ),
        },
        *history[-8:],
        {"role": "user", "content": f"Paper context:\n{context}\n\nQuestion: {question}"},
    ]
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or "I couldn't generate an answer."


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_sources = []


st.set_page_config(page_title=APP_NAME, page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .stApp { background: #f6f7fb; color: #18212f; }
      [data-testid="stSidebar"] { background: #111827; }
      [data-testid="stSidebar"] * { color: #e8edf7; }
      .block-container { max-width: 1120px; padding-top: 2.6rem; padding-bottom: 2rem; }
      .eyebrow { color: #6366f1; font-size: .76rem; font-weight: 750; letter-spacing: .11em; text-transform: uppercase; }
      .hero-title { font-size: 2.35rem; font-weight: 760; letter-spacing: -.045em; margin: .25rem 0 .45rem; }
      .hero-copy { color: #64748b; font-size: 1.03rem; margin: 0 0 1.8rem; }
      .paper-card { background: white; border: 1px solid #e6e9f0; border-radius: 16px; padding: 1rem 1.15rem; box-shadow: 0 3px 12px rgba(15,23,42,.035); }
      .source-card { background: #f8faff; border-left: 3px solid #818cf8; border-radius: 0 9px 9px 0; padding: .65rem .8rem; margin: .45rem 0; font-size: .87rem; color: #475569; }
      .metric { color: #94a3b8; font-size: .78rem; }
      .stChatMessage { border-radius: 14px; }
      .stChatInputContainer { border-top: 0; }
      .stButton > button { border-radius: 9px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, value in {"messages": [], "index": None, "chunks": [], "paper_name": None, "page_count": 0, "last_sources": []}.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.markdown("## ◈ Research RAG")
    st.caption("Your private reading companion")
    st.divider()
    api_key = st.text_input(
        "OpenAI API key", value=os.getenv("OPENAI_API_KEY", ""), type="password", help="Used only for this session."
    )
    uploaded_file = st.file_uploader("Add a research paper", type=["pdf"], label_visibility="visible")
    if uploaded_file:
        st.caption(f"Ready to index: **{uploaded_file.name}**")
    process = st.button("Process paper", use_container_width=True, type="primary", disabled=not (uploaded_file and api_key))
    st.divider()
    if st.session_state.paper_name:
        st.markdown("**Active paper**")
        st.markdown(f"📄 {st.session_state.paper_name}")
        st.caption(f"{st.session_state.page_count} pages · {len(st.session_state.chunks)} passages")
        st.button("Start a new chat", use_container_width=True, on_click=reset_chat)
    else:
        st.caption("Upload a PDF and process it to begin.")
    st.divider()
    st.caption("Answers are grounded in the indexed paper and include page references.")

if process:
    try:
        client = OpenAI(api_key=api_key)
        with st.spinner("Reading the paper and creating its research index…"):
            index, chunks, page_count = build_index(client, uploaded_file.getvalue())
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.paper_name = uploaded_file.name
        st.session_state.page_count = page_count
        reset_chat()
        st.rerun()
    except Exception as error:
        st.error(f"The paper could not be processed: {error}")

main, context_panel = st.columns([1.65, 0.75], gap="large")
with main:
    st.markdown('<div class="eyebrow">Paper intelligence, without the scavenger hunt</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Ask better questions of your research.</div>', unsafe_allow_html=True)
    if st.session_state.paper_name:
        st.markdown(
            f'<div class="paper-card">📄 <b>{escape(st.session_state.paper_name)}</b><br><span class="metric">Indexed and ready for conversation · {len(st.session_state.chunks)} retrievable passages</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<p class="hero-copy">Upload a PDF, let the app build a semantic index, and explore the paper through grounded conversation.</p>', unsafe_allow_html=True)

    if not st.session_state.paper_name:
        st.info("Start in the sidebar: enter an OpenAI API key, upload a PDF, then select **Process paper**.")
        st.markdown("#### Useful questions to try")
        st.write("• What problem does this paper solve?  \n• Summarize the methodology.  \n• What are the key limitations?  \n• Which results best support the conclusion?")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask about the paper…")
        if question:
            if not api_key:
                st.warning("Enter your OpenAI API key in the sidebar to continue.")
            else:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Finding evidence in the paper…"):
                        try:
                            client = OpenAI(api_key=api_key)
                            sources = search_paper(client, question, st.session_state.index, st.session_state.chunks)
                            answer = answer_question(client, question, st.session_state.messages[:-1], sources)
                            st.markdown(answer)
                            st.session_state.last_sources = sources
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as error:
                            st.error(f"I couldn't answer that question: {error}")

with context_panel:
    st.markdown("#### Evidence desk")
    if st.session_state.last_sources:
        st.caption("The passages used for the most recent answer")
        for source in st.session_state.last_sources[:3]:
            preview = source.text[:230].rsplit(" ", 1)[0] + "…"
            st.markdown(f'<div class="source-card"><b>Page {source.page}</b><br>{escape(preview)}</div>', unsafe_allow_html=True)
    elif st.session_state.paper_name:
        st.caption("Ask a question to see the supporting passages here.")
    else:
        st.caption("Relevant passages will appear here, so you can verify each answer against the original paper.")
