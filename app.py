# Fix sqlite3 cho chromadb trên Streamlit Cloud (sqlite hệ thống thường quá cũ)
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import time
import streamlit as st
import pypdf
import chromadb
from openai import OpenAI

# ---------- Cấu hình ----------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
LLM_MODEL = "gpt-4o-mini"                  # rẻ, đủ tốt (có thể đổi gpt-4o)
EMBED_MODEL = "text-embedding-3-small"     # embedding OpenAI (1536 chiều)

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
    st.session_state.setdefault(k, v)

# ---------- Lõi RAG ----------
def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def chunk_text(text, size=1000, overlap=200):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:                       # cắt nhỏ đoạn quá dài
            if cur:
                chunks.append(cur.strip()); cur = ""
            chunks.append(p[:size].strip())
            p = p[size - overlap:]
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def process_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    chunks = chunk_text(text)
    col = chromadb.Client().get_or_create_collection(f"rag_{int(time.time())}")
    col.add(ids=[str(i) for i in range(len(chunks))],
            documents=chunks, embeddings=embed(chunks))
    return col, len(chunks)

def rag(question, collection, k=4):
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    context = "\n\n".join(res["documents"][0])
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user",
                   "content": PROMPT.format(context=context, question=question)}],
        temperature=0)
    return resp.choices[0].message.content

# ---------- Giao diện ----------
st.set_page_config(page_title="RAG Chatbot (OpenAI)", layout="wide")
st.title("RAG Chatbot — hỏi đáp tài liệu PDF")

with st.sidebar:
    st.subheader("Tài liệu")
    f = st.file_uploader("Tải lên file PDF", type="pdf")
    if f and st.button("Xử lý PDF", use_container_width=True):
        with st.spinner("Đang xử lý..."):
            st.session_state.collection, n = process_pdf(f)
            st.session_state.pdf_name = f.name
            st.session_state.chat_history = []
        st.success(f"Đã index {n} đoạn")
    st.info(st.session_state.pdf_name or "Chưa có tài liệu")

for m in st.session_state.chat_history:
    st.chat_message(m["role"]).write(m["content"])

if st.session_state.collection is None:
    st.info("Tải lên và xử lý một PDF trước khi đặt câu hỏi.")
    st.chat_input("Nhập câu hỏi...", disabled=True)
else:
    if q := st.chat_input("Nhập câu hỏi của bạn..."):
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.chat_message("user").write(q)
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                ans = rag(q, st.session_state.collection)
                st.write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
