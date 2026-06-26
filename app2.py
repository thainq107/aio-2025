import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
LLM_MODEL = "gpt-4o-mini"   # có thể đổi gpt-4o

SYSTEM = {"role": "system",
          "content": "Bạn là trợ lý hữu ích, trả lời ngắn gọn, chính xác, bằng tiếng Việt."}

st.set_page_config(page_title="Chatbot đơn giản (OpenAI)")
st.title("Chatbot đơn giản (OpenAI)")

# Lưu lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# Ô nhập câu hỏi
if q := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": q})
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        with st.spinner("Đang trả lời..."):
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[SYSTEM] + st.session_state.messages,
                temperature=0)
            ans = resp.choices[0].message.content
            st.write(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
