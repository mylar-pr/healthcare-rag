"""Streamlit chat UI for the Healthcare RAG pipeline."""
import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Healthcare Benefits Assistant", page_icon="🏥")
st.title("🏥 Healthcare Benefits Assistant")
st.caption("Ask questions about your benefits, PTO policy, or health programs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("e.g. What is the copay for generic drugs?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json={"question": prompt},
                    timeout=30,
                )
                response.raise_for_status()
                answer = response.json()["answer"]
            except httpx.ConnectError:
                answer = "Cannot connect to the API. Make sure the server is running:\n\n```\npython -m uvicorn src.api:app --reload\n```"
            except Exception as e:
                answer = f"Error: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
