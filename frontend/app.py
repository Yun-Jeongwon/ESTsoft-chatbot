"""Streamlit frontend for the ESTsoft chatbot."""

from __future__ import annotations

import json
from typing import Any, Dict

import requests
import streamlit as st

API_ENDPOINT = "http://localhost:8000/query"
DEFAULT_TIMEOUT = 30


def post_query(query: str) -> Dict[str, Any]:
    """Send the query to the backend and return the parsed JSON response."""
    response = requests.post(
        API_ENDPOINT,
        json={"query": query},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError("백엔드 응답을 JSON으로 파싱할 수 없습니다.") from exc

    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise ValueError("백엔드 응답 형식이 올바르지 않습니다.")

    return payload


def init_page() -> None:
    st.set_page_config(page_title="ESTsoft Perso.ai 챗봇", page_icon="💬")
    header = st.container()
    header.title("💬 Perso.ai 챗봇")
    header.subheader(": Perso.ai 바이브코딩(미래내일일경험 인턴십) 과제")
    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.875rem;
            background-color: white;
            z-index: 999;
        }
        .fixed-header {
            border-bottom: 1px solid black;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_submission() -> None:
    user_query = st.chat_input("질문을 입력하세요")

    if user_query is None:
        return

    cleaned_query = user_query.strip()
    if not cleaned_query:
        st.warning("질문을 입력해 주세요.")
        return

    st.session_state.messages.append({"role": "user", "content": cleaned_query})
    
    with st.chat_message("user"):
        st.markdown(cleaned_query)

    try:
        with st.spinner("답변 생성 중..."):
            payload = post_query(cleaned_query)
    except requests.exceptions.HTTPError as exc:
        st.error(f"백엔드 요청이 실패했습니다: {exc}")
        return
    except requests.exceptions.RequestException as exc:
        st.error(f"백엔드에 연결할 수 없습니다: {exc}")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    answer = payload.get("answer", "").strip()
    if not answer:
        st.warning("백엔드에서 유효한 답변을 받지 못했습니다.")
        return

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)



def main() -> None:
    init_page()
    render_chat()
    handle_submission()

if __name__ == "__main__":
    main()

    