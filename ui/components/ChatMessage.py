from typing import List, Dict, Iterable
import streamlit as st

from ui.components.Source import Source

class ChatMessage:
    def __init__(self, message: Dict):
        self.message = message

    def render(self):
        with st.chat_message(self.message["role"]):
            st.markdown(self.message["content"])
            if "sources" in self.message:
                for source in self.message["sources"]:
                    Source.render(source)

    @staticmethod # So we can call without instantiating the class
    def render_stream(token_stream: Iterable[str], sources: List[Dict]) -> str:
        full_text = ""
        with st.chat_message("assistant"): # Only assistant responses are streamed anyway
            placeholder = st.empty()
            for token in token_stream:
                full_text += token
                placeholder.markdown(full_text)
            if sources:
                for source in sources:
                    Source(source).render()
        return full_text