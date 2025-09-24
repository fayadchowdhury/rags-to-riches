from typing import Dict
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
                    Source.render_sources(source)