from typing import List, Dict
import streamlit as st

class Source:
    def __init__(self, source: Dict):
        self.source = source

    def render(self):
        st.markdown(f"- {self.source['title']}: {self.source['content'][:100] + "..."}")