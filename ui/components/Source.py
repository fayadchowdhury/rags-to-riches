from typing import List, Dict
import streamlit as st

class Source:
    def __init__(self, source: Dict):
        '''
        Initialize Source UI object with
        - source
        '''
        self.source = source

    def render(self):
        '''
        Render
        - title
        - content
        '''
        st.markdown(f"- {self.source['title']}: {self.source['content'][:100] + "..."}")