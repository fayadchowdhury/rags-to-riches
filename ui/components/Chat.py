from typing import Dict, Callable
import streamlit as st

from ui.components.ChatMessage import ChatMessage
from ui.components.Input import Input

class Chat:
    def __init__(self, session: Dict, handle_prompt_submit: Callable):
        self.session = session
        self.handle_prompt_submit = handle_prompt_submit
        pass

    def render(self):
        Input(self.handle_prompt_submit).render()
    
        st.markdown(f"## Chat Session: {self.session['name']}")
        for message in self.session.get("messages", []):
            ChatMessage(message).render()
    