from typing import List, Dict, Callable
import streamlit as st

class Sidebar:
    def __init__(self, sessions: List[Dict], handle_new_chat_click: Callable, handle_session_click: Callable, handle_clear_chat_click: Callable):
        self.sessions = sessions
        self.handle_new_chat_click = handle_new_chat_click
        self.handle_session_click = handle_session_click
        self.handle_clear_chat_click = handle_clear_chat_click

    def render(self):
        with st.sidebar:
            if st.button("New Chat"):
                self.handle_new_chat_click()
            st.markdown("## Chat sessions")
            if self.sessions:
                for session in self.sessions:
                    st.button(session["name"], key=session["id"], on_click=lambda s=session: self.handle_session_click(s))
            if st.button("Clear current chat"):
                self.handle_clear_chat_click()