from typing import Dict, Callable
import streamlit as st

from ui.components.ChatMessage import ChatMessage

class Chat:
    def __init__(self, session: Dict, handle_prompt_submit: Callable):
        self.session = session
        self.handle_prompt_submit = handle_prompt_submit
        pass

    def render(self):
        st.markdown(f"## {self.session['name']}")
    
        for message in self.session.get("messages", []):
            ChatMessage(message).render()

        if user_input := st.chat_input("Type your message here..."):
            with st.chat_message("user"):
                st.markdown(user_input)

            self.handle_prompt_submit(user_input)