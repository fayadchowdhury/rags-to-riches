from typing import Dict, Callable
import streamlit as st

from ui.components.ChatMessage import ChatMessage

class Chat:
    def __init__(self, session: Dict, handle_prompt_submit: Callable, handle_file_upload: Callable):
        '''
        Initialize a Chat UI object with
        - session
        - handle_prompt_submit - a callback to handle submitting prompt input
        '''
        self.session = session
        self.handle_prompt_submit = handle_prompt_submit
        self.handle_file_upload = handle_file_upload
        pass

    def render(self):
        '''
        Render
        - chat title
        - all messages
        - input area
        '''
        st.markdown(f"## {self.session['name']}")
    
        for message in self.session.get("messages", []):
            ChatMessage(message).render()

        if user_input := st.chat_input("Type your message here and/or attach a file...", accept_file="multiple", file_type=["pdf", "csv", "html", "ipynb"]):
            with st.chat_message("user"):
                st.markdown(user_input["text"])
                self.handle_file_upload(user_input["files"])

            self.handle_prompt_submit(user_input["text"])