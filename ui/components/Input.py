from typing import Callable
import streamlit as st

class Input:
    def __init__(self, handle_prompt_submit: Callable):
        self.handle_prompt_submit = handle_prompt_submit
        pass

    def render(self):
        if user_input := st.chat_input("Type your message here..."):
            self.handle_prompt_submit(user_input)