import streamlit as st
import logging
from datetime import datetime

from ui.components.Sidebar import Sidebar
from ui.components.Chat import Chat
from ui.components.ChatMessage import ChatMessage

from ui.functions.messages import get_messages_for_session_from_db, save_message_for_session_to_db, delete_messages_for_session_from_db
from ui.functions.sessions import get_session_from_db, get_sessions_from_db, create_new_session, save_session_to_db

from core.utils.app_utils import initialize_database, get_app_env_config
from core.utils.pipeline_utils import load_config_yaml, initialize_all_parsers, initialize_chunker, initialize_embedder, initialize_vector_store, initialize_retriever, initialize_generator_app
from core.utils.logging_utils import setup_logger

from core.pipelines.UIPipeline import UIPipeline

logger = setup_logger("app", "logs", logging.DEBUG)

def get_chat_history(messages, token_limit):
    token_count = 0
    history = []
    for message in reversed(messages):
        content = message["content"]
        role = message["role"]
        token_count += len(content.split())
        # Check to see if within token_limit
        if token_count > token_limit:
            logger.debug(f"Exceeding token count, stopping history here")
            break
        else:
            history.append({
                "role": role,
                "content": content
            })
    return [c for c in reversed(history)]


def handle_prompt_submit(db, pipeline, prompt: str):
    query = {"role": "user", "content": prompt}
    logger.debug(f"Saving to db user query:\n{query}")
    save_message_for_session_to_db(db, st.session_state["current_session"]["id"], query)
    st.session_state["current_session"]["messages"].append(query)
    logger.debug(f"Saved query to db for session id: {st.session_state['current_session']['id']}")

    history = get_chat_history(st.session_state["current_session"]["messages"], token_limit=5000)

    prompt_template = "{query}:\n\n{context}:\n\n"
    response_stream = pipeline.query(prompt_template, prompt, history)
    logger.debug(f"Sending query to LLM to stream response")
    response = ChatMessage.render_stream(response_stream)

    reply = {"role": "assistant", "content": response}
    logger.debug(f"Done streaming response:\n{reply}")
    save_message_for_session_to_db(db, st.session_state["current_session"]["id"], reply)
    st.session_state["current_session"]["messages"].append(reply)

    logger.debug(f"Saved response to db for session id: {st.session_state['current_session']['id']}")

    logger.debug(f"Chat context: {get_chat_history(st.session_state["current_session"]["messages"], token_limit=1500)}")


def handle_new_chat_click(db):
    logger.debug(f"Creating new session")
    new_session = create_new_session(f"Session @ {datetime.now().strftime("%H:%M:%S-%d-%m-%Y")}") # Need to figure this out somehow, maybe update with summary of first query??
    session_id = save_session_to_db(db, new_session)
    logger.debug(f"Saved new session to db with id: {session_id}")
    messages = get_messages_for_session_from_db(db, session_id)
    
    new_session["id"] = session_id
    new_session["messages"] = messages

    logger.debug(f"Placing new session into 0th at session_state sessions and setting current_session to new session and setting messages to retrieved messages")
    st.session_state["sessions"].insert(0, new_session)
    st.session_state["current_session"] = new_session
    st.session_state["messages"] = messages
    logger.debug(f"Done")

def handle_session_click(db, session):
    logger.debug(f"Fetching session {session["name"]} with id: {session["id"]}")
    new_session = get_session_from_db(db, session["id"])
    messages = get_messages_for_session_from_db(db, session["id"])
    
    new_session["messages"] = messages
    logger.debug(f"Fetched {len(messages)} messages for session from db")

    logger.debug(f"Placing {new_session["name"]} into 0th at session_state sessions and setting current_session to {new_session["name"]} and setting messages to retrieved messages")
    st.session_state["sessions"] = [x for x in st.session_state["sessions"] if x["id"] != new_session["id"]]
    st.session_state["sessions"].insert(0, new_session)
    st.session_state["current_session"] = new_session
    st.session_state["messages"] = messages
    logger.debug(f"Done")


def handle_clear_chat_click(db, session_id):
    logger.debug(f"Deleting all messages in chat from database")
    delete_messages_for_session_from_db(db, session_id)
    logger.debug(f"Deleted messages")
    messages = get_messages_for_session_from_db(db, session_id)
    logger.debug(f"Fetched {len(messages)} messages for session from db")
    logger.debug(f"Updating session_state current_session and messages")
    st.session_state["current_session"]["messages"] = messages
    st.session_state["messages"] = messages



def main():
    logger.debug(f"Starting main function with session_state: {st.session_state}")
    
    # Read app environment variables 
    app_config = get_app_env_config(".env")
    app_config["db_type"] = "SQLiteDatabase"
    logger.debug(f"App config: {app_config}")

    # Read configs
    logger.debug(f"Reading config YAMLs from configs directory")
    configs_base_dir = f"config/"
    parsers_config = load_config_yaml(configs_base_dir, "parsers")
    chunker_config = load_config_yaml(configs_base_dir, "chunker")
    embedder_config = load_config_yaml(configs_base_dir, "embedder")
    vector_store_config = load_config_yaml(configs_base_dir, "vector_store")
    retriever_config = load_config_yaml(configs_base_dir, "retriever")
    generator_config = load_config_yaml(configs_base_dir, "generator")
    
    logger.debug(f"Initializing components - parsers, chunker, embedder, vector_store, retriever, generator")
    # Initialize all parsers
    all_parsers = initialize_all_parsers(parsers_config)

    # Initialize chunker
    chunker = initialize_chunker(chunker_config)

    # Initialize embedder
    embedder_config["config"]["api_key"] = app_config.get(embedder_config["config"].get("api_key", ""), "")
    embedder = initialize_embedder(embedder_config)

    # Initialize vector store
    vector_store_config["config"]["api_key"] = app_config.get(vector_store_config["config"].get("api_key", ""), "")
    vector_store = initialize_vector_store(vector_store_config)

    # Initialize retriever
    retriever = initialize_retriever(embedder, vector_store, retriever_config)
    
    # Initialize generator
    generator_config["config"]["api_key"] = app_config.get(generator_config["config"].get("api_key", ""), "")
    generator = initialize_generator_app(generator_config)

    # Initialize pipeline
    if not "pipeline" in st.session_state:
        logger.debug(f"No pipeline found; initializing and saving to session_state")
        pipeline = UIPipeline(
            all_parsers,
            chunker,
            embedder,
            vector_store,
            retriever,
            generator
        )
        st.session_state["pipeline"] = pipeline
    else:
        logger.debug(f"Using pipeline with:\n{st.session_state["pipeline"].list_components()}")

    # Initialize db and create tables
    db = initialize_database(app_config)
    db._create_tables()
    logger.debug(f"Initialized database")

    # Set up UI metadata
    st.set_page_config(
        page_title="NLP Course Assistant",
        page_icon="🤖",
        layout="wide",
    )
    st.title("NLP Course Assistant 🤖")
    logger.debug(f"Set up Streamlit page config")
    
    # Check to see if sessions exist in session_state
    # Get from db if not
    logger.debug(f"Checking for 'sessions' in session_state")
    if "sessions" not in st.session_state:
        logger.debug(f"'sessions' not found in session_state, fetching from db")
        st.session_state["sessions"] = get_sessions_from_db(db)
        logger.debug(f"Fetched {len(st.session_state['sessions'])} sessions from db")

    # Start at most recent session if found in db
    # Or create a temporary session
    logger.debug(f"Checking for 'current_session' in session_state")
    if "current_session" not in st.session_state:
        logger.debug(f"'current_session' not found in session_state")
        if st.session_state["sessions"] and len(st.session_state["sessions"]) > 0:
            logger.debug(f"Setting current_session to most recent")
            st.session_state["current_session"] = st.session_state["sessions"][0]
        else:
            logger.debug(f"No sessions found in db, creating temporary session")
            st.session_state["current_session"] = create_new_session("Temporary session")
            st.session_state["sessions"].insert(0, st.session_state["current_session"])
            logger.debug(f"Inserted temporary session into sessions list")
        
        logger.debug(f"Current session set to: {st.session_state['current_session']}")
        
        # Save current session to db and update id in current_session
        session_id = save_session_to_db(db, st.session_state["current_session"])
        logger.debug(f"Saved current_session to db with id: {session_id}")
        st.session_state["current_session"]["id"] = session_id
        st.session_state["sessions"][0]["id"] = session_id
        logger.debug(f"Updated current_session and sessions[0] in session_state with id: {st.session_state['current_session']['id']}")
        

        # Initialize messages in session state
        # Replace with actual messages fetched from store
        # If no messages, initialize to empty list
        if "messages_loaded" not in st.session_state:
            logger.debug(f"Loading messages for current_session from db")
            messages = get_messages_for_session_from_db(db, session_id=st.session_state["current_session"].get("id", None))
            if not messages:
                logger.debug(f"No messages found for current_session, initializing to empty list")
                messages = []
            logger.debug(f"Loaded {len(messages)} messages for current_session")
            st.session_state["current_session"]["messages"] = messages
            st.session_state["messages_loaded"] = True

    # Render sidebar component (new chat button, session buttons, clear chat button)
    logger.debug(f"Rendering Sidebar component with sessions: {st.session_state['sessions']}")
    sidebar = Sidebar(
        sessions=st.session_state["sessions"],
        handle_new_chat_click=lambda: handle_new_chat_click(db),
        handle_session_click=lambda s: handle_session_click(db, s),
        handle_clear_chat_click=lambda: handle_clear_chat_click(db, st.session_state["current_session"]["id"])
    )
    sidebar.render()

    # Render chat component (title, messages, input)
    logger.debug(f"Rendering Chat component with session: {st.session_state['current_session']}")
    chat = Chat(
        session=st.session_state["current_session"],
        handle_prompt_submit=lambda prompt: handle_prompt_submit(db, st.session_state["pipeline"], prompt)
    )
    chat.render()
    
    # Think of more elegant way to close connection if at all required
    # db.close()

if __name__ == "__main__":
    main()