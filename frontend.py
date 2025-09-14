import streamlit as st
from openai import OpenAI
import pinecone
from typing import List, Dict
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]  

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with hardcoded API keys and configurations."""
        try:
            logger.info("Initializing OpenAI client...")
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            logger.info("Initializing Pinecone client...")
            self.pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
            
            # List available indexes for debugging
            indexes = self.pinecone_client.list_indexes()
            logger.info(f"Available Pinecone indexes: {[index.name for index in indexes]}")
            
            logger.info(f"Attempting to connect to index: {PINECONE_INDEX_NAME}")
            self.index = self.pinecone_client.Index(PINECONE_INDEX_NAME)
            
            # Test index connection with a simple query
            logger.info("Testing index connection...")
            test_vector = [0.0] * 1536  # Standard dimension for OpenAI embeddings
            self.index.query(vector=test_vector, top_k=1)
            logger.info("Successfully connected to Pinecone index")
            
            self.embedding_model = "text-embedding-ada-002"
            self.chat_model = "gpt-3.5-turbo"
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    def create_embedding(self, text: str) -> List[float]:
        """Create embeddings for input text using OpenAI's API."""
        try:
            logger.debug(f"Creating embedding for text: {text[:100]}...")
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            logger.debug("Successfully created embedding")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}", exc_info=True)
            raise

    def retrieve_from_pinecone(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve relevant information from Pinecone index."""
        try:
            logger.debug(f"Creating embedding for query: {query}")
            query_embedding = self.create_embedding(query)
            
            logger.debug("Querying Pinecone index...")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            source_knowledge = []
            for match in results.matches:
                source_info = {
                    'source': match.metadata.get('file_name', 'Unknown'),
                    'context': match.metadata.get('first_10_words', 'No context available'),
                    'score': f"{match.score:.4f}"
                }
                source_knowledge.append(source_info)
            
            logger.debug(f"Retrieved {len(source_knowledge)} results from Pinecone")
            return source_knowledge
            
        except Exception as e:
            logger.error(f"Error in retrieve_from_pinecone: {str(e)}", exc_info=True)
            raise

    def generate_augmented_prompt(self, query: str, source_knowledge: List[Dict[str, str]]) -> str:
        """Generate an augmented prompt using retrieved context."""
        sources_text = "\n\n".join([
            f"Source: {source['source']}\nContext: {source['context']}"
            for source in source_knowledge
        ])
        
        augmented_prompt = f"""You are an NLP expert. Based on the query provided and the relevant context retrieved from the knowledge base, provide:
1. A direct answer that addresses the specific query
2. If present, give evidence/examples from the relevant benchmark mentioned in the query
3. If needed, Technical explanation of the scoring criteria as shown in the course materials
4. References to any specific datasets or metrics mentioned in the retrieved lecture content

If the provided information is insufficient to answer the query, state this clearly and explain what additional information would be needed.

Relevant Course Materials:
{sources_text}

Query: {query}

Answer:"""
        
        return augmented_prompt

    def get_response(self, query: str) -> tuple[str, List[Dict[str, str]]]:
        """Get a response from the LLM using the augmented prompt."""
        try:
            # First retrieve relevant sources
            source_knowledge = self.retrieve_from_pinecone(query)
            
            # Generate the prompt with the sources
            prompt = self.generate_augmented_prompt(query, source_knowledge)
            
            # Get the response from OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert NLP teaching assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content, source_knowledge
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise

def initialize_rag_system():
    """Initialize the RAG system if not already in session state."""
    if 'rag_system' not in st.session_state:
        try:
            logger.info("Creating new RAG system instance...")
            st.session_state.rag_system = RAGSystem()
            logger.info("RAG system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}", exc_info=True)
            st.error(f"""
            Error initializing RAG system. Debug information:
            - Available indexes: {[index.name for index in pinecone.Pinecone(api_key=PINECONE_API_KEY).list_indexes()]}
            - Attempted index name: {PINECONE_INDEX_NAME}
            - Error message: {str(e)}
            
            Please check:
            1. API keys are correct
            2. Index name matches exactly
            3. Index permissions are properly set
            """)
            return False
    return True
def main():
    st.set_page_config(page_title="NLP Course Assistant", page_icon="🎓", layout="wide")
    load_dotenv()

    # Page header
    st.title("🎓 NLP Course Assistant")
    st.markdown("""
    Ask questions about NLP concepts, benchmarks, and course materials.
    The assistant will provide answers based on the course content and relevant sources.
    """)
    
    # Initialize RAG system
    if not initialize_rag_system():
        return
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="E.g., Explain the importance of MMLU benchmark in evaluating language models..."
    )
    
    # Submit button
    if st.button("Submit Question", type="primary"):
        if not query:
            st.warning("Please enter a question!")
            return
            
        with st.spinner("Searching knowledge base and generating response..."):
            try:
                # Get response and sources
                response, sources = st.session_state.rag_system.get_response(query)
                
                # Display response
                st.markdown("### Answer")
                st.markdown(response)
                
                # Display sources
                st.markdown("### Reference Sources")
                for source in sources:
                    with st.expander(f"Source: {source['source']}"):
                        st.markdown(f"**Context:** {source['context']}")
                        st.markdown(f"**Relevance Score:** {source['score']}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()