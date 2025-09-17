import yaml
from dotenv import load_dotenv
import os
from typing import Dict

from core.parsers.BaseParser import BaseParser
from core.parsers.PdfParser import PdfParser
from core.parsers.HtmlParser import HtmlParser
from core.parsers.QACsvParser import QACsvParser
from core.parsers.NotebookParser import NotebookParser

from core.chunkers.BaseChunker import BaseChunker
from core.chunkers.FixedTokenSizeChunker import FixedTokenSizeChunker

from core.embedders.BaseEmbedder import BaseEmbedder
from core.embedders.OpenAIEmbedder import OpenAIEmbedder

from core.vector_stores.BaseVectorStore import BaseVectorStore
from core.vector_stores.ChromaVectorStore import ChromaVectorStore
from core.vector_stores.PineconeVectorStore import PineconeVectorStore

from core.retrievers.BaseRetriever import BaseRetriever
from core.retrievers.TopKRetriever import TopKRetriever

from core.generators.BaseGenerator import BaseGenerator
from core.generators.OpenAIGenerator import OpenAIGenerator

def get_env_config(path: str) -> Dict:
    load_dotenv(path)
    return {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY", ""),
    }

def load_config_yaml(base_dir: str, type: str) -> Dict:
    '''
    Take base directory and type of config YAML
    Return config dictionary
    '''
    with open(f"{base_dir}/{type}.yaml", "r") as f:
        return yaml.safe_load(f)
    
def read_prompt(base_dir: str, type: str) -> str:
    '''
    Take base directory and type of prompt
    Return prompt string
    '''
    with open(f"{base_dir}/{type}.txt", "r") as f:
        return f.read()

def initialize_parser(type: str, parser_config: Dict) -> BaseParser:
    '''
    Take a type and a parser config
    Return a parser
    '''
    parser = None
    if type == "pdf":
        parser = PdfParser(**parser_config)
    elif type == "html":
        parser = HtmlParser(**parser_config)
    elif type == "csv":
        parser = QACsvParser(**parser_config)
    elif type == "ipynb":
        parser = NotebookParser(**parser_config)

    return parser

def initialize_all_parsers(parsers_config: Dict) -> Dict:
    all_parsers = {}
    for file_type, config in parsers_config.items():
        all_parsers[file_type] = initialize_parser(file_type, config.get("config", {}))

    return all_parsers

def parser_router(all_parsers: Dict, path: str) -> BaseParser:
    '''
    Take a file path and all parsers
    Return a parser
    '''
    extension = path.split(".")[-1]
    parser = all_parsers.get(extension, None)

    return parser

def initialize_chunker(chunker_config: Dict) -> BaseChunker:
    '''
    Take chunker config
    Return a chunker
    '''
    chunker = None
    config = chunker_config.get("config", {})
    type = chunker_config.get("type", "")
    if type == "FixedTokenSizeChunker":
        chunker = FixedTokenSizeChunker(**config)

    return chunker

def initialize_embedder(embedder_config: Dict) -> BaseEmbedder:
    '''
    Take embedder config
    Return an embedder
    '''
    embedder = None
    type = embedder_config.get("type", "")
    config = embedder_config.get("config", {})
    if type == "OpenAIEmbedder":
        embedder = OpenAIEmbedder(**config)

    return embedder

def initialize_vector_store(vector_store_config: Dict) -> BaseVectorStore:
    '''
    Take a type and vector store config
    Return a vector store
    '''
    vector_store = None
    config = vector_store_config.get("config", {})
    type = vector_store_config.get("type", "")
    if type == "PineconeVectorStore":
        vector_store = PineconeVectorStore(**config)
    elif type == "ChromaVectorStore":
        vector_store = ChromaVectorStore(**config)

    return vector_store

def initialize_retriever(embedder: BaseEmbedder, vector_store: BaseVectorStore, retriever_config: Dict) -> BaseRetriever:
    '''
    Take a type and retriever config
    Return a retriever
    '''
    retriever = None
    config = retriever_config.get("config", {})
    type = retriever_config.get("type", "")
    if type == "TopKRetriever":
        retriever = TopKRetriever(embedder, vector_store, **config)

    return retriever

def initialize_generator(system_prompt: str, prompt_template: str, generator_config: Dict) -> BaseGenerator:
    '''
    Take a type and generator config
    Return a generator
    '''
    generator = None
    config = generator_config.get("config", {})
    type = generator_config.get("type", "")
    generator_config["system_prompt"] = system_prompt
    generator_config["prompt_template"] = prompt_template
    if type == "OpenAIGenerator":
        generator = OpenAIGenerator(**config)

    return generator