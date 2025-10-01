from typing import Dict

from core.utils.pipeline_utils import (
    load_config_yaml,
    initialize_all_parsers,
    initialize_chunker,
    initialize_embedder,
    initialize_vector_store,
    initialize_retriever,
    initialize_generator_app
)

from core.pipelines.UIPipeline import UIPipeline

def setup_pipeline(app_config: Dict, config_path: str) -> UIPipeline:
    '''
    Take an app config dictionary and a YAML config filepath string to initialize a UI pipeline object
    Return a UI pipeline object
    '''
    
    # Read configs
    configs_base_dir = f"{config_path}/"
    parsers_config = load_config_yaml(configs_base_dir, "parsers")
    chunker_config = load_config_yaml(configs_base_dir, "chunker")
    embedder_config = load_config_yaml(configs_base_dir, "embedder")
    vector_store_config = load_config_yaml(configs_base_dir, "vector_store")
    retriever_config = load_config_yaml(configs_base_dir, "retriever")
    generator_config = load_config_yaml(configs_base_dir, "generator")

    # Initialize all parsers
    all_parsers = initialize_all_parsers(parsers_config)
    
    # Initialize chunker
    chunker = initialize_chunker(chunker_config)

    # # Initialize embedder
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

    pipeline = UIPipeline(
        parsers=all_parsers,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        generator=generator
    )

    return pipeline
