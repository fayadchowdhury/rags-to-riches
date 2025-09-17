import sys
from pathlib import Path

from core.utils import (
    load_config_yaml,
    initialize_all_parsers,
    initialize_chunker,
    initialize_embedder,
    initialize_vector_store,
    initialize_retriever,
    initialize_generator,
    parser_router,
    get_env_config,
    read_prompt
)

if __name__=="__main__":

    env_config = get_env_config(".env")

    experiment_name = sys.argv[1]
    
    # # Read configs
    configs_base_dir = f"experiments/configs/{experiment_name}"
    experiment_config = load_config_yaml(configs_base_dir, "experiment")
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

    # Initialize embedder
    embedder_config["config"]["api_key"] = env_config.get(embedder_config["config"].get("api_key", ""), "")
    embedder = initialize_embedder(embedder_config)

    # Initialize vector store
    vector_store_config["config"]["api_key"] = env_config.get(vector_store_config["config"].get("api_key", ""), "")
    vector_store = initialize_vector_store(vector_store_config)

    # Initialize retriever
    retriever = initialize_retriever(embedder, vector_store, retriever_config)

    # Read prompts
    prompts_base_dir = f"experiments/prompts/{experiment_name}"
    prompt_template = read_prompt(prompts_base_dir, "qa")
    system_prompt = read_prompt(prompts_base_dir, "system")
    
    # Initialize generator
    generator_config["config"]["api_key"] = env_config.get(generator_config["config"].get("api_key", ""), "")
    generator = initialize_generator(system_prompt, prompt_template, generator_config)

    # Loop over data files
    data_path = Path(experiment_config["data"])
    for file_path in data_path.rglob("*"):
        if file_path.is_file():
            file_path = str(file_path)
            print(f"Working on: {file_path}")
            parser = parser_router(all_parsers, file_path)
            parsed_document = parser.parse(file_path)
            print(f"Finished parsing")
            print(f"Starting chunking")
            chunks = chunker.chunk(parsed_document)
            print(f"Finished chunking")
            print(f"Starting embedding")
            embeddings = embedder.embed_data(chunks)
            print(f"Finished embedding")
            print(f"Pushing to vector store")
            vector_store.store_batch(embeddings, batch_size=10)
            print(f"Finished pushing to vector store")
    
    # Get queries
    query_pass = experiment_config["query_pass"]
    query_fail = experiment_config["query_fail"]

    # Retrieve documents
    query_pass_docs = retriever.retrieve(query_pass)
    query_fail_docs = retriever.retrieve(query_fail)

    # Generate response
    response_pass = generator.generate(query_pass, query_pass_docs)
    response_fail = generator.generate(query_fail, query_fail_docs)
    
    print(f"PASS:")
    print(response_pass)

    print(f"FAIL:")
    print(response_fail)