import logging
import sys
from pathlib import Path
import json

import mlflow

from core.utils.pipeline_utils import (
    load_config_yaml,
    initialize_all_parsers,
    initialize_chunker,
    initialize_embedder,
    initialize_vector_store,
    initialize_retriever,
    initialize_generator,
    parser_router,
    get_env_config,
    read_prompt,
    check_for_embeddings,
    save_embeddings,
    load_embeddings,
    save_config_yaml,
    save_results_json
)

from core.utils.logging_utils import setup_logger

if __name__=="__main__":

    env_config = get_env_config(".env")

    experiment_name = sys.argv[1]
    mlflow_uri = sys.argv[2] if len(sys.argv) > 2 else None

    logger = setup_logger(experiment_name, f"experiments/logs/{experiment_name}", logging.DEBUG)
    
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
    prompt_template = read_prompt(prompts_base_dir, generator_config["config"]["qa_prompt_path"])
    system_prompt = read_prompt(prompts_base_dir, generator_config["config"]["system_prompt_path"])
    
    # Initialize generator
    generator_config["config"]["api_key"] = env_config.get(generator_config["config"].get("api_key", ""), "")
    generator = initialize_generator(system_prompt, prompt_template, generator_config)

    # Start MLFlow run
    if mlflow_uri:
        print(f"Using MLFlow URI: {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("RAGs To Riches")
    with mlflow.start_run(run_name=experiment_name):
        # Log config YAMLs
        mlflow.log_artifact(f"{configs_base_dir}/experiment.yaml", artifact_path="configs")
        mlflow.log_artifact(f"{configs_base_dir}/parsers.yaml", artifact_path="configs")
        mlflow.log_artifact(f"{configs_base_dir}/chunker.yaml", artifact_path="configs")
        mlflow.log_artifact(f"{configs_base_dir}/embedder.yaml", artifact_path="configs")
        mlflow.log_artifact(f"{configs_base_dir}/vector_store.yaml", artifact_path="configs")
        mlflow.log_artifact(f"{configs_base_dir}/retriever.yaml", artifact_path="configs")
        mlflow.log_artifact(f"{configs_base_dir}/generator.yaml", artifact_path="configs")
        
        # Check to see if database insertion logged
        if not experiment_config.get("vector_store_exists", False):
            logger.debug(f"Vector store does not exist")
            # Check to see if embeddings exist already
            if not check_for_embeddings(experiment_config["embeddings_dir"]):
                logger.debug(f"Embeddings not found")
                # Loop over data files
                embeddings_to_save = []
                data_path = Path(experiment_config["data"])
                for file_path in data_path.rglob("*"):
                    if file_path.is_file():
                        file_path = str(file_path)
                        logger.debug(f"Working on: {file_path}")
                        parser = parser_router(all_parsers, file_path)
                        parsed_document = parser.parse(file_path)
                        logger.debug(f"Finished parsing")
                        logger.debug(f"Starting chunking")
                        chunks = chunker.chunk(parsed_document)
                        logger.debug(f"Finished chunking")
                        logger.debug(f"Starting embedding")
                        embeddings = embedder.embed_data(chunks)
                        embeddings_to_save += [embedding for embedding in embeddings]
                        logger.debug(f"Finished embedding")
                        
                
                logger.debug(f"Saving embeddings")
                save_embeddings(embeddings_to_save, experiment_config["embeddings_dir"])
                experiment_config["embeddings_saved"] = True
                save_config_yaml(experiment_config, configs_base_dir, "experiment")
            else:
                logger.debug(f"Embeddings found")
                embeddings = load_embeddings(experiment_config["embeddings_dir"])
                logger.debug(f"Loaded embeddings")

            # Save
            logger.debug(f"Pushing to vector store")
            vector_store.store_batch(embeddings, batch_size=10)
            logger.debug(f"Finished pushing to vector store")
            experiment_config["vector_store_exists"] = True
            save_config_yaml(experiment_config, configs_base_dir, "experiment")
        else:
            logger.debug(f"Vector store exists")
        
        # Get queries
        query_pass = experiment_config["query_pass"]["question"]
        query_pass_gold_docs = experiment_config["query_pass"]["relevant_docs"]
        query_fail = experiment_config["query_fail"]["question"]
        query_fail_gold_docs = experiment_config["query_fail"]["relevant_docs"]
        mlflow.log_param("query_pass", query_pass)
        mlflow.log_param("query_fail", query_fail)

        # Retrieve documents
        query_pass_doc_names, query_pass_docs = retriever.retrieve(query_pass)
        query_fail_doc_names, query_fail_docs = retriever.retrieve(query_fail)

        # Generate response
        response_pass = generator.generate(query_pass, query_pass_docs)
        response_fail = generator.generate(query_fail, query_fail_docs)
        
        logger.debug(f"PASS:")
        logger.debug(response_pass)

        logger.debug(f"FAIL:")
        logger.debug(response_fail)

        results = {
            "pass": {
                "query": query_pass,
                "response": json.loads(response_pass),
                "retrieved_doc_names": query_pass_doc_names,
                "retrieved_texts": query_pass_docs,
                "gold_doc_names": query_pass_gold_docs,
                "response": json.loads(response_pass)
            },
            "fail": {
                "query": query_fail,
                "response": json.loads(response_fail),
                "retrieved_doc_names": query_fail_doc_names,
                "retrieved_texts": query_fail_docs,
                "gold_doc_names": query_fail_gold_docs,
                "response": json.loads(response_fail)
            }
        }
        save_results_json(results, f"experiments/results/{experiment_name}/results.json")
        mlflow.log_artifact(f"experiments/results/{experiment_name}/results.json", artifact_path="results")
        mlflow.log_artifact(f"experiments/logs/{experiment_name}/{experiment_name}.log", artifact_path="logs")