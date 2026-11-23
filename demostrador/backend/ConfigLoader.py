from dataclasses import dataclass
from typing import Optional, Dict
from dotenv import load_dotenv
import json
import os

@dataclass
class RetrievalModelConfig:
    embeddings: bool
    model_name: str
    elastic_index: str

@dataclass
class RerankerModelConfig:
    model_name: str

@dataclass
class GenerationModelConfig:
    openai_base_url: str
    openai_api_key: str
    model_name: str
    quantization: bool

@dataclass
class PromptsConfig:
    retrieval_system_prompt: str
    non_retrieval_system_prompt: str
    # Respuesta fija cuando el contexto no contiene la información.
    # El texto generado se corta justo después de ella.
    no_answer_text: str

@dataclass
class ElasticConfig:
    url: str
    username: str
    password: str

@dataclass
class Config:
    # Diccionarios con todos los modelos disponibles
    retrieval_models: Dict[str, RetrievalModelConfig]
    reranker_models: Dict[str, RerankerModelConfig]
    generation_models: Dict[str, GenerationModelConfig]
    
    # Modelos por defecto seleccionados
    default_retriever: str
    default_reranker: str
    default_generator: str
    
    # Configuraciones generales
    num_docs_retrieval: int
    num_docs_reranker: int
    prompts: PromptsConfig

    # Datos que dependen de cada instalación: se leen del fichero .env
    elastic_config: ElasticConfig
    hf_cache_dir: Optional[str]  # None = carpeta por defecto de Hugging Face


class ConfigLoader:
    @staticmethod
    def load(config_path: str) -> Config:
        """
        Carga la configuración desde un archivo JSON.
        
        Args:
            config_path: Ruta al archivo general_config.json
            
        Returns:
            Config: Objeto de configuración completo
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            config_dict = json.load(file)
        
        # Parsear modelos de retrieval
        retrieval_models = {}
        for name, model_data in config_dict['RETRIEVAL_MODELS'].items():
            retrieval_models[name] = RetrievalModelConfig(
                embeddings=model_data['embeddings'].lower() == 'true',
                model_name=model_data['model_name'],
                elastic_index=model_data['elastic_index']
            )
        
        # Parsear modelos de reranking
        reranker_models = {}
        for name, model_data in config_dict['RERANKER_MODELS'].items():
            reranker_models[name] = RerankerModelConfig(
                model_name=model_data['model_name']
            )
        
        # Parsear modelos de generación
        generation_models = {}
        for name, model_data in config_dict['GENERATION_MODELS'].items():
            generation_models[name] = GenerationModelConfig(
                openai_base_url=model_data.get('openai_base_url', ''),
                openai_api_key=model_data.get('openai_api_key', ''),
                model_name=model_data['model_name'],
                quantization=model_data.get('quantization', 'false').lower() == 'true'
            )
        
        # Parsear prompts
        prompts = PromptsConfig(
            retrieval_system_prompt=config_dict['PROMPTS']['retrieval_system_prompt'],
            non_retrieval_system_prompt=config_dict['PROMPTS']['non_retrieval_system_prompt'],
            no_answer_text=config_dict['PROMPTS']['no_answer_text']
        )

        # Crear el objeto de configuración
        config = Config(
            retrieval_models=retrieval_models,
            reranker_models=reranker_models,
            generation_models=generation_models,
            default_retriever=config_dict['DEFAULT_RETRIEVER'],
            default_reranker=config_dict['DEFAULT_RERANKER'],
            default_generator=config_dict['DEFAULT_GENERATOR'],
            num_docs_retrieval=config_dict['NUM_DOCS_RETRIEVAL'],
            num_docs_reranker=config_dict['NUM_DOCS_RERANKER'],
            prompts=prompts,
            elastic_config=ConfigLoader.load_elastic_from_env(),
            hf_cache_dir=os.getenv('HF_CACHE_DIR') or None
        )

        return config

    @staticmethod
    def load_elastic_from_env() -> ElasticConfig:
        """
        Lee la conexión a Elasticsearch del fichero .env (ver .env.example).
        """
        load_dotenv()
        password = os.getenv('ES_PASSWORD')
        if not password:
            raise ValueError("Falta ES_PASSWORD en el fichero .env (ver .env.example).")

        return ElasticConfig(
            url=os.getenv('ES_URL', 'http://localhost:9200'),
            username=os.getenv('ES_USER', 'elastic'),
            password=password
        )


if __name__ == "__main__":
    # Ejemplo de uso
    config_path = "configs/general_config.json"
    config = ConfigLoader.load(config_path)
    
    print("=" * 50)
    print("CONFIGURACIÓN CARGADA")
    print("=" * 50)
    
    print(f"\nHF Cache Dir: {config.hf_cache_dir}")
    print(f"Elasticsearch: {config.elastic_config.url}")
    print(f"Num Docs Retrieval: {config.num_docs_retrieval}")
    print(f"Num Docs Reranker: {config.num_docs_reranker}")
    
    print("\n" + "=" * 50)
    print("MODELOS DISPONIBLES")
    print("=" * 50)
    print("\nRetrievers:", list(config.retrieval_models.keys()))
    print("Rerankers:", list(config.reranker_models.keys()))
    print("Generators:", list(config.generation_models.keys()))

    

