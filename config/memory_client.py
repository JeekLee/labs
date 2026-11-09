import os

from dotenv import load_dotenv
from neo4j_agent_memory import ExtractionConfig, ExtractorType, MemorySettings, Neo4jConfig
from neo4j_agent_memory.config.settings import MergeStrategy
from pydantic import SecretStr


def load_neo4j_config() -> Neo4jConfig:
    if not (uri := os.getenv("NEO4J_URI")): raise ValueError("NEO4J_URI environment variable not set")
    if not (username := os.getenv("NEO4J_USERNAME")): raise ValueError("NEO4J_USERNAME environment variable not set")
    if not (password := os.getenv("NEO4J_PASSWORD")): raise ValueError("NEO4J_PASSWORD environment variable not set")
    if not (database := os.getenv("NEO4J_DATABASE")): raise ValueError("NEO4J_DATABASE environment variable not set")

    return Neo4jConfig(
        uri=uri,
        username=username,
        password=SecretStr(password),
        database=database,
    )

def load_extraction_config() -> ExtractionConfig:
    return ExtractionConfig(
        extractor_type=ExtractorType.PIPELINE,

        enable_spacy=False,
        enable_gliner=True,
        enable_llm_fallback=True,

        merge_strategy=MergeStrategy.CONFIDENCE,
        fallback_on_empty=True,

        gliner_model="fastino/gliner2.5-base-v1",
        gliner_threshold=0.2,
        gliner_device="cpu",

        llm_model="gpt-5.4-mini",

        entity_types=["PERSON", "LOCATION", "ORGANIZATION", "EVENT", "OBJECT"],

        extract_relations=True,
        extract_preferences=True,
        confidence_threshold=0.0
    )

def load_memory_settings() -> MemorySettings:
    load_dotenv()
    return MemorySettings(
        neo4j=load_neo4j_config(),
        extraction=load_extraction_config(),
        llm="gpt-5.4-mini",
        embedding="openai/text-embedding-3-small"
    )

def print_memory_settings(settings: MemorySettings):
    print("1. Extraction Settings")
    print(f"  - spaCy: {'✓' if settings.extraction.enable_spacy else '✗'}")
    print(f"  - GLiNER: {'✓' if settings.extraction.enable_gliner else '✗'}")
    print(f"  - LLM Fallback: {'✓' if settings.extraction.enable_llm_fallback else '✗'}")
    print(f"  - Merge Strategy: {settings.extraction.merge_strategy}")

    print("2. Memory Settings")
    print(f"  - llm: {settings.llm}")
    print(f"  - embedding: {settings.embedding}")