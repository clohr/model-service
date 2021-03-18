import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    neo4j_url = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_BOLT_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_BOLT_PASSWORD", "blackandwhite")
    neo4j_max_connection_lifetime = int(
        os.environ.get("NEO4J_MAX_CONNECTION_LIFETIME", 300)
    )
    log_level = os.environ.get("LOG_LEVEL")
    environment = os.environ.get("ENVIRONMENT", "dev")
