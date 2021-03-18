import os
from dataclasses import dataclass

from auth_middleware import JwtConfig  # type: ignore
from flask import current_app

from core import Config as CoreConfig


@dataclass(frozen=True)
class Config(CoreConfig):
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    service_name = os.environ.get("SERVICE_NAME", "model-service")
    jwt_secret_key: str = os.environ.get("JWT_SECRET_KEY", "test-key")
    pennsieve_api_host: str = os.environ.get("PENNSIEVE_API_HOST", "api.pennsieve.net")
    gateway_internal_host: str = os.environ.get(
        "GATEWAY_INTERNAL_HOST", "api.pennsieve.net"
    )
    max_record_count_for_property_deletion: int = int(
        os.environ.get("MAX_RECORD_COUNT_FOR_PROPERTY_DELETION", 100000)
    )
    victor_ops_url: str = os.environ.get(
        "VICTOR_OPS_URL",
        "https://alert.victorops.com/integrations/generic/20131114/alert/a0e9a781-de39-4e2a-98be-111fae93d247",
    )
    jobs_sqs_queue_id: str = os.environ.get("JOBS_SQS_QUEUE_ID", "jobs-sqs-queue-id")

    @property
    def jwt_config(self) -> JwtConfig:
        return JwtConfig(self.jwt_secret_key)

    @classmethod
    def from_app(cls) -> "Config":
        """
        Get the currently loaded config for a request from the Flask app context.
        """
        return current_app.config["config"]
