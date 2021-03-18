import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import boto3  # type: ignore
import connexion  # type: ignore
import prance  # type: ignore
import structlog  # type: ignore
from audit_middleware import AuditLogger, Auditor, GatewayHost  # type: ignore
from connexion import App  # type: ignore
from connexion.apps.flask_app import FlaskJSONEncoder  # type: ignore
from connexion.exceptions import OAuthProblem  # type: ignore
from jwt.exceptions import ExpiredSignatureError  # type: ignore

from core.clients import PennsieveApiClient, PennsieveJobsClient, VictorOpsClient
from core.util import is_datetime, normalize_datetime

from . import errors
from .config import Config
from .db import Database

logger = structlog.get_logger(__name__)


class CustomizedEncoder(FlaskJSONEncoder):
    def default(self, obj):
        if is_datetime(obj):
            return normalize_datetime(obj).isoformat()
        else:
            return super().default(obj)


def get_error_context() -> List[str]:
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_type is not None and exc_value is not None and exc_tb is not None:
        tbe = traceback.TracebackException(exc_type, exc_value, exc_tb)
        return list(tbe.format())
    return []


def create_app(
    config: Config = None,
    db: Database = None,
    api_client: PennsieveApiClient = None,
    jobs_client: PennsieveJobsClient = None,
    audit_logger: Auditor = None,
    victor_ops_client: VictorOpsClient = None,
):
    app = App(__name__)

    health = bundled("health.yml")
    internal = bundled("model-service-internal.yml")
    api_v1 = bundled("model-service-v1.yml")
    api_v2 = bundled("model-service-v2.yml")
    api_v2_streaming = bundled("model-service-streaming-v2.yml")

    app.add_api(
        internal, validate_responses=True, pythonic_params=True, base_path="/internal"
    )
    app.add_api(api_v1, validate_responses=True, pythonic_params=True, base_path="/v1")
    app.add_api(api_v2, validate_responses=True, pythonic_params=True, base_path="/v2")

    app.app.json_encoder = CustomizedEncoder

    # Mount the v1 API again with no `v1/` prefix. Ideally this would rewritten
    # in the gateway, but internal services need to be updated to us the `/v1`
    # prefix first. This needs to be merged with `health` so that these routes
    # can share the same base path.
    #
    # See ticket: https://app.clickup.com/t/5mcufd
    root_api = {}
    root_api.update(api_v1)
    root_api["paths"].update(health["paths"])
    app.add_api(root_api, validate_responses=True, pythonic_params=True, base_path="/")

    # Unfortunately the only way to stream responses with connexion is to turn
    # response validation off.
    app.add_api(
        api_v2_streaming,
        validate_responses=False,
        pythonic_params=True,
        base_path="/v2/organizations",
    )

    @app.app.errorhandler(ValueError)
    def handle_value_error(error):
        stacktrace = get_error_context()
        return (
            dict(message=str(error), stacktrace=stacktrace),
            400,
            {"Content-Type": "application/json"},
        )

    @app.app.errorhandler(NotImplementedError)
    def handle_not_implemented_error(error):
        stacktrace = get_error_context()
        return (
            dict(message=str(error), stacktrace=stacktrace),
            415,
            {"Content-Type": "application/json"},
        )

    @app.app.errorhandler(errors.ExternalRequestError)
    def handle_external_request_failure(error):
        return (
            dict(message=str(error)),
            500,
            {"Content-Type": "application/json"},
        )

    @app.app.errorhandler(errors.MissingTraceId)
    @app.app.errorhandler(errors.ModelServiceError)
    @app.app.errorhandler(errors.OperationError)
    @app.app.errorhandler(errors.InvalidOrganizationError)
    @app.app.errorhandler(errors.InvalidDatasetError)
    def handle_service_error(error):
        return error.to_json(), 400, {"Content-Type": "application/json"}

    @app.app.errorhandler(ExpiredSignatureError)
    @app.app.errorhandler(OAuthProblem)
    def handle_auth_error(error):
        return dict(message=str(error)), 401, {"Content-Type": "application/json"}

    @app.app.errorhandler(errors.RecordRelationshipNotFoundError)
    @app.app.errorhandler(errors.LegacyModelRelationshipNotFoundError)
    @app.app.errorhandler(errors.ModelRelationshipNotFoundError)
    @app.app.errorhandler(errors.ModelNotFoundError)
    def handle_not_found(error):
        return error.to_json(), 404, {"Content-Type": "application/json"}

    @app.app.errorhandler(errors.PackageProxyNotFoundError)
    def handle_proxy_package_not_found(error):
        return error.to_json(), 404, {"Content-Type": "application/json"}

    @app.app.errorhandler(errors.ExceededTimeLimitError)
    def handle_operation_timed_out(error):
        return error.to_json(), 408, {"Content-Type": "application/json"}

    @app.app.errorhandler(errors.ModelPropertyInUseError)
    def handle_model_property_in_use(error):
        return error.to_json(), 422, {"Content-Type": "application/json"}

    @app.app.errorhandler(errors.LockedDatasetError)
    def handle_locked_dataset(error):
        return error.to_json(), 423, {"Content-Type": "application/json"}

    if config is None:
        config = Config()
    app.app.config["config"] = config

    if db is None:
        db = Database.from_config(config)
    app.app.config["db"] = db

    if api_client is None:
        api_client = PennsieveApiClient(config.pennsieve_api_host)
    app.app.config["api_client"] = api_client

    if jobs_client is None:
        sqs_client = boto3.client("sqs", region_name=config.aws_region)
        jobs_client = PennsieveJobsClient(sqs_client, config.jobs_sqs_queue_id)
    app.app.config["jobs_client"] = jobs_client

    if victor_ops_client is None:
        victor_ops_client = VictorOpsClient(
            config.victor_ops_url, f"{config.environment}-data-management"
        )
    app.app.config["victor_ops_client"] = victor_ops_client

    if audit_logger is None:
        audit_logger = AuditLogger(GatewayHost(config.gateway_internal_host))
    app.app.config["audit_logger"] = audit_logger

    app.app.after_request(log_request)

    return app


def log_request(response):
    """
    Log all requests except for health checks.
    """
    if connexion.request.path == "/health":
        return response

    log = logger.bind(
        method=connexion.request.method,
        status_code=response.status_code,
        path=connexion.request.full_path,
        endpoint=connexion.request.endpoint,
    )

    if response.status_code >= 400 and connexion.request.data:
        body = f"body={connexion.request.json}"
    else:
        body = ""

    log.info(
        f"{connexion.request.method} {response.status_code} {connexion.request.full_path} {body}"
    )

    return response


def bundled(spec_file: str) -> Dict[str, Any]:
    """
    Resolve multi-file OpenAPI specifications.

    See https://github.com/zalando/connexion/issues/254
    """
    path = Path(__file__, "../../openapi", spec_file).resolve().absolute()

    parser = prance.ResolvingParser(str(path), strict=True)
    parser.parse()
    return parser.specification
