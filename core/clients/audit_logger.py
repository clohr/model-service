import connexion  # type: ignore
from audit_middleware import Auditor, TraceId  # type: ignore
from flask import current_app

from server.errors import MissingTraceId


class AuditLogger:
    @classmethod
    def trace_id_header(cls) -> TraceId:
        x_bf_trace_id = connexion.request.headers.get(Auditor.TRACE_ID_HEADER)
        if x_bf_trace_id is None:
            raise MissingTraceId
        return TraceId(x_bf_trace_id)

    @classmethod
    def get(cls) -> Auditor:
        """
        Get a audit logger instance from the Flask app config.
        """
        return current_app.config["audit_logger"]
